import math
import torch
import torch.nn.functional as F
import unittest
from UNAT import UNAT

# Import your UNAT model definition
# from your_model_module import UNAT

def compute_spectral_norm(weight: torch.Tensor) -> float:
    """
    Computes the spectral norm (largest singular value) of a weight tensor.
    For convolutional weights, reshapes to 2D (out_channels, -1).
    """
    weight_mat = weight.view(weight.shape[0], -1)
    # Use torch.linalg.svdvals to get singular values and take the maximum
    singular_values = torch.linalg.svdvals(weight_mat)
    return singular_values.max().item()

class TestModelStability(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize your model with training hyperparameters
        self.model = UNAT(
            in_channels=4,
            out_channels=3,
            down_channels=[384, 768, 1152],
            up_channels=[768, 384],
            down_strides=[(2,1,1), (1,2,2), (1,2,2)],  
            down_block_depths=[4, 4, 4],
            down_kernel_sizes=[(5,5), (5,5), (5,5)],
            up_strides=[(1,2,2), (1,2,2)],
            up_block_depths=[4,4],
            up_kernel_sizes=[(5,5),(5,5)],
            norm=None,
            layer_scale=0.5,
            mlp_ratio=4,
            num_blocks=1,
            skip_type='layer_scale',
            skip_down_levels=[0, 1],
            skip_up_levels=[0, 1],
            in_steps=2,
            resolution=1.0,
            final_conv=True,
            emb_method="spherical_rope",
            downsample_type=["conv", "avgpool", "avgpool"],
            upsample_type="interp",
            interp_mode="nearest"
        ).to(self.device)
        self.model.train()  # Ensure we are in training mode

    def _run_dummy_step(self):
        """
        Generates dummy inputs matching the model's expected dimensions.
          - x: [B, 3, 2, H, W]
          - inv: [B, 1, 3, H, W]
          - grid: [B, H, W, 2]
          - target: [B, 3, 1, H, W]
        """
        B, H, W = 2, 64, 64
        x = torch.randn(B, 3, 2, H, W, device=self.device)
        inv = torch.randn(B, 1, 3, H, W, device=self.device)
        grid = torch.randn(B, H, W, 2, device=self.device)
        target = torch.randn(B, 3, 1, H, W, device=self.device)
        return x, inv, grid, target

    def _check_gradients(self, step: int = 0, threshold: float = 1e4):
        """
        Checks each parameter's gradient to ensure that there are no NaN/Inf values
        and that the gradient norm does not exceed a specified threshold.
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} is NaN")
                self.assertFalse(torch.isinf(param.grad).any(), f"Gradient for {name} is Inf")
                self.assertLessEqual(
                    grad_norm, threshold,
                    f"Gradient norm for {name} ({grad_norm}) exceeds threshold {threshold} at step {step}"
                )

    def test_single_forward_backward(self):
        """
        Runs a single forward/backward pass with dummy data and checks gradients.
        """
        x, inv, grid, target = self._run_dummy_step()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        output = self.model(x, inv, grid)
        loss = F.mse_loss(output, target)
        loss.backward()
        self._check_gradients()

    def test_multiple_steps_gradient_stability(self):
        """
        Runs multiple forward/backward passes to ensure that gradients remain stable over several updates.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        num_steps = 50
        for step in range(num_steps):
            x, inv, grid, target = self._run_dummy_step()
            optimizer.zero_grad()
            output = self.model(x, inv, grid)
            loss = F.mse_loss(output, target)
            loss.backward()
            self._check_gradients(step)
            optimizer.step()

    def test_spectral_norm_analysis(self):
        """
        Iterates over modules with weights and computes their spectral norms.
        Fails the test if any spectral norm exceeds a defined threshold.
        """
        spectral_norm_threshold = 10.0  # Adjust based on your initialization and architecture
        for name, module in self.model.named_parameters():
            if hasattr(module, 'weight') and module.weight is not None:
                norm_val = compute_spectral_norm(module.weight)
                print(f"Spectral norm for {name}: {norm_val:.4f}")
                self.assertLessEqual(
                    norm_val, spectral_norm_threshold,
                    f"Spectral norm of {name} exceeds threshold: {norm_val}"
                )
            else:
                norm_val = compute_spectral_norm(module)
                print(f"Spectral norm for {name}: {norm_val:.4f}")

    def test_variance_propagation_analysis(self):
        """
        Attaches forward hooks on selected modules to record the mean and variance of activations.
        Checks that the variances are neither vanishing nor exploding.
        """
        activations = {}

        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Record the mean and variance of the activation
                    activations[name] = {
                        "mean": output.mean().item(),
                        "var": output.var().item()
                    }
            return hook

        # Choose modules of interest: convolution layers and NAT blocks.
        hook_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                h = get_hook(name)
                module.register_forward_hook(h)
                hook_modules.append((name, module))
            elif module.__class__.__name__ == "NATBlock2D":
                h = get_hook(name)
                module.register_forward_hook(h)
                hook_modules.append((name, module))

        # Run a forward pass with dummy data.
        x, inv, grid, target = self._run_dummy_step()
        _ = self.model(x, inv, grid)

        # Define thresholds for activation variance.
        # (These thresholds are examples and might need tuning.)
        lower_var_threshold = 1e-4   # Too low may indicate vanishing activations.
        upper_var_threshold = 100.0    # Too high may indicate exploding activations.
        for name, stats in activations.items():
            print(f"Activation stats for {name}: mean={stats['mean']:.4f}, var={stats['var']:.4f}")
            self.assertFalse(math.isnan(stats["var"]), f"Activation variance for {name} is NaN")
            self.assertFalse(math.isinf(stats["var"]), f"Activation variance for {name} is Inf")
            self.assertGreater(
                stats["var"], lower_var_threshold,
                f"Activation variance for {name} is too low (vanishing activations): {stats['var']}"
            )
            self.assertLess(
                stats["var"], upper_var_threshold,
                f"Activation variance for {name} is too high (exploding activations): {stats['var']}"
            )

        # Clean up hooks if needed
        for name, module in hook_modules:
            module._forward_hooks.clear()

if __name__ == "__main__":
    unittest.main()