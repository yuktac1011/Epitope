import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple

class GradCAMExplainer:
    """
    Generates Grad-CAM visual heatmaps for MobileNetV3 artifact detector.
    Highlights facial regions that influenced the synthetic/manipulation score.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else self.model.features[-1]
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """
        Generates a 2D normalized heatmap [0.0, 1.0] for the target class_idx.
        class_idx = 0 corresponds to Synthetic / Fake.
        """
        self.model.zero_grad()
        # Enable gradient calculation even if model was in eval mode
        with torch.enable_grad():
            tensor = tensor.clone().detach().requires_grad_(True)
            output = self.model(tensor)
            score = output[0, class_idx]
            score.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        # ReLU to keep only positive influence
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_on_image(
        self,
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Image.Image:
        """
        Overlays the 2D heatmap on the PIL face crop image.
        """
        w, h = pil_image.size
        # Resize heatmap to match image dimensions
        resized_cam = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * resized_cam)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        np_orig = np.array(pil_image)
        overlay = np.uint8(alpha * heatmap_colored + (1.0 - alpha) * np_orig)
        return Image.fromarray(overlay)
