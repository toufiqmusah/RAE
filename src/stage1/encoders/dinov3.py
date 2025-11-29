from transformers import AutoModel
from torch import nn
import torch
from . import register_encoder


@register_encoder()
class Dinov3withNorm(nn.Module):
    def __init__(
        self,
        dinov3_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # DINOv3 uses AutoModel instead of Dinov2WithRegistersModel
        try:
            self.encoder = AutoModel.from_pretrained(dinov3_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = AutoModel.from_pretrained(dinov3_path, local_files_only=False)
        
        self.encoder.requires_grad_(False)
        
        if normalize:
            # Check if layernorm exists and configure it
            if hasattr(self.encoder, 'layernorm'):
                self.encoder.layernorm.elementwise_affine = False
                self.encoder.layernorm.weight = None
                self.encoder.layernorm.bias = None
            elif hasattr(self.encoder, 'norm'):
                self.encoder.norm.elementwise_affine = False
                self.encoder.norm.weight = None
                self.encoder.norm.bias = None
        
        # Get config attributes
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, output_hidden_states=True)
        
        # DINOv3 might not use register tokens, adjust accordingly
        # Check the actual output structure of your DINOv3 model
        if hasattr(outputs, 'last_hidden_state'):
            # Remove CLS token (and register tokens if present)
            # For DINOv3, you may need to check if it uses register tokens
            unused_token_num = 1  # Just CLS token, adjust if needed
            image_features = outputs.last_hidden_state[:, unused_token_num:]
        else:
            # Fallback if structure is different
            image_features = outputs[0][:, 1:]  # Remove CLS token
        
        return image_features