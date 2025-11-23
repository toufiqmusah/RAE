
from torch import nn
import torch
from math import *
from . import register_encoder
from transformers import AutoModel


@register_encoder()
class MedSigLIPwNorm(nn.Module):
	def __init__(self, model_name: str, num_tokens: int = 256):
		super().__init__()
		self.model_name = model_name
		self.num_tokens = num_tokens

		# Use AutoModel to support different SigLIP/MedSigLIP model classes
		model = AutoModel.from_pretrained(self.model_name)

		# Some HF wrappers put the vision transformer under `.vision_model`,
		# others expose it directly. Prefer `.vision_model` when present.
		self.model = getattr(model, "vision_model", model)

		# Try to remove the affine parameters of the final layernorm if present.
		# Different implementations use different attribute names; try common ones.
		for ln_name in ("post_layernorm", "layernorm", "ln", "post_ln"):
			ln = getattr(self.model, ln_name, None)
			if ln is not None:
				try:
					ln.elementwise_affine = False
				except Exception:
					pass
				# Remove weight/bias tensors if present so they are not used.
				if hasattr(ln, "weight"):
					ln.weight = None
				if hasattr(ln, "bias"):
					ln.bias = None
				break

		# Expose config-derived values used elsewhere in the codebase
		self.hidden_size = getattr(self.model.config, "hidden_size", None)
		self.patch_size = getattr(self.model.config, "patch_size", None)

	@torch.no_grad()
	def forward(self, images: torch.Tensor) -> torch.Tensor:
		"""
		images: (B, C, H, W)
		Returns image token embeddings (including possible CLS tokens depending on model).
		"""
		# Many vision models accept `output_hidden_states` and `interpolate_pos_encoding`.
		# Use kwargs if supported; otherwise fall back to a plain call.
		try:
			outputs = self.model(images, output_hidden_states=True, interpolate_pos_encoding=True)
		except TypeError:
			outputs = self.model(images, output_hidden_states=True)

		# Most HF vision models return `last_hidden_state`.
		image_features = getattr(outputs, "last_hidden_state", outputs)
		return image_features
