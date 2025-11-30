import os
import torch
from huggingface_hub import snapshot_download
from transformers import (
    Dinov2WithRegistersModel,
    ViTMAEForPreTraining,
    SiglipModel,
    AutoImageProcessor
)
from src.disc.lpips_utils import get_ckpt_path  # Triggers LPIPS download

def download_rae_artifacts():
    """
    Downloads the official RAE collection (Decoders, Stats, DiT weights)
    into the local 'models/' directory.
    """
    print("\n[1/3] Downloading RAE Collection (Decoders, Stats, Discriminators)...")
    snapshot_download(
        repo_id="nyu-visionx/RAE-collections",
        local_dir="models",
        local_dir_use_symlinks=False,
        repo_type="model"
    )
    print("✅ RAE Artifacts downloaded to ./models")


def cache_public_encoders():
    """
    Pre-downloads the frozen encoder backbones from Hugging Face
    so training doesn't hang waiting for downloads.
    """
    print("\n[2/3] Caching Public Encoders (DINOv2, SigLIP2, MAE, MedSigLIP)...")
    
    encoders_to_download = [
        # Model ID                              # Model Class
        ("facebook/dinov2-with-registers-base",  Dinov2WithRegistersModel),
        # ("facebook/vit-mae-base",                ViTMAEForPreTraining),
        ("google/siglip2-base-patch16-256",      SiglipModel),
        ("google/medsiglip-448",                 SiglipModel), 
        ("facebook/dinov3-vitl16-pretrain-lvd1689m", AutoImageProcessor)
    ]

    for model_id, model_class in encoders_to_download:
        print(f"   ... fetching {model_id}")  
        model_class.from_pretrained(model_id)
        AutoImageProcessor.from_pretrained(model_id)

def setup_lpips():
    """
    Ensures the LPIPS VGG weights are downloaded.
    The code typically does this lazily, but we do it now to be safe.
    """
    print("\n[3/3] Verifying LPIPS weights...")
    path = get_ckpt_path("vgg_lpips")
    print(f"✅ LPIPS weights located at: {path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    download_rae_artifacts()    
    cache_public_encoders()    
    setup_lpips()
    
    print("\nAll models downloaded ...")