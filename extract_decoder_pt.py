import torch
import argparse
from pathlib import Path



def main(args):
    full_ckpt = torch.load(args.ckpt_pt, map_location="cpu")

    model_sd = full_ckpt["model"]

    # Strip the "decoder." prefix to match model.pt format
    decoder_sd = {
        k.replace("decoder.", ""): v
        for k, v in model_sd.items()
        if k.startswith("decoder.")
    }

    assert len(decoder_sd) > 0, "No decoder parameters found in checkpoint."


    print(f"Saved decoder-only checkpoint to {args.save_path}")
    print(f"Number of decoder parameters: {len(decoder_sd)}")
    print("Example keys:")
    for k in list(decoder_sd.keys())[:10]:
        print(" ", k)

    if args.ref_ckpt:
        ref_sd = torch.load(args.ref_ckpt, map_location="cpu")

        ref_keys = set(ref_sd.keys())
        dec_keys = set(decoder_sd.keys())

        missing = sorted(ref_keys - dec_keys)
        extra = sorted(dec_keys - ref_keys)

        if missing or extra:
            if missing:
                print(f"Missing keys ({len(missing)}):")
                for k in missing[:20]:
                    print("  ", k)
            if extra:
                print(f"Extra keys ({len(extra)}):")
                for k in extra[:20]:
                    print("  ", k)

            raise RuntimeError('Missing or extra keys present.')
        else:
            print('Keys in decoder match reference checkpoint exactly.')

    torch.save(decoder_sd, args.save_path)








if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reconstruct an input image using a Stage-1 RAE loaded from config."
    )
    parser.add_argument(
        "--ckpt_pt",
        required=True,
        help="Path to the checkpoint .pt file from an intermediate training step.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Path to save the decoder part checkpoint as a .pt file to be compatible with inference and stage 2 models.",
    )

    parser.add_argument(
        "--ref_ckpt",
        type=Path,
        required=False,
        default='models/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt',
        help="Path to an existing decode checkpoint with the correct keys expected. Useful for checking if all expected keys are present"
    )

    args = parser.parse_args()
    main(args)



# ckpt_pt = '/pscratch/sd/j/jehr/MEDRAE/RAE/results_medrae/stage1/005-RAE/checkpoints/0010500.pt'
# save_path = '/pscratch/sd/j/jehr/MEDRAE/RAE/models/decoders/medrae/model_0010500.pt'


