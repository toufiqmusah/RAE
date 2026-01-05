
# train stage 1 continue
/pscratch/sd/j/jehr/MEDRAE/RAE/src/train_stage1.py --config configs/stage1/training/DINOv2-B_decXL_continue.yaml --data-path ../BiomedParseDataRAE/train --results-dir results_medrae_continue/stage1 --image-size 256 --precision fp32 --wandb


# train stage 1 finetune
/pscratch/sd/j/jehr/MEDRAE/RAE/src/train_stage1.py --config configs/stage1/training/DINOv2-B_decXL_finetune.yaml --data-path ../BiomedParseDataRAE/train --results-dir results_medrae_finetune/stage1 --image-size 256 --precision fp32 --wandb
N=4
CONFIG=configs/stage1/training/DINOv2-B_decXL_finetune.yaml
DATA_PATH=../BiomedParseDataRAE/train
RESULTS_DIR=results_medrae_finetune/stage1



N=4
CONFIG=configs/stage1/training/DINOv2-B_decXL_finetune_lower_lr.yaml
DATA_PATH=../BiomedParseDataRAE/train
RESULTS_DIR=results_medrae_finetune_lower_lr_v2/stage1


torchrun --standalone --nproc_per_node=$N   src/train_stage1.py   --config $CONFIG   --data-path $DATA_PATH   --results-dir $RESULTS_DIR   --image-size 256   --precision fp32   --wandb --latest


# train stage 1 from scratch
/pscratch/sd/j/jehr/MEDRAE/RAE/src/train_stage1.py --config configs/stage1/training/DINOv2-B_decXL.yaml --data-path ../BiomedParseDataRAE/train --results-dir results_medrae/stage1 --image-size 256 --precision fp32 --wandb --ckpt /pscratch/sd/j/jehr/MEDRAE/RAE/results_medrae/stage1/-01-RAE/checkpoints/0005000.pt


# sample stage 1 baseline
python src/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B_512.yaml \
  --image assets/benign5_ultrasound_breast.png \
  --output assets/benign5_ultrasound_breast_recon.png \


python src/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B_512.yaml \
  --image assets/acdc_patient101_frame01_2_MRI_heart.png \
  --output assets/acdc_patient101_frame01_2_MRI_heart_recon.png \


  python src/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B_512.yaml \
  --image assets/patient007_frame07_2_MRI_heart.png \
  --output assets/patient007_frame07_2_MRI_heart_recon.png \


# sample stage 1 medrae continue
python src/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B-MED.yaml \
  --image assets/acdc_patient101_frame01_2_MRI_heart.png \
  --output assets/acdc_patient101_frame01_2_MRI_heart_recon-med-ft.png \

  python src/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B-MED.yaml \
  --image assets/acdc_patient101_frame01_2_MRI_heart.png \
  --output assets/acdc_patient101_frame01_2_MRI_heart_recon-med-continue.png \


  python src/stage1_sample.py \
  --config configs/stage1/pretrained/DINOv2-B-MED.yaml \
  --image assets/patient007_frame07_2_MRI_heart.png \
  --output assets/patient007_frame07_2_MRI_heart_recon-med-ft.png \



  # torchrun --standalone --nproc_per_node=$N \
  # src/train_stage1.py \
  # --config $CONFIG \
  # --data-path $DATA_PATH \
  # --results-dir $RESULTS_DIR \
  # --image-size 256 \
  # --precision fp32 \
  # --wandb \
  # --latest


python3 extract_decoder_pt.py --ckpt_pt /pscratch/sd/j/jehr/MEDRAE/RAE/results_medrae_finetune/stage1/latest.pt --save_path models/decoders/medrae/model_finetune_9600.pt

