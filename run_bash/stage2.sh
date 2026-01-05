conda activate rae
wandb login 
export WANDB_LOGIN=0
export ENTITY="brats_dann"
export PROJECT="MEDRAE"


N=4
CONFIG=configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml
DATA_PATH=../BiomedParseDataRAE/train
RESULTS_DIR=results_medrae/stage2
CKPT_PATH=/pscratch/sd/j/jehr/MEDRAE/RAE/results_medrae/stage2/010-DiTwDDTHead-Linear-velocity-none-acc4/checkpoints/0018000.pt

torchrun --standalone --nproc_per_node=$N \
  src/train.py \
  --config $CONFIG \
  --data-path $DATA_PATH \
  --results-dir $RESULTS_DIR \
  --image-size 256 \
  --precision fp32 \
  --ckpt $CKPT_PATH \
  --wandb




N=4
CONFIG=configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B-ft.yaml
DATA_PATH=../BiomedParseDataRAE/train
RESULTS_DIR=results_medrae/stage2_finetune
CKPT_PATH=/pscratch/sd/j/jehr/MEDRAE/RAE/models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-S_ep20/stage2_model.pt

torchrun --standalone --nproc_per_node=$N \
  src/train.py \
  --config $CONFIG \
  --data-path $DATA_PATH \
  --results-dir $RESULTS_DIR \
  --image-size 256 \
  --precision fp32 \
  --ckpt $CKPT_PATH \
  --wandb