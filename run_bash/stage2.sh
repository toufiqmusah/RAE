wandb login 
export WANDB_LOGIN=0
export ENTITY="brats_dann"
export PROJECT="MEDRAE"


N=4
CONFIG=configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml
DATA_PATH=../BiomedParseDataRAE/train
RESULTS_DIR=results_medrae/stage2


torchrun --standalone --nproc_per_node=$N \
  src/train.py \
  --config $CONFIG \
  --data-path $DATA_PATH \
  --results-dir $RESULTS_DIR \
  --image-size 256 \
  --precision fp32 \
  --wandb