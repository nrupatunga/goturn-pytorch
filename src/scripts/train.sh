# main script to train

IMAGENET_PATH='/media/nthere/datasets/ISLVRC2014_Det/'
ALOV_PATH='/media/nthere/datasets/ALOV/'
MEAN_FILE='../goturn/dataloaders/imagenet.mean.npy'
SAVE_PATH='./caffenet-dbg-2/'
PRETRAINED_MODEL_PATH='../goturn/models/pretrained/caffenet_weights.npy'

python train.py \
--imagenet_path $IMAGENET_PATH \
--alov_path $ALOV_PATH \
--save_path $SAVE_PATH \
--pretrained_model $PRETRAINED_MODEL_PATH
