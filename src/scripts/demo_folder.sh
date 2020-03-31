MODEL_DIR='./caffenet-dbg-2/lightning_logs/version_1/'
#MODEL_DIR='/home/nthere/2020/pytorch-goturn/src/scripts/caffenet-dbg-2/lightning_logs/version_0/'
#FOLDER='/media/nthere/datasets/vot/ball/'
#MEAN_FILE='../goturn/dataloaders/imagenet.mean.npy'
FOLDER='/media/nthere/datasets/goturn/surfer/imgs'

python demo_folder.py --model_dir $MODEL_DIR\
	--input $FOLDER \
