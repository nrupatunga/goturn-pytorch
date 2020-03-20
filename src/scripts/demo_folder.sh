MODEL_DIR='./caffenet-dbg-2/lightning_logs/version_1/'
#FOLDER='/media/nthere/datasets/vot/ball/'
MEAN_FILE='../goturn/dataloaders/imagenet.mean.npy'
FOLDER='/media/nthere/datasets/goturn/2'

python demo_folder.py --model_dir $MODEL_DIR\
	--input $FOLDER \
	--mean_file $MEAN_FILE \
