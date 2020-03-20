MODEL_DIR='./lightning_logs/version_0/'
IMAGENET_PATH='/media/nthere/datasets/ISLVRC2014_Det/'
ALOV_PATH='/media/nthere/datasets/ALOV/'

python test.py --model_dir $MODEL_DIR\
	--imagenet_path $IMAGENET_PATH \
	--alov_path $ALOV_PATH
