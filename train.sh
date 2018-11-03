NUM=0
GPU=0

EPOCH=1000
BATCH=96

EMBED=200
UNITS=200

DIR=expr/$NUM
if [ ! -d $DIR ];then
	mkdir $DIR
fi
python train.py --embed $EMBED --units $UNITS --num $NUM --epoch $EPOCH --gpu $GPU --batch $BATCH --save $DIR | tee $DIR/train_log.txt
