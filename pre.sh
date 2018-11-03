

NUM=0
GPU=0
EMBED=200
UNITS=200


DIR=expr/$NUM
python predict.py --embed $EMBED --units $UNITS --num $NUM --gpu $GPU | tee $DIR/predict_log.txt


