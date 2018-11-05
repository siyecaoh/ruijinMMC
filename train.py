import bilstm_crf_model
import argparse
from utils import *
from keras.callbacks import ModelCheckpoint
import keras

lr_base = 0.001
epochs = 250
lr_power = 0.9
def lr_scheduler(epoch, mode='adam'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

    if mode is 'power_decay':
        # original lr scheduler
        lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.0001
        elif epoch > 0.75 * epochs:
            lr = 0.001
        elif epoch > 0.5 * epochs:
            lr = 0.01
        else:
            lr = 0.1

    print('lr: %f' % lr)
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN')
    parser.add_argument('--num', type=int)
    parser.add_argument('--embed', type=int)
    parser.add_argument('--units', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--save', type=str)
    parser.add_argument('--batch', type=int, default=2)

    args = parser.parse_args()
    args.num = 0
    args.gpu = 0

    args.epoch = 200
    args.batch = 64

    args.embed = 200
    args.units = 200
    gpu_config(args.gpu)
    

    model, (train_x, train_y), (test_x, test_y) = bilstm_crf_model.create_model(args.embed, args.units)
    # used for multi checkpoints to vote
    #filepath = args.save+'/weights-improvement-{epoch:02d}-{val_acc:.4f}.h5'
    
    # only get the best single model
    filepath = args.save+'/model.h5'
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)

    model.fit(train_x, train_y,batch_size=args.batch,epochs=args.epoch,
              validation_data=[test_x, test_y],callbacks=[checkpoint,scheduler])
    
