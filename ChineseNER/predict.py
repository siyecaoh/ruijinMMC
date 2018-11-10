# encoding=utf8
import os
import codecs
import pickle
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

flags = tf.app.flags
flags.DEFINE_boolean("clean", False, "clean train folder")
flags.DEFINE_boolean("train", False, "Wither train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim", 50, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 200, "Embedding size for characters")
flags.DEFINE_integer("lstm_dim", 200, "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip", 5, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 96, "batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", True, "Wither lower case")

flags.DEFINE_integer("max_epoch", 10000, "maximum training epochs")
flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")

flags.DEFINE_string("summary_path", "summary", "Path to store summaries")
flags.DEFINE_string("log_file", "train.log", "File for log")
flags.DEFINE_string("map_file", "maps.pkl", "file for maps")
flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")
flags.DEFINE_string("config_file", "config_file", "File for config")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", "result", "Path for results")
flags.DEFINE_string("emb_file", "wiki_100.utf8", "Path for pre_trained embedding")
# flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
# flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
# flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")

flags.DEFINE_string("train_file", os.path.join("data", "ruijin_train.data"), "Path for train data")
flags.DEFINE_string("dev_file", os.path.join("data", "ruijin_dev.data"), "Path for dev data")
flags.DEFINE_string("test_file", os.path.join("data", "ruijin_dev.data"), "Path for test data")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]

flags.DEFINE_string("ckpt_path", "ckpt", "Path to save model")


def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        test_file_path = "/home/leo/Downloads/MMC/ruijin_round1_test_a_20181022/"
        all_files = os.listdir(test_file_path)
        all_num = len(all_files)
        idx_cnt = 0
        for file in all_files:
            idx_cnt += 1
            if os.path.splitext(file)[1] == ".txt":
                # if file != "128_20_10.txt":
                #     continue
                print(" processing {}  {} of {}".format(file, idx_cnt, all_num))
                with open(os.path.join(test_file_path, file), 'rb') as f:
                    predict_text = ""
                    # for line in f.readlines():
                    #     line = line.decode('utf-8').strip("\n")
                    #     # line = line.strip("\n")
                    #     predict_text += line
                    predict_text = f.read().decode('utf-8')
                results = model.evaluate_line(sess, input_from_line(predict_text, char_to_id), id_to_tag)
                tag_data = open(os.path.join(test_file_path, os.path.splitext(file)[0] + ".ann"), 'w')
                num = 1
                for ret in results['entities']:
                    content = ret['word']
                    start = int(ret['start'])
                    end = int(ret['end'])
                    # if '顿服' in content:
                    #     print(content)
                    if predict_text[start] != content[0] or predict_text[end-1] != content[-1]:
                        start_tmp =  predict_text[start-5:end+5].find(content)
                        if start_tmp == -1 :
                            continue
                        end = start - 5 + start_tmp + len(content)
                        start = start-5 + start_tmp
                    if content.find('\n') != -1:
                        content = content.replace('\n', ' ')
                    if content[0] == ' ':
                        content = content[1:]
                        start = start + 1
                    prev = ret['type']

                    f_content = content
                    if f_content.find(' ') != -1:
                        num_str = str(start) + ' '
                        f_start = start
                        while f_content.find(' ') != -1:
                            idx = f_content.find(' ')
                            num_str += str(f_start + idx) + ";" + str(f_start + idx + 1) + " "
                            f_start += idx + 1
                            f_content = f_content[idx + 1:]
                        num_str += str(end)
                    else:
                        num_str = str(start) + ' ' + str(end)

                    tag_data.write('T' + str(num) + '\t' + prev + ' ' + num_str + '\t' + content + '\n')
                    num += 1
                tag_data.close()
            # print(result)


def main(_):
    evaluate_line()


if __name__ == "__main__":
    tf.app.run(main)
