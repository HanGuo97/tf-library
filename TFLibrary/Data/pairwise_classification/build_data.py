import sys
sys.path.append("/Users/AlexGuo/Downloads/mtl-disparate/")
sys.path.append("/Users/AlexGuo/Downloads/mtl-disparate/preproc/")
from constants import *
from data_reader import task2data_reader
from TFLibrary.Data.pairwise_classification import utils


def build_stance_data():
    # data_train.keys() = ['seq1', 'seq2', 'stance',
    #                      'opinion_towards', 'sentiment', 'labels']
    read_data = task2data_reader(STANCE)
    data_train, data_dev, data_test = read_data(
        datafolder="/Users/AlexGuo/Downloads/mtl-disparate/data/",
        debug=False,
        num_instances=None)

    utils.write_to_file(
        base_file_name="./DATA/TextClassifData/Misc/train",
        sequences_1=data_train["seq1"],
        sequences_2=data_train["seq2"],
        labels=data_train["stance"],
        lower=True, verbose=True)

    utils.write_to_file(
        base_file_name="./DATA/TextClassifData/Misc/val",
        sequences_1=data_dev["seq1"],
        sequences_2=data_dev["seq2"],
        labels=data_dev["stance"],
        lower=True, verbose=True)

    utils.write_to_file(
        base_file_name="./DATA/TextClassifData/Misc/test",
        sequences_1=data_test["seq1"],
        sequences_2=data_test["seq2"],
        labels=data_test["stance"],
        lower=True, verbose=True)
