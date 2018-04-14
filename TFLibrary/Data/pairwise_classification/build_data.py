"""https://github.com/coastalcph/mtl-disparate.git"""
import os
import sys
sys.path.append("/Users/AlexGuo/Downloads/mtl-disparate/")
sys.path.append("/Users/AlexGuo/Downloads/mtl-disparate/preproc/")
from constants import *
from data_reader import task2data_reader
from TFLibrary.Data.pairwise_classification import utils

DataFolder = "/Users/AlexGuo/Downloads/mtl-disparate/data/"


def build_data(datafolder=DataFolder,
               base_folder_name="./DATA/TextClassifData/"):
    """
    Read, Process, and Save data into tf.data readable formats
    Given a base_folder_name, we will save multiple files, including
    mutiple input sequences, targets, vocabulary files. Structure:
        
        base_folder_name/
            Task_1/
                train.*
                val.*
                test.*
            Task_2/
                ...
            Task_3/
                ...
            ...

    Args:
        datafolder: folder to all data, see Ruder's code for documentation
        base_folder_name: folder to save all data
    """

    # data_train.keys() = [
    #   "seq1", "seq2", "stance",
    #   "opinion_towards", "sentiment", "labels"]
    # where seq1, seq2 are input sequences, stance is the target
    # everything else seems irrelevant for this case
    for task in [STANCE, FNC, NLI, TOPIC,
                 TOPIC_5WAY, LAPTOP, RESTAURANT, TARGET]:
        
        # sorry for hard-coding this
        # original names are like: semeval2016-task6-stance
        # and we process them to be Semeval2016Task6Stance
        task_name = "".join([t.upper() for t in task.split("-")])
        # /base/Semeval2016Task6Stance
        task_folder_name = os.path.join([base_folder_name, task_name])
        # /base/Semeval2016Task6Stance/train
        _get_base_file_name = lambda m: os.path.join([task_folder_name, m])

        read_data = task2data_reader(STANCE)
        data_train, data_dev, data_test = read_data(
            datafolder=DataFolder,
            debug=False,
            num_instances=None)

        utils.write_to_file(
            base_file_name=_get_base_file_name("train"),
            sequences_1=data_train["seq1"],
            sequences_2=data_train["seq2"],
            labels=data_train["stance"],
            lower=True, verbose=True)

        utils.write_to_file(
            base_file_name=_get_base_file_name("val"),
            sequences_1=data_dev["seq1"],
            sequences_2=data_dev["seq2"],
            labels=data_dev["stance"],
            lower=True, verbose=True)

        utils.write_to_file(
            base_file_name=_get_base_file_name("test"),
            sequences_1=data_test["seq1"],
            sequences_2=data_test["seq2"],
            labels=data_test["stance"],
            lower=True, verbose=True)
