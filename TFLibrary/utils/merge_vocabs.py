"""Merging Multiple Vocab Files"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
from copy import deepcopy
from TFLibrary.utils import misc_utils
from TFLibrary.Data.utils import vocab_utils


def merge_vocabs(names,
                 vocab_files,
                 joint_vocab_file,
                 special_tokens=None):
    """Iteratively merge pair of vocabularies"""
    
    # initialize the joint vocab to be an empty list
    # which will be iterally joined
    joint_vocab = []
    vocab_collections = []


    # first join all vocabs
    vocab_sizes = []
    checked_vocab_files = []
    for vocab_file in vocab_files:
        (vocab_size,
         checked_vocab_file) = vocab_utils.check_vocab(
            vocab_file=vocab_file,
            out_dir=os.path.dirname(vocab_file),
            check_special_token=special_tokens is not None)
    
        with open(checked_vocab_file) as f:
            vocab = [d.strip() for d in f.readlines()]
            vocab_collections.append(vocab)
        
        _joint_vocab = deepcopy(joint_vocab)
        # we do this rather than using set()
        # to make sure the order is preserved at every run
        _joint_vocab.extend(deepcopy(vocab))
        _joint_vocab = misc_utils.unique_ordered_list(_joint_vocab)
    

        # Check the existence of Special Tokens:
        #
        # in many downstream applications, for example my
        # own vocab_utils.check_vocab, will check and append
        # certain special_tokens (START, END, UNK etc) at
        # the start of the vocab file. And most of my models
        # rely on this assumption. Since there is no guarantee
        # that during the merging process these special tokens
        # will be placed on the top, it's rather simpler to remove
        # them, and rely on later processing (e.g. vocab_utils.check_vocab)
        # to check or process them.
        if special_tokens is not None:
            # remove the special tokens
            joint_vocab = [v for v in _joint_vocab
                           if v not in special_tokens]
            print("MERGING:\t%d -> %d after removing special_tokens "
                  % (len(_joint_vocab), len(joint_vocab)), special_tokens)
            
            for t in special_tokens:
                assert t not in joint_vocab
        else:
            joint_vocab = _joint_vocab

    # log all the necessary information
    for name, vocab in zip(names, vocab_collections):
        # check the vocab overlaps
        vocab_size = len(vocab)
        joint_vocab_size = len(joint_vocab)
        joint_vocab_overlap = len(set(joint_vocab).intersection(set(vocab)))
        print(misc_utils.bcolors.WARNING +
              "MERGING:\tV-%s = %d, JV = %d, JV^V-%s = %d (%.2f)" %
              (name, vocab_size, joint_vocab_size,
               name, joint_vocab_overlap, vocab_size / joint_vocab_size) +
              misc_utils.bcolors.ENDC)

        # note that the joint vocab is unique
        if special_tokens is not None:
            vocab_indices = [str(joint_vocab.index(v)) for v in vocab
                             if v not in special_tokens]
        else:
            vocab_indices = [str(joint_vocab.index(v)) for v in vocab]
        
        # /path/to/joint_vocab_file.name
        indices_fname = ".".join([joint_vocab_file, name, "indices"])
        with open(indices_fname, "w") as wtr:
            wtr.write("\n".join(vocab_indices))
            print("Wrote Indices File to ", indices_fname)
            
    
    if joint_vocab_file is not None:
        with open(joint_vocab_file, "w") as wtr:
            wtr.write("\n".join(joint_vocab))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", type=str,
        help="comma-separated list of names or tags")
    parser.add_argument("--vocab_files", type=str,
        help="comma-separated list of vocab files to be procecssed")
    parser.add_argument("--joint_vocab_file", default=None,
        help="The directory to save processed vocab_file")
    parser.add_argument("--check_special_token", action="store_true",
        help="Whether to invoke check_special_token")

    FLAGS, unparsed = parser.parse_known_args()

    names = FLAGS.names.split(",")
    vocab_files = FLAGS.vocab_files.split(",")
    special_tokens = (
        [vocab_utils.EOS, vocab_utils.SOS, vocab_utils.UNK]
        if FLAGS.check_special_token else None)

    if len(names) != len(vocab_files):
        raise ValueError("len(names) != len(vocab_files)")

    merge_vocabs(names=names,
                 vocab_files=vocab_files,
                 joint_vocab_file=FLAGS.joint_vocab_file,
                 special_tokens=special_tokens)
