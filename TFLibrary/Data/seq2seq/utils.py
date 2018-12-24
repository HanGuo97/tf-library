from warnings import warn
from collections import Counter
from nltk.tokenize import word_tokenize


def make_vocab(file_name, sequences, vocab_size=None):
    """
    Tokenization, and write to files, format:
    
    vocab.txt:
        emerson
        lake
        palmer
        
    This format can be easily read via:
    `tf.contrib.lookup.index_table_from_file`
    """
    counter = Counter()
    for seq in sequences:
        # tokenization, texts are already preprocessed
        # with nltk such that they are space separable
        tokens = seq.split()
        # remove empty
        tokens = [t for t in tokens if t != ""]
        # build vocab
        counter.update(tokens)

    with open(file_name, "w") as wtr:
        filtered_vocab = [w for w, c in counter.most_common(vocab_size)]
        wtr.write("\n".join(filtered_vocab))
    
    print("Finished writing vocab file to ", file_name)


def write_to_file(base_file_name,
                  source_sequences,
                  target_sequences,
                  vocab_size=None,
                  lower=True,
                  tokenize_fn=None,
                  verbose=True):
    """
    Write Sources and Targets, together with
    corresponding vocab files (capped) to separate files
    """
    if not lower:
        raise NotImplementedError("Only Lower Case is supported")

    if tokenize_fn is None:
        tokenize_fn = word_tokenize

    if not callable(tokenize_fn):
        raise TypeError("`tokenize_fn` should be Callable")

    def _sequence_prepro(seq):
        # lower case
        if not isinstance(seq, (str)):
            warn("`seq` is not string, but %s" % type(seq))
            seq = str(seq)

        preproc_seq = seq.lower()
        # tokenize and join by space
        preproc_seq = " ".join(tokenize_fn(preproc_seq))
        return preproc_seq
        
    
    processed_src = list(map(_sequence_prepro, source_sequences))
    processed_tgt = list(map(_sequence_prepro, target_sequences))
    
    with open(base_file_name + ".source", "w") as f:
        f.write("\n".join(processed_src))
    with open(base_file_name + ".target", "w") as f:
        f.write("\n".join(processed_tgt))
    
    make_vocab(file_name=base_file_name + ".vocab",
               sequences=processed_src + processed_tgt,
               vocab_size=vocab_size)
    
    if verbose:
        print("First 3 Sources:\n\n" + "\n".join(processed_src[:3]), "\n")
        print("First 3 Targets:\n\n" + "\n".join(processed_tgt[:3]), "\n")
