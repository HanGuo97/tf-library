import os
import pickle


class Vocabulary(object):
    """Vocabulary class for an image-to-text model."""

    def __init__(self,
                 reverse_vocab,
                 unk_word="UNK"):
        """Initializes the vocabulary"""
        if unk_word not in reverse_vocab:
            reverse_vocab.append(unk_word)
        vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

        print("Created vocabulary with %d words" % len(vocab))

        self.vocab = vocab  # vocab[word] = id
        self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word
        self.unk_id = vocab[unk_word]

    def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        else:
            return self.reverse_vocab[word_id]

    def save(self, file_dir):
        with open(file_dir, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        print("INFO: Successfully Saved Vocabulary to ", file_dir)

    def load(self, file_dir):
        if not os.path.exists(file_dir):
            raise ValueError("File not exist ", file_dir)

        with open(file_dir, "rb") as f:
            dump = pickle.load(f)

        self.vocab = dump.vocab
        self.reverse_vocab = dump.reverse_vocab
        self.unk_id = dump.unk_id

        print("INFO: Successfully Loaded Vocabulary from ", file_dir)
