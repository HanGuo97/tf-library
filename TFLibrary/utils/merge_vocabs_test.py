import os
from TFLibrary.utils import misc_utils

VOCAB_FILES = ["../Data/test_data/merge_vocabs/label_vocab_A.txt",
               "../Data/test_data/merge_vocabs/label_vocab_B.txt",
               "../Data/test_data/merge_vocabs/label_vocab_C.txt",
               "../Data/test_data/merge_vocabs/label_vocab_D.txt"]
JOINT_VOCAB_FILE = "../Data/test_data/merge_vocabs/joint_vocab"
INDICES_FILES = [".".join([JOINT_VOCAB_FILE, name, "indices"])
                 for name in ["A", "B", "C", "D"]]

bash_script = """
python merge_vocabs.py \\
    --names "A,B,C,D" \\
    --vocab_files "%s" \\
    --joint_vocab_file "%s"
""" % (",".join(VOCAB_FILES), JOINT_VOCAB_FILE)



def test(remove_after_run=False):
    misc_utils.run_command(bash_script)
    for fname in [JOINT_VOCAB_FILE] + INDICES_FILES:
        indices = misc_utils.read_text_file(fname)
        indices_expected = misc_utils.read_text_file(fname + ".expected")
        if indices != indices_expected:
            raise ValueError("Error")
        print(indices, indices_expected, indices == indices_expected)
    
        if remove_after_run:
            os.remove(fname)


if __name__ == "__main__":
    test(True)
