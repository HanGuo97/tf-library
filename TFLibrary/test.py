from TFLibrary.Bandits import bandits_test
from TFLibrary.DDPG import ddpg_test
from TFLibrary.SPG import pg_decoder_test


if __name__ == "__main__":
    bandits_test.test()
    ddpg_test.test()
    pg_decoder_test.test()
