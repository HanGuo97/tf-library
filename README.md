# Tensorflow Libraries

[`Tuner`](TFLibrary/utils/tuner.py) a light helper class performing grid search of hyper-parameters.

[`Module`](TFLibrary/Modules/base.py) a simplified version of [`sonnet.modules`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py)

[`Bandit`](TFLibrary/Bandits/bandits.py) multi-armed bandit

[`Data`](TFLibrary/Data/utils/) utility functions for `tf.data`, inspired from [Google's NMT](https://github.com/tensorflow/nmt)

# Installation
```sh
pip install -e .
pip install -r REQUIREMENTS.txt
```

# Docker
```sh
docker build -t tf-library:`git rev-parse --abbrev-ref HEAD` .
```