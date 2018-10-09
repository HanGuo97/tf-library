# Tensorflow Libraries
TFLibrary is an open-source toolkit based on Tensorflow, with the design goals of modularity, flexibility and extensibility. It is in active development and used extensively in my own researches.


* [`Data`](TFLibrary/Data/utils/) utility functions for `tf.data`, inspired from [Google's NMT](https://github.com/tensorflow/nmt)

* [`Models`](TFLibrary/Models/) simple wrapper class around models.

* [`Modules`](TFLibrary/Modules/) a simplified version of [`sonnet.modules`](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/base.py) for modular model design.
    - [`Modules.attention`](TFLibrary/Modules/attentions.py) attention modules takes in as input multiple inputs and compute attended features, currently includes matrix cross attention and BiDAF-style attention.
    - [`Modules.embedding`](TFLibrary/Modules/embedding.py) embedding modules takes texts or text-ids and produce dense representations, currently includes matrix embedding, ELMO embedding.
    - [`Modules.encoders`](TFLibrary/Modules/encoders.py) encoder modules encodes dense representations of input texts, currently includes LSTM encoder and BiDAF-style encoder.
    - [`Modules.transformer`](TFLibrary/Modules/transformer.py) transformer modules are simplified / decomposed version of Transformer in [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) library, currently include Transformer encoder.

* [`utils`](TFLibrary/utils/) a collection of various utility functions.
    - [`utils.TrainingManager`](TFLibrary/utils/training_manager.py) a light helper class for monitoring training progress, and providing signals for early stopping.
    - [`utils.Tuner`](TFLibrary/utils/tuner.py) a light helper class performing grid search of hyper-parameters in a multi-GPU settings.

* [`Bandit`](TFLibrary/Bandits/bandits.py) multi-armed bandit.

* __`misc`__ there are also a few other models, including [`DDPG`](TFLibrary/DDPG), [`pointer network`](TFLibrary/SPG/pg_decoder.py) and its [`rnn_cell_impl.RNNCell wrapper`](TFLibrary/Seq2Seq/pointer_cell.py) version.


# Installation
```sh
pip install -e .
pip install -r REQUIREMENTS.txt
```

# Docker
```sh
# using `--no-cache` to avoid accidentally missing changes in parent tensorflow image
docker build --no-cache -t tf-library:`git rev-parse --abbrev-ref HEAD` .
```