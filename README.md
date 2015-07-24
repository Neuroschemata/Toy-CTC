######For details on the setup used for the demo, refer to "synopsis.pdf".
--------------------------------------------------------------------------
A barebones demo to illustrate a working implementation of Graves' Connectionist Temporal Classification

1. Since the demo is quite small, we don't need the full machinery of a bi-directional RNN with LSTM units. We can make do with a single layer RNN w/ either sigmoid or tanh units.

2. There are two data sets included with the demo:

+ data.pkl uses 4-dim input vectors and can be trained relatively quickly (in under 100 epochs) using an RNN with a dozen or so units.

+ dataLonger.pkl uses 8-dim input vectors, and is best trained using an RNN with about 50 units, running for about 500 to 1000 epochs.

As of this release, the number of epochs and the RNN parameters are hard-coded in configs and num_epochs

**NB:** The demo assumes a python 3 installation, as well as Theano and numpy ...

> To execute the simplest demo, run `python train.py data.pkl`

> To execute the larger model, edit **train.py** by setting

  > > `configs = ((RecurrentLayer, {"nunits": 50}),)` and `num_epochs = 500`, and then run

      `python train.py dataLonger.pkl`

######In the works ...
----------------------
+ Optimize for GPUs to facilitate working with real speech data.
+ Implement *Prefix Search Decoding* and the *Token-Passing Algorithm*.
