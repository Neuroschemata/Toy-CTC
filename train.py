"""
The following code is set up to demonstrate a
a working CTC implementation.
(1) Since the demo is quite small, we don't need
the full machinery of a bi-directional RNN with
LSTM units. We can make do with a single layer
RNN w/ either sigmoid or tanh units.
(2) There are two data sets included with the demo:
    (a) data.pkl uses 4-dim input vectors and can be
        trained relatively quickly (in under 100 epochs)
        using an RNN with a dozen or so units.
    (b) dataLonger.pkl uses 8-dim input vectors, and is
        best trained using an RNN with 50 units, running
        for about 500 to 1000 epochs.
As of this release, the number of epochs and the RNN
parameters are hard-coded in <<configs>> and <<num_epochs>>.
"""
import sys
import pickle

import numpy as np
import theano

from neuralnet import NeuralNet
from typeset import terminal_print, print_CTC_decoding

from lstm import LSTM, BDLSTM
from recurrent import RecurrentLayer
configs = ((RecurrentLayer, {"nunits": 12}),)

def diagnostix(inputLabels, currentImage,
             softmaxActivs=None,
             aux_img=None, aux_name=None):
    """
    Monitoring function to show inputs/outputs
    :@param inputLabels: what is says on the tin
    :@param currentImage: the input image
    :@param softmaxActivs: softmax activations
    :@param aux_img: auxiliary image/matrix for debugging
    :@param aux_name: name given to aux_img
    :@return:
    """
    print('Sequence presented at test time:     ', end='')
    labels_print(inputLabels)

    if softmaxActivs is not None:
        print('Decoding (largest softmax response): ', end='')
        winners = np.argmax(softmaxActivs, 0)
        labels_print(winners)

    print('Image sequence on display:')
    terminal_print(currentImage)

    if softmaxActivs is not None:
        print('SoftMax Activations:')
        terminal_print(softmaxActivs)


# diagnostics complete
# begin processing

config_num = 0
in_log_scale = True

if len(sys.argv) < 2:
    print('Usage\n{} <data_file.pkl> [configuration#={}] [which_scale={}]'
          ''.format(sys.argv[0], config_num, in_log_scale))
    sys.exit(1)

with open(sys.argv[1], "rb") as pkl_file:
    data = pickle.load(pkl_file)

if len(sys.argv) > 2:
    config_num = int(sys.argv[2])

if len(sys.argv) > 3:
    in_which_scale_scale = sys.argv[3][0] in "TtYy1"


# training details, network parameters, etc.
hiddenLyr, hiddenLyrArgs = configs[config_num]
phones = data['phones']
num_classes = len(phones)
num_dims = len(data['x'][0])
num_examples = len(data['x'])
num_training_examples = 0.8*num_examples
num_epochs = 100
blankIndex = num_classes
labels_print, labels_len = print_CTC_decoding(phones)

print("\nInput Vector Dimensions: {}"
      "\nNo. of Classes: {}"
      "\nHidden Layer: {} {}"
      "\nNo. of Samples: {}"
      "\nFloatX: {}"
      "\nCTC working in log scale: {}"
      "\n".format(num_dims, num_classes, hiddenLyr, hiddenLyrArgs,
                  num_examples, theano.config.floatX, in_log_scale))

################################
print("Loading Data ▪▪▪")
try:
    conv_sz = hiddenLyrArgs["conv_sz"]
except KeyError:
    conv_sz = 1

data_x, data_y = [], []
bad_data = False

for x, y in zip(data['x'], data['y']):
    # Insert blanks b/w every pair of labels & at the beginning & end of sequence
    y1 = [blankIndex]
    for phonemeIndex in y:
        y1 += [phonemeIndex, blankIndex]

    data_y.append(np.asarray(y1, dtype=np.int32))
    data_x.append(np.asarray(x, dtype=theano.config.floatX))

    if labels_len(y1) > (1 + len(x[0])) // conv_sz:
        bad_data = True
        diagnostix(y1, x, None, x[:, ::conv_sz], "auxiliary name")


network = NeuralNet(num_dims, num_classes, hiddenLyr, hiddenLyrArgs, in_log_scale)

print("Training ▪▪▪")
for epoch in range(num_epochs):
    print('Epoch : ', epoch)
    for example in range(num_examples):
        x = data_x[example]
        y = data_y[example]

        if example < num_training_examples:
            if in_log_scale and len(y) < 2:
                continue

            cst, pred, aux = network.train(x, y)
            if (epoch % 12 == 0 and example < 3) or np.isinf(cst):
               print('\n▪▪▪▪▪▪▪▪▪▪▪▪▪▪ COST = {}  ▪▪▪▪▪▪▪▪▪▪▪▪▪▪ '.format(np.round(cst, 3)))
               diagnostix(y, x, pred, aux > 1e-20, 'Forward probabilities:')
            if np.isinf(cst):
                print('Cost Blew Up! Exiting ...')
                sys.exit()

        elif ((epoch >1 and epoch % 12 == 0) and example - num_training_examples < 3) \
                or epoch == num_epochs - 1:
            # Sample some images for testing
            pred, aux = network.test(x)
            aux = (aux + 1) / 2.0
            print('\n▪▪▪▪▪▪▪▪▪▪▪▪▪▪ TESTING ▪▪▪▪▪▪▪▪▪▪▪▪▪▪')
            diagnostix(y, x, pred, aux)
