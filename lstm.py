"""
Acknowledgements: LSTM implementations by Shawn Tan,
Kyle Kastner, Thomas Breuel, Volkmar Frinken, Marcus Liwicki, et al.
"""
import theano.tensor as TT
import theano
from theano.tensor.nnet import sigmoid
import numpy as np

from activations import activation_by_name
from weights import stacked_ortho_wts, share


class LSTM():

    def __init__(self, inpt,
                 nin, nunits,
                 forget=False,
                 pre_activation='tanh',
                 post_activation='linear',
                 learn_init_states=True):
        """
        Init
        :@param inpt: activations from incoming layer.
        :@param nin: dimensions of incoming layer.
        :@param nunits: number of units.
        :@param forget: use forget gate
        :@param pre_activation: activation pre-synaptic to central cell.
        :@param post_activation: activation applied to central cell b4 output.
        :@param learn_init_states: learn the initial states
        :@return: Output
        """

        num_activations = 3 + forget
        w = stacked_ortho_wts(nin, nunits, num_activations)
        u = stacked_ortho_wts(nunits, nunits, num_activations)
        b = share(np.zeros(num_activations * nunits))
        out0 = share(np.zeros(nunits))
        cell0 = share(np.zeros(nunits))

        pre_activation = activation_by_name(pre_activation)
        post_activation = activation_by_name(post_activation)

        def step(in_t, out_tm1, cell_tm1):
            """
            Scan function.
            :@param in_t: current input from incoming layer
            :@param out_tm1: prev output of LSTM layer
            :@param cell_tm1: prev central cell value
            :@return: current output and central cell value
            """
            tmp = TT.dot(out_tm1, u) + in_t

            inn_gate = sigmoid(tmp[:nunits])
            out_gate = sigmoid(tmp[nunits:2 * nunits])
            fgt_gate = sigmoid(
                tmp[2 * nunits:3 * nunits]) if forget else 1 - inn_gate

            cell_val = pre_activation(tmp[-nunits:])
            cell_val = fgt_gate * cell_tm1 + inn_gate * cell_val
            out = out_gate * post_activation(cell_val)

            return out, cell_val

        inpt = TT.dot(inpt, w) + b
        # seqlen x nin * nin x 3*nout + 3 * nout  = seqlen x 3*nout

        rval, updates = theano.scan(step,
                                sequences=[inpt],
                                outputs_info=[out0, cell0], )

        self.output = rval[0]
        self.params = [w, u, b]
        if learn_init_states:
            self.params += [out0, cell0]
        self.nout = nunits


class BDLSTM():
    """
    Bidirectional LSTM Layer.
    """
    def __init__(self, inpt,
                 nin, nunits,
                 forget=False,
                 pre_activation='tanh',
                 post_activation='linear',
                 learn_init_states=True):

        fwd = LSTM(inpt, nin, nunits, forget, pre_activation, post_activation,
                   learn_init_states)
        bwd = LSTM(inpt[::-1], nin, nunits, forget, pre_activation, post_activation,
                   learn_init_states)

        self.params = fwd.params + bwd.params
        self.nout = fwd.nout + bwd.nout
        self.output = TT.concatenate([fwd.output,
                                      bwd.output[::-1]],
                                     axis=1)
