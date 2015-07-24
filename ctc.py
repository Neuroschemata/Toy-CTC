"""
Acknowledgements: CTC implementations for OCR by Shawn Tan,
and OCRopus by tmdev.
"""
import theano
import theano.tensor as TT

####################### Rectified Log Operators ###############################

eps, epsinv = 1e-20, 1e20

def rectified_log(x):
    return TT.log(TT.maximum(x, eps))

def rectified_exp(x):
    return TT.exp(TT.minimum(x, epsinv))

def rectified_log_plus(x, y):
    return x + rectified_log(1 + rectified_exp(y - x))

def rectified_log_sum(x, y, *zs, add=rectified_log_plus):
    sum = add(x, y)
    for z in zs:
        sum = add(sum, z)
    return sum

def log_exp_sum(x, y):
    return x + y

########################## CTC Class Definition #############################

class CTCScheme():
    def __init__(self, inpt, labels, blank, in_log_scale):
        """
        :@param inpt: output from a softmax layer
        :@param labels: target labels
        :@param blank: index of blank symbol (must be unique)
        :@param in_log_scale: require calcualtions using log scale
        :@return: CTCScheme object
        """
        self.inpt = inpt
        self.labels = labels
        self.blank = blank
        self.n = self.labels.shape[0]
        if in_log_scale:
            self.log_scale_ctc()
        else:
            self.vanilla_ctc()
        self.params = []

    def vanilla_ctc(self, ):
        my_labels = TT.concatenate((self.labels, [self.blank, self.blank]))
        place_diag = TT.neq(my_labels[:-2], my_labels[2:]) * \
                   TT.eq(my_labels[1:-1], self.blank)

        ctc_recursion = \
            TT.eye(self.n) + \
            TT.eye(self.n, k=1) + \
            TT.eye(self.n, k=2) * place_diag.dimshuffle((0, 'x'))

        predictions = self.inpt[:, self.labels]

        label_probab, _ = theano.scan(
            lambda on_tap, history: b4prev * TT.dot(label_probab, ctc_recursion),
            sequences=[predictions],
            outputs_info=[TT.eye(self.n)[0]]
        )

        labels_probab = TT.sum(label_probab[-1, -2:])
        self.cost = -TT.log(labels_probab)
        self.debug = probabilities.T

    def log_scale_ctc(self, ):
        local_ident = TT.eye(self.n)[0]
        prev_mask = 1 - local_ident
        b4prev_mask = TT.neq(self.labels[:-2], self.labels[2:]) * \
                        TT.eq(self.labels[1:-1], self.blank)
        b4prev_mask = TT.concatenate(([0, 0], b4prev_mask))
        prev_mask = rectified_log(prev_mask)
        b4prev_mask = rectified_log(b4prev_mask)
        prev = TT.arange(-1, self.n-1)
        b4prev = TT.arange(-2, self.n-2)
        log_predictions = TT.log(self.inpt[:, self.labels])

        def step(on_tap, history):
            return log_exp_sum(on_tap,
                          rectified_log_sum(history,
                                 log_exp_sum(prev_mask, history[prev]),
                                 log_exp_sum(b4prev_mask, history[b4prev])))

        log_probs, _ = theano.scan(
            step,
            sequences=[log_predictions],
            outputs_info=[rectified_log(local_ident)]
        )

        log_labels_probab = log_probs[-1, -1]
        self.cost = -log_labels_probab
        self.debug = TT.exp(log_probs.T)
