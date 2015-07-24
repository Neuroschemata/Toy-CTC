import theano.tensor as TT
from weights import init_wts, share


class LinearLayer():
    def __init__(self, inpt, in_sz, n_classes, tied=False):
        if tied:
            b = share(init_wts(n_classes-1))
            w = share(init_wts(in_sz, n_classes-1))
            w1 = TT.horizontal_stack(w, TT.zeros((in_sz, 1)))
            b1 = TT.concatenate((b, TT.zeros(1)))
            self.output = TT.dot(inpt, w1) + b1
        else:
            b = share(init_wts(n_classes))
            w = share(init_wts(in_sz, n_classes))
            self.output = TT.dot(inpt, w) + b
        self.params = [w, b]


class SoftmaxLayer(LinearLayer):
    def __init__(self, inpt, in_sz, n_classes, tied=False):
        super().__init__(inpt, in_sz, n_classes, tied)
        self.output = TT.nnet.softmax(self.output)
