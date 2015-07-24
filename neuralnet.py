"""
Acknowledgements: Chris Olah's neural net implementations,
and various implementations by Theano's developers.
"""
import theano
import theano.tensor as TT
from ctc import CTCScheme
from outputLayers import SoftmaxLayer


class NeuralNet():
    def __init__(self, n_dims, n_classes,
                 midlayer, midlayer_args,
                 in_log_scale=True):
        image = TT.matrix('image')
        labels = TT.ivector('labels')

        layer1 = midlayer(image.T, n_dims, **midlayer_args)
        layer2 = SoftmaxLayer(layer1.output, layer1.nout, n_classes + 1)
        layer3 = CTCScheme(layer2.output, labels, n_classes, in_log_scale)

        updates = []
        for lyr in (layer1, layer2, layer3):
            for p in lyr.params:
                grad = TT.grad(layer3.cost, p)
                updates.append((p, p - .001 * grad))

        self.train = theano.function(
            inputs=[image, labels],
            outputs=[layer3.cost, layer2.output.T, layer3.debug],
            updates=updates, )

        self.test = theano.function(
            inputs=[image],
            outputs=[layer2.output.T, layer1.output.T], )
