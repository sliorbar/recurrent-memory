import lasagne
import theano


class LayerNormalization(object):

    def __init__(self, num_units,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 b=lasagne.init.Constant(0.),
                 g=lasagne.init.Constant(1.),
                 eps=1e-5):
        self.num_units = num_units
        self.b = theano.shared(b.sample(num_units), name='layer_norm.b')
        self.g = theano.shared(g.sample(num_units), name='layer_norm.g')
        self.eps = eps
        self.nonlinearity = nonlinearity

    def normalizing_nonlinearity(self, x):
        mean = x.mean(-1, keepdims=True)
        sigma = theano.tensor.sqrt(x.var(-1, keepdims=True) + self.eps)
        b = self.b.reshape((1,) * (x.ndim - 1) + (-1,))
        g = self.g.reshape((1,) * (x.ndim - 1) + (-1,))
        return self.nonlinearity(g * (x - mean) / sigma + b)

    def register_to(self, layer):
        layer.add_param(self.b, (self.num_units,))
        layer.add_param(self.g, (self.num_units,))


class RecurrentNormalizingLayer(lasagne.layers.RecurrentLayer):

    def __init__(self, incoming, num_units,
                 W_hid_to_hid=lasagne.init.Uniform(1e-4),
                 b=lasagne.init.Uniform(0.1),
                 g=lasagne.init.Constant(1.),
                 hid_init=lasagne.init.Uniform(0.1),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 eps=0.05,
                 **kwargs):
        self.layer_normalization = LayerNormalization(
            num_units, nonlinearity=nonlinearity, b=b, g=g, eps=eps)
        super(RecurrentNormalizingLayer, self).__init__(
            incoming, num_units,
            W_hid_to_hid=W_hid_to_hid,
            b=None,
            hid_init=hid_init,
            nonlinearity=self.layer_normalization.normalizing_nonlinearity,
            **kwargs)
        self.layer_normalization.register_to(self.hidden_to_hidden)