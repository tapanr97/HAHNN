from keras import initializers
from keras.layers import *

from dataLoadUtilities import *


class Attention(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.att_size = 38
        super(Attention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def build(self, input_shape):
        self.weight = backend.variable(self.init((input_shape[-1], 1)))
        self.v = backend.variable(self.init((self.att_size, 1)))
        self.bias = backend.variable(self.init((self.att_size,)))
        self._trainable_weights = [self.weight, self.bias, self.v]
        super(Attention, self).build(input_shape)

    def call(self, _input, mask=None, **kwargs):
        context_vector = backend.tanh(backend.dot(_input, self.weight) + self.bias)
        alpha = backend.dot(context_vector, self.v)
        alpha = backend.squeeze(alpha, -1)
        alpha = backend.exp(alpha)
        alpha /= backend.cast(backend.sum(alpha, axis=1, keepdims=True) + backend.epsilon(), backend.floatx())
        alpha = backend.expand_dims(alpha)
        word_annotations = _input * alpha
        _sum = backend.sum(word_annotations, axis=1)
        return _sum


