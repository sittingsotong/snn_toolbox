# -*- coding: utf-8 -*-
"""INI simulator with temporal pattern code.

@author: rbodo
"""

import tensorflow as tf
import numpy as np

from snntoolbox.simulation.target_simulators.\
    INI_temporal_mean_rate_target_sim import SNN as SNN_
from snntoolbox.simulation.utils import get_layer_synaptic_operations, \
    remove_name_counter


class SNN(SNN_):
    """
    The compiled spiking neural network, using layers derived from
    Keras base classes (see
    `snntoolbox.simulation.backends.inisim.temporal_pattern`).

    Aims at simulating the network on a self-implemented Integrate-and-Fire
    simulator using a timestepped approach.

    Attributes
    ----------

    snn: keras.models.Model
        Keras model. This is the output format of the compiled spiking model
        because INI simulator runs networks of layers that are derived from
        Keras layer base classes.
    """

    def __init__(self, config, queue=None):

        SNN_.__init__(self, config, queue)

        self.num_bits = self.config.getint('conversion', 'num_bits')

    def compile(self):

        self.snn = tf.keras.models.Model(
            self._input_images,
            self._spiking_layers[self.parsed_model.layers[-1].name])
        self.snn.compile('sgd', 'categorical_crossentropy', ['accuracy'])

        # Tensorflow 2 lists all variables as weights, including our state
        # variables (membrane potential etc). So a simple
        # snn.set_weights(parsed_model.get_weights()) does not work any more.
        # Need to extract the actual weights here:
        parameter_map = {remove_name_counter(p.name): p for p in
                         self.parsed_model.weights}
        count = 0
        for p in self.snn.weights:
            name = remove_name_counter(p.name)
            if name in parameter_map:
                p.assign(parameter_map[name])
                count += 1
        assert count == len(parameter_map), "Not all weights have been " \
                                            "transferred from ANN to SNN."

        for layer in self.snn.layers:
            if hasattr(layer, 'bias'):
                # Adjust biases to time resolution of simulator.
                layer.bias.assign(layer.bias / self._num_timesteps)

    # @tf.function
    def simulate(self, **kwargs):

        from snntoolbox.utils.utils import echo

        input_b_l = kwargs[str('x_b_l')] * self._dt

        output_b_l_t = np.zeros((self.batch_size, self.num_classes,
                                 self._num_timesteps))

        self._input_spikecount = 0
        self.set_time(self._dt)

        # Main step: Propagate input through network and record output spikes.
        out_spikes = self.snn.predict_on_batch(input_b_l)

        # Broadcast the raw output (softmax) across time axis.
        output_b_l_t[:, :, :] = np.expand_dims(out_spikes, -1)

        # Record neuron variables.
        i = 0
        for layer in self.snn.layers:
            # Excludes Input, Flatten, Concatenate, etc:
            if hasattr(layer, 'spikerates') and layer.spikerates is not None:
                spikerates_b_l = layer.spikerates.numpy()
                spiketrains_b_l_t = to_binary_numpy(spikerates_b_l,
                                                    self.num_bits)
                self.set_spikerates(spikerates_b_l, i)
                self.set_spiketrains(spiketrains_b_l_t, i)
                if self.synaptic_operations_b_t is not None:
                    self.set_synaptic_operations(spiketrains_b_l_t, i)
                if self.neuron_operations_b_t is not None:
                    self.set_neuron_operations(i)
                i += 1
        if 'input_b_l_t' in self._log_keys:
            self.input_b_l_t[Ellipsis, 0] = input_b_l
        if self.neuron_operations_b_t is not None:
            self.neuron_operations_b_t[:, 0] += self.fanin[1] * \
                self.num_neurons[1] * np.ones(self.batch_size) * 2

        print("Current accuracy of batch:")
        if self.config.getint('output', 'verbose') > 0:
            guesses_b = np.argmax(np.sum(output_b_l_t, 2), 1)
            echo('{:.2%}_'.format(np.mean(kwargs[str('truth_b')] ==
                                          guesses_b)))

        return np.cumsum(output_b_l_t, 2)

    def load(self, path, filename):
        SNN_.load(self, path, filename)

    def set_spiketrains(self, spiketrains_b_l_t, i):
        if self.spiketrains_n_b_l_t is not None:
            self.spiketrains_n_b_l_t[i][0][:] = spiketrains_b_l_t

    def set_spikerates(self, spikerates_b_l, i):
        if self.spikerates_n_b_l is not None:
            self.spikerates_n_b_l[i][0][:] = spikerates_b_l

    def set_neuron_operations(self, i):
        self.neuron_operations_b_t += self.num_neurons_with_bias[i + 1]

    def set_synaptic_operations(self, spiketrains_b_l_t, i):
        for t in range(self.synaptic_operations_b_t.shape[-1]):
            ops = get_layer_synaptic_operations(spiketrains_b_l_t[Ellipsis, t],
                                                self.fanout[i + 1])
            self.synaptic_operations_b_t[:, t] += 2 * ops


def to_binary_numpy(x, num_bits):
    """Transform an array of floats into binary representation.

    Parameters
    ----------

    x: ndarray
        Input array containing float values. The first dimension has to be of
        length 1.
    num_bits: int
        The fixed point precision to be used when converting to binary.

    Returns
    -------

    y: ndarray
        Output array with same shape as ``x`` except that an axis is added to
        the last dimension with size ``num_bits``. The binary representation of
        each value in ``x`` is distributed across the last dimension of ``y``.
    """

    print(x)
    n = 2 ** num_bits - 1
    a = np.round(x * n) / n

    y = np.zeros(list(x.shape) + [num_bits])
    for i in range(num_bits):
        p = 2 ** -(i + 1)
        b = np.greater(a, p) * p
        y[Ellipsis, i] = b
        a -= b

    print(y)
    return y
