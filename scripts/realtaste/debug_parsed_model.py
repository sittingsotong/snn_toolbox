import os
import keras
import numpy as np
import matplotlib.pyplot as plt

from snntoolbox.bin.utils import update_setup
from snntoolbox.conversion.utils import get_activations_batch, \
    get_activations_layer
from snntoolbox.datasets.utils import get_dataset

path = '/mnt/2646BAF446BAC3B9/Data/snn_conversion/realtaste/keras'

input_model = keras.models.load_model(os.path.join(path, 'model.h5'))
parsed_model = keras.models.load_model(os.path.join(path, 'model_parsed.h5'))

config = update_setup(os.path.join(path, 'config'))
normset, testset = get_dataset(config)
x = testset['x_test']

activations_parsed = get_activations_batch(parsed_model, x)
layer_names = []
for i in range(len(activations_parsed)):
    name = activations_parsed[i][1]
    print(name)
    layer = input_model.get_layer(name)
    a_input = get_activations_layer(input_model.input, layer.output, x).ravel()
    a_parsed = activations_parsed[i][0].ravel()
    print(np.array_equal(a_input, a_parsed))
    print(np.corrcoef(a_input, a_parsed)[0, 1])
    print('')
