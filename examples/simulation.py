import os

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

from datetime import datetime


# SNN TOOLBOX CONFIGURATION #®
#############################

# paths
path_wd = '../temp/ann'
model_name = 'mnist_cnn'
parsed_name = 'mnist_cnn_parsed'

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

# brian2
# config['paths'] = {
#     'path_wd': path_wd,             # Path to model.
#     'dataset_path': path_wd,        # Path to dataset.
#     'filename_ann': model_name,      # Name of input model.
#     'filename_parsed_model': parsed_name
# }
#
# config['tools'] = {
#     'evaluate_ann': True,           # Test ANN on dataset before conversion.
#     'normalize': True,              # Normalize weights for full dynamic range.
# }
#
# config['simulation'] = {
#     'simulator': 'brian2',          # Chooses execution backend of SNN toolbox.
#     'duration': 50,                 # Number of time steps to run each sample.
#     'num_to_test':30,               # How many test samples to run.
#     'batch_size': 1,                # Batch size for simulation.
#     'dt': 0.1,                       # Time resolution for ODE solving.
# }
#
# config['input'] = {
#     'poisson_input': False,          # Images are encodes as spike trains.
#     # 'input_rate': 3000
# }

# config['output'] = {
    # 'plot_vars': {                  # Various plots (slows down simulation).
        # 'spiketrains',              # Leave section empty to turn off plots.
        # 'spikerates',
        # 'activations',
        # 'correlation',
        # 'v_mem',
        # 'input_image',
        # 'error_t'}
# }

# INI
config['paths'] = {
    'path_wd': path_wd,             # Path to model.
    'dataset_path': path_wd,        # Path to dataset.
    'filename_ann': model_name      # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    'normalize': True               # Normalize weights for full dynamic range.
}

config['conversion'] = {
    'spike_code': 'temporal_pattern',
    'num_bits': 6
}

config['simulation'] = {
    'simulator': 'INI',             # Chooses execution backend of SNN toolbox.
    'duration': 50,                 # Number of time steps to run each sample.
    'num_to_test': 30,             # How many test samples to run.
    'batch_size': 1,               # Batch size for simulation.
    'keras_backend': 'tensorflow'   # Which keras backend to use.
}

# config['output'] = {
#     'plot_vars': {                  # Various plots (slows down simulation).
#         'spiketrains',              # Leave section empty to turn off plots.
#         'spikerates',
#         'activations',
#         'correlation',
#         'v_mem',
#         'error_t'}
# }

# Store config file.
config_filepath = os.path.join(path_wd, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################

# time
startTime = datetime.now()

main(config_filepath)

print(datetime.now() - startTime)
