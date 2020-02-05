from snntoolbox.bin.utils import update_setup, run_pipeline

filepath = '/mnt/2646BAF446BAC3B9/Data/snn_conversion/realtaste/keras/config'
config = update_setup(filepath)
run_pipeline(config)
