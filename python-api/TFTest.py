import tensorflow as tf

# Print TensorFlow version and available GPUs
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
