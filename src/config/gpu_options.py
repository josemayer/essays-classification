import tensorflow as tf

def gpu_config():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    return config
