import tensorflow as tf
print(tf.config.list_physical_devices())
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available:")
    for gpu in gpus:
        print(f"  - {gpu.name}")
else:
    print("No GPU detected. TensorFlow is running on CPU only.")
print("Available XPUs:", tf.config.list_physical_devices('XPU'))
exit(0)