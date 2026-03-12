import tensorflow as tf

tfrecord_path = "dataset/canadian_sign_dataset/train/Train_Signs13-tfvision.tfrecord"

feature_description = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),

    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),

    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
}

def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

dataset = tf.data.TFRecordDataset(tfrecord_path)
dataset = dataset.map(parse_example)
print("-" * 50)

for example in dataset.take(3):
    image_bytes = example["image/encoded"]
    image = tf.image.decode_jpeg(image_bytes)

    height = example["image/height"]
    width = example["image/width"]

    labels = tf.sparse.to_dense(example["image/object/class/label"])

    xmin = tf.sparse.to_dense(example["image/object/bbox/xmin"])
    xmax = tf.sparse.to_dense(example["image/object/bbox/xmax"])
    ymin = tf.sparse.to_dense(example["image/object/bbox/ymin"])
    ymax = tf.sparse.to_dense(example["image/object/bbox/ymax"])

    print("Image shape:", image.shape)
    print("Height:", height.numpy(), "Width:", width.numpy())
    print("Number of objects:", labels.shape[0])
    print("Labels:", labels.numpy())
    print("Bounding boxes:", list(zip(xmin.numpy(), ymin.numpy(), xmax.numpy(), ymax.numpy())))
    print("-" * 50)