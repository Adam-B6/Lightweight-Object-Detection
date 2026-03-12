import tensorflow as tf

input_file = "dataset/canadian_sign_dataset/train/Train13.tfrecord"
output_file = "dataset/canadian_sign_dataset/train/Train_Signs13-tfvision.tfrecord"

raw_dataset = tf.data.TFRecordDataset(input_file)
writer = tf.io.TFRecordWriter(output_file)

for i, raw_record in enumerate(raw_dataset):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    features = example.features.feature

    if "image/source_id" not in features:
        features["image/source_id"].bytes_list.value.append(str(i).encode())

    writer.write(example.SerializeToString())

writer.close()

print("Finished writing:", output_file)