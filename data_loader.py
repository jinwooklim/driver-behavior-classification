import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *


class DataLoader:
    def __init__(self, batch_size=16):
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.BATCH_SIZE = batch_size
        self.NUM_CLASSES = NUM_CLASSES

        self.data_path = '/home/titan/hdd_ext/hdd2/state-farm-distracted-driver-detection'
        self.train_img_path = os.path.join(self.data_path, 'imgs', 'train')
        self.train_label_path = os.path.join(self.data_path, 'driver_imgs_list.csv')

        self.build_train_data()

    def build_train_data(self):
        self.train_dataframe = pd.read_csv(self.train_label_path)
        self.train_dataframe = self.train_dataframe.drop('subject', 1)

        self.train_dataframe, self._val_dataframe = train_test_split(self.train_dataframe, test_size=0.2)
        self.train_array = self.train_dataframe.to_numpy()
        self._val_array = self._val_dataframe.to_numpy()

        def _attach_full_img_path_and_labeling(input_array):
            for i in range(len(input_array)):
                input_array[i, 1] = os.path.join(self.train_img_path, input_array[i, 0], input_array[i, 1])
                input_array[i, 0] = str(input_array[i, 0])[-1]
            return input_array

        self.train_array = _attach_full_img_path_and_labeling(self.train_array)
        self._val_array = _attach_full_img_path_and_labeling(self._val_array)

        def _decode_img(img_path):
            img = tf.io.read_file(img_path)
            # convert the compressed string to a 3D uint8 tensor
            img = tf.image.decode_jpeg(img, channels=3)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tf.image.convert_image_dtype(img, tf.float32)
            # resize the image to the desired size.
            return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

        def _decode_string(str):
            str = tf.strings.to_number(str, out_type=tf.int32)
            return str

        def _parse_function(label_and_img_path):
            img = _decode_img(label_and_img_path[1])
            str = _decode_string(label_and_img_path[0])
            label = tf.one_hot(str, self.NUM_CLASSES)
            return img, label

        self.train_tf_dataset = tf.data.Dataset.from_tensor_slices(self.train_array)
        self.train_tf_dataset = self.train_tf_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.train_tf_dataset = self.train_tf_dataset.shuffle(len(self.train_array)) # 1000
        self.train_tf_dataset = self.train_tf_dataset.repeat()
        self.train_tf_dataset = self.train_tf_dataset.batch(self.BATCH_SIZE)
        self.train_tf_dataset = self.train_tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.train_iter = self.train_tf_dataset.make_one_shot_iterator()

        self._val_tf_dataset = tf.data.Dataset.from_tensor_slices(self._val_array)
        self._val_tf_dataset = self._val_tf_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self._val_tf_dataset = self._val_tf_dataset.batch(self.BATCH_SIZE)
        self._val_tf_dataset = self._val_tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self._val_iter = self._val_tf_dataset.make_one_shot_iterator()

    def get_train_data(self):
        return self.train_iter.get_next()

    def get_val_data(self):
        return self._val_iter.get_next()


if __name__ == "__main__":
    dataloader = DataLoader()
    print(dataloader.get_train_data())
