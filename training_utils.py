import os
import numpy as np
from PIL import Image
from scipy import ndimage

import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io


# Generic utils
def resample_image(image_data=None, shape=None, interpolation="CUBIC"):
    if interpolation == 'CUBIC':
        return np.array(Image.fromarray(image_data).resize(shape, Image.BICUBIC, reducing_gap=3))
    elif interpolation == 'NEAREST':
        return np.array(Image.fromarray(image_data).resize(shape, Image.NEAREST))


def largest_connected_component_binary_image(binary_image_data):
    label_im, nb_labels = ndimage.label(binary_image_data)
    sizes = ndimage.sum(binary_image_data, label_im, range(1, nb_labels + 1))
    return label_im == np.argmax(sizes) + 1


# Pre-processing of data
def load_img_masks(input_img_path, target_img_path, img_size=(128, 128)):
    input_img = tf_io.read_file(input_img_path)
    input_img = tf_io.decode_png(input_img, channels=3)
    target_img = tf_io.read_file(target_img_path)
    target_img = tf_io.decode_png(target_img, channels=1)

    # Resize and type casting
    input_img = tf_image.resize(input_img, img_size)
    input_img = tf_image.convert_image_dtype(input_img, "float32")
    target_img = tf_image.resize(target_img, img_size, method="nearest")
    target_img = tf_image.convert_image_dtype(target_img, "uint8")

    # Normalize
    input_img = tf.cast(input_img, tf.float32) / 255.0
    # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2.
    target_img -= 1
    target_img = tf.one_hot(tf.squeeze(target_img), depth=3)

    return input_img, target_img


# Prepare training data
def get_dataset_training(batch_size, input_img_paths, target_img_paths, max_dataset_len=None):
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)


def get_dataset_validation(batch_size, input_img_paths, target_img_paths, max_dataset_len=None):
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)


def my_callbacks(path):
    monitor_loss = 'val_loss'
    model_file_path = os.path.join(path, 'models', "{epoch:02d}-{val_loss:.3f}.hdf5")
    tensorboard_log_dir = os.path.join(path, 'models', "log")
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_file_path, monitor='{}'.format(monitor_loss), save_best_only=True,
                                           save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='{}'.format(monitor_loss), factor=0.25, patience=5, verbose=1,
                                             mode='auto', min_lr=1e-8),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, write_images=True),
        tf.keras.callbacks.EarlyStopping(monitor='{}'.format(monitor_loss), patience=20, restore_best_weights=True)
        ]

    return callbacks
