import tensorflow as tf


def focal_loss(targets, prediction, alpha=0.999, gamma=3):
    prediction = tf.keras.backend.flatten(prediction)
    targets = tf.keras.backend.flatten(targets)

    bce = tf.keras.backend.binary_crossentropy(targets, prediction)
    bce_exp = tf.keras.backend.exp(-bce)

    return tf.keras.backend.mean(alpha * tf.keras.backend.pow((1 - bce_exp), gamma) * bce)


def dice_coefficient(y_true, y_prediction):
    smooth = 1
    y_true = tf.keras.layers.Flatten()(y_true)
    y_prediction = tf.keras.layers.Flatten()(y_prediction)
    intersection = tf.reduce_sum(y_true * y_prediction)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_prediction) + smooth)


def dice_loss(y_true, y_prediction):
    return 1.0 - dice_coefficient(y_true, y_prediction)


def combine_dice_focal_loss(y_true, y_prediction):

    return dice_loss(y_true, y_prediction) + focal_loss(y_true, y_prediction)
