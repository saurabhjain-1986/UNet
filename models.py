import tensorflow as tf
from keras import layers
from loss import combine_dice_focal_loss, dice_coefficient


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def double_conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding="same",
                      activation="relu",
                      kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same",
                      activation="relu",
                      kernel_initializer="he_normal")(x)
    return x


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x


def build_unet_model(input_shape, num_classes):

    num_filters = 64
    # inputs
    inputs = layers.Input(shape=input_shape)

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, num_filters)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 2 * num_filters)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 4 * num_filters)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 8 * num_filters)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 16 * num_filters)
    bottleneck = layers.Dropout(0.2)(bottleneck)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 8 * num_filters)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 4 * num_filters)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 2 * num_filters)
    # 9 - upsample
    u9 = upsample_block(u8, f1, num_filters)

    # outputs
    outputs = layers.Conv2D(num_classes, 1, padding="same",
                            activation="softmax")(u9)

    # unet model with Keras Functional API
    return tf.keras.Model(inputs, outputs, name="U-Net")


def build_compile_model(input_shape=(128, 128, 3), num_classes=3):
    unet_model = build_unet_model(input_shape=input_shape, num_classes=num_classes)
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss=combine_dice_focal_loss,
                       metrics=dice_coefficient)
    return unet_model


if __name__ == '__main__':
    model = build_compile_model()
    tf.keras.utils.plot_model(model, show_shapes=True)
