from keras.models import *
from keras.layers import *

def seg_head(input_img, output_softmax):
    """
    Create a segmentation head that outputs the finished model.
    Args:
        input_img: The input layer of the model to be created
        output_softmax: The softmaxed output of the last model layer
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    # create model
    model = Model(input_img, output_softmax)

    # calculate output shape
    o_shape = model.output_shape
    output_height = o_shape[2]
    output_width = o_shape[3]

    # set model attributes
    model.outputWidth = output_width
    model.outputHeight = output_height

    return model, output_height, output_width


def seg_test_net(input_height, input_width, input_channels, n_classes):
    """
    Create a simple image segmentation model based on UNet
    Args:
        input_height: The height dimension of the input images (divisible by 32)
        input_width: The widht dimension of the input images (divisible by 32)
        input_channels: The number of channels of the input images#
        n_classes: The number of classes of the output images
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    assert(input_height % 32 == 0)
    assert(input_width % 32 == 0)

    img_input = Input(shape=(input_channels, input_height, input_width)) # channel_first

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first')(x)
    c1 = x

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first')(x)
    x = BatchNormalization(name='block1_batchnorm')(x)
    x = Dropout(0.5, name='block1_dropout')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conn_conv1', data_format='channels_first')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conn_conv2', data_format='channels_first')(x)
    x = Dropout(0.5, name='conn_dropout')(x)

    x = UpSampling2D((2, 2), name='up1_upsample', data_format='channels_first')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='up1_conv1', data_format='channels_first')(x)
    x = Concatenate(axis=1, name='up1_concatenate')([c1, x])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='up1_conv2', data_format='channels_first')(x)
    o = Conv2D(n_classes, (3, 3), activation='softmax', padding='same', name='up1_conv3', data_format='channels_first')(x)

    return seg_head(img_input, o)