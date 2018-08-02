from keras.models import *
from keras.layers import *

def seg_input(input_height, input_width, input_channels):
    """
    Create the input layer to a network
    Args:
        input_height: Height of the input images
        input_width: Width of the input images
        input_channels: Number of channels of the input images (1 for grayscale)
    Return:
        Output of the layer, to be further used in models
    """
    assert(input_height % 32 == 0)
    assert(input_width % 32 == 0)

    img_input = Input(shape=(input_height, input_width, input_channels)) # channel_last
    return img_input

def seg_head(input_img, output, n_classes):
    """
    Create a segmentation head that outputs the finished model.
    Args:
        input_img: The input layer of the model to be created
        output_softmax: The softmaxed output of the last model layer
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    output = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(output)

    # create model
    model = Model(input_img, output)

    o_shape = model.output_shape
    output_height = o_shape[1]
    output_width = o_shape[2]

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
    img_input = seg_input(input_height, input_width, input_channels)    

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    c1 = x
    
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    c2 = x

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)    
    c3 = x

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)    

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate(axis=-1)([c3, x])

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate(axis=-1)([c2, x])

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate(axis=-1)([c1, x])

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    return seg_head(img_input, x, n_classes)

def seg_unet_shallow(input_height, input_width, input_channels, n_classes):
    """
    Segmentation network that is adapted from UNet to have less layers.
    To have a similar receptive field, the kernel size was increased.
    https://arxiv.org/abs/1505.04597

    The network outputs image-like data with the dimensions (None, input_height*input_width, n_classes).

    Args:
        input_height: The height dimension of the input images (divisible by 32)
        input_width: The widht dimension of the input images (divisible by 32)
        input_channels: The number of channels of the input images#
        n_classes: The number of classes of the output images
    Return:
        A tuple containing the model, the height, and the width of the output 
    """

    img_input = seg_input(input_height, input_width, input_channels)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(img_input)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    conv1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    conv2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    
    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([conv2, x])
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([conv1, x])
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    
    o = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)
    
    return seg_head(img_input, o, n_classes)

def seg_unet(input_height, input_width, input_channels, n_classes):
    """
    Segmentation network that is built after UNet.
    https://arxiv.org/abs/1505.04597

    The network outputs image-like data with the dimensions (None, input_height*input_width, n_classes).

    Args:
        input_height: The height dimension of the input images (divisible by 32)
        input_width: The widht dimension of the input images (divisible by 32)
        input_channels: The number of channels of the input images#
        n_classes: The number of classes of the output images
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    img_input = seg_input(input_height, input_width, input_channels)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    conv1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    conv2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    conv3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    conv4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    
    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([conv4, x])
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([conv3, x])
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([conv2, x])
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([conv1, x])
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)
    
    return seg_head(img_input, x, n_classes)

def seg_net_shallow(input_height, input_width, input_channels, n_classes):
    """
    Segmentation network that is built after a shallow version of SegNet
    https://arxiv.org/abs/1511.00561

    The network outputs image-like data with the dimensions (None, input_height*input_width, n_classes).

    Args:
        input_height: The height dimension of the input images (divisible by 32)
        input_width: The widht dimension of the input images (divisible by 32)
        input_channels: The number of channels of the input images#
        n_classes: The number of classes of the output images
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    img_input = seg_input(input_height, input_width, input_channels)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    o = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)
    
    return seg_head(img_input, o, n_classes)

def seg_net(input_height, input_width, input_channels, n_classes):
    """
    Segmentation network that is built after SegNet.
    https://arxiv.org/abs/1511.00561

    The network outputs image-like data with the dimensions (None, input_height*input_width, n_classes).

    Args:
        input_height: The height dimension of the input images (divisible by 32)
        input_width: The widht dimension of the input images (divisible by 32)
        input_channels: The number of channels of the input images#
        n_classes: The number of classes of the output images
    Return:
        A tuple containing the model, the height, and the width of the output 
    """

    img_input = seg_input(input_height, input_width, input_channels)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
       
    # Decoder
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    o = Conv2D(n_classes, (1, 1), activation='relu', padding='same')(x)
    
    return seg_head(img_input, o, n_classes)