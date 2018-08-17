from keras.models import *
from keras.layers import *

def timebased_input(timesteps, input_height, input_width, input_channels):
    """
    Create the input layer to a network that is working on time based data
    Args:
        timesteps: The number of time steps in data
        input_height: Height of the input images
        input_width: Width of the input images
        input_channels: Number of channels of the input images (1 for grayscale)
    Return:
        Output of the layer, to be further used in models
    """
    assert(input_height % 32 == 0)
    assert(input_width % 32 == 0)

    img_input = Input(shape=(timesteps, input_height, input_width, input_channels)) # channel_last
    return img_input

def collapsing_head(input_img, output, timesteps, n_classes):
    """
    Create a head that collapses the timebased data into a single image
     Args:
        input_img: The input layer of the model to be created
        output: The output of the last layer (without any special activation function)
        timesteps: The number of timesteps the data has
        n_classes: The number of classes of the target data
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    output = TimeDistributed(Conv2D(n_classes, (1, 1), activation='relu', padding='same'))(output)
    output = Conv3D(timesteps, kernel_size=(1, 1, 1), activation=activation, padding='same')(output)

    # create model
    model = Model(input_img, output)

    o_shape = model.output_shape
    output_height = o_shape[1]
    output_width = o_shape[2]

    return model, output_height, output_width

def timebased_head(input_img, output, n_classes):
    """
    Create a segmentation head that outputs the finished model.
    Args:
        input_img: The input layer of the model to be created
        output_softmax: The softmaxed output of the last model layer
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    output = TimeDistributed(Conv2D(n_classes, (1, 1), activation=activation, padding='same'))(output)

    # create model
    model = Model(input_img, output)

    o_shape = model.output_shape
    output_height = o_shape[2]
    output_width = o_shape[3]

    return model, output_height, output_width


def lstm_head(input_img, output, n_classes):
    """
    Create a segmentation head that outputs the finished model.
    Args:
        input_img: The input layer of the model to be created
        output_softmax: The softmaxed output of the last model layer
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    output = ConvLSTM2D(n_classes, (1, 1), activation=activation, padding='same', return_sequences=True)(output)

    # create model
    model = Model(input_img, output)

    o_shape = model.output_shape # bs, ts, h, w, nc
    output_height = o_shape[2]
    output_width = o_shape[3]

    return model, output_height, output_width


def lstm_test_net(input_height, input_width, input_channels, n_classes, timesteps=None, collapse=False):
    """
    Create a simple image segmentation model based on UNet
    Args:
        input_height: The height dimension of the input images (divisible by 32)
        input_width: The widht dimension of the input images (divisible by 32)
        input_channels: The number of channels of the input images
        n_classes: The number of classes of the output images
        timesteps: A fixed number of timesteps per sequence
        collapse: Collapse the output to produce only one image
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    img_input = timebased_input(timesteps, input_height, input_width, input_channels)    

    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(img_input)
    x = TimeDistributed(BatchNormalization())(x)
    c1 = x
    
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    c2 = x

    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x)    
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)    
    c3 = x

    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x)    
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)    

    x = ConvLSTM2D(128, (3, 3), activation='relu', padding='same', return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Concatenate(axis=-1)([c3, x])

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Concatenate(axis=-1)([c2, x])

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = Concatenate(axis=-1)([c1, x])

    x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    
    if collapse:
        return collapsing_head(img_input, x, timesteps, n_classes)
    else:
        return timebased_head(img_input, x, n_classes)

def lstm_shallow(input_height, input_width, input_channels, n_classes, timesteps=None, collapse=False):
    """
    Create a fully LSTM network that does some downsampling followed by some upsampling
    Args:
        input_height: The height dimension of the input images (divisible by 32)
        input_width: The widht dimension of the input images (divisible by 32)
        input_channels: The number of channels of the input images#
        n_classes: The number of classes of the output images
    Return:
        A tuple containing the model, the height, and the width of the output 
    """
    img_input = timebased_input(timesteps, input_height, input_width, input_channels)    
    
    x = ConvLSTM2D(32, (3, 3), dropout=0.2, activation='relu', padding='same', return_sequences=True)(img_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x)    

    x = ConvLSTM2D(64, (3, 3), dropout=0.2, activation='relu', padding='same', return_sequences=True)(x)
    x = TimeDistributed(BatchNormalization())(x)
    
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = ConvLSTM2D(32, (3, 3), dropout=0.2, activation='relu', padding='same', return_sequences=True)(x)
    x = TimeDistributed(BatchNormalization())(x)

    return lstm_head(img_input, x, n_classes)
