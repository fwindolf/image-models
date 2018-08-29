# Adapted from https://github.com/rinuboney/ladder/
from keras import backend as K
from keras import losses
from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.engine.topology import Layer

from image_models.models.segmentation import seg_input, seg_head

import os

class Noise(Layer):
    """
    Layer that adds gaussian noise to the input
    """
    def __init__(self, noise_level=0.05, **kwargs):
        self.noise_level = noise_level
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        noise = K.random_normal(x.shape, mean=0.0, stddev=self.noise_level)
        return x + noise
    
    def compute_output_shape(self, input_shape):
        return input_shape

class Denoise(Layer):
    """
    Layer with denoising function g(.,.) defined in the ladder network paper.
    The estimated latent z_est is calculated by denoising the noisy latent z_noisy 
    with the help of the distribution of the projection vector u.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        z_noisy_shape = input_shape[0]
        u_shape = input_shape[1] # n, c, h, w
        assert(z_noisy_shape[1:] == u_shape[1:])

        # one weights vector for every activation pixel
        a_shape = (u_shape[2], u_shape[3], )

        self.a1 = self.add_weight(name='a01', shape=a_shape, initializer='zeros', trainable=True)
        self.a2 = self.add_weight(name='a02', shape=a_shape, initializer='ones', trainable=True)
        self.a3 = self.add_weight(name='a03', shape=a_shape, initializer='zeros', trainable=True)
        self.a4 = self.add_weight(name='a04', shape=a_shape, initializer='zeros', trainable=True)
        self.a5 = self.add_weight(name='a05', shape=a_shape, initializer='zeros', trainable=True)
        self.a6 = self.add_weight(name='a06', shape=a_shape, initializer='zeros', trainable=True)
        self.a7 = self.add_weight(name='a07', shape=a_shape, initializer='ones', trainable=True)
        self.a8 = self.add_weight(name='a08', shape=a_shape, initializer='zeros', trainable=True)
        self.a9 = self.add_weight(name='a09', shape=a_shape, initializer='zeros', trainable=True)
        self.a10 = self.add_weight(name='a10', shape=a_shape, initializer='zeros', trainable=True)

        super().build(input_shape)

    def call(self, x):
        assert(len(x) == 2)
        z_noisy, u = x

        # a2, a7 must be one for the output to be 0, but still u having some effect
        mu = self.a1 * sigmoid(self.a2 * u + self.a3) + self.a4 * u + self.a5
        vu = self.a6 * sigmoid(self.a7 * u + self.a8) + self.a9 * u + self.a10

        # denoising function g as specified in the paper
        z_est = (z_noisy - mu) * vu + mu
        return z_est

    def compute_output_shape(self, input_shape):
        return input_shape[1] # shaped like u 

class DeBatchNormalization(Layer):
    """
    Rescale the estimated latents z_est with the parameters of the
    gaussian from the input latent distribution z_preac
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        assert(len(x) == 2)
        z_est, z_preac = x
        # Extract parameters of gaussian
        mu = K.mean(z_preac)
        sigma = K.std(z_preac)
        return (z_est - mu) / sigma

    def compute_output_shape(self, input_shape):
        return input_shape[0] # like z_est

class TrainableRescale(Layer):
    """
    Layer with learnable rescaling parameters beta and gamma.
    """
    def __init__(self, activation='relu', train_beta=True, train_gamma=False, **kwargs):
        self.train_beta = train_beta
        self.train_gamma = train_gamma
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = input_shape[1:]
        self.beta = self.add_weight(name='beta', shape=weight_shape, initializer='zeros', trainable=self.train_beta)
        self.gamma = self.add_weight(name='gamma', shape=weight_shape, initializer='ones', trainable=self.train_gamma)
        super().build(input_shape)

    def call(self, x):
        output = self.gamma * (x + self.beta)
        if self.activation is not None:
            output = activations.get(self.activation)(output)
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape

class LateralConnectionLoss(Layer):
    """
    Layer that provides the denoising loss term
    Calculates and returns the binary crossentropy loss between the noisy (x[0]) and
    denoised (x[1]) inputs.
    """
    def __init__(self, weight=0.01, **kwargs):
        self.weight = weight
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, x):
        assert(len(x) == 2)
        noisy, denoised = x
        mse = losses.mean_squared_error(noisy, denoised)
        loss = K.sum(mse)
        #crossentropy = losses.binary_crossentropy(noisy, denoised)
        #loss = K.sum(crossentropy)
        self.add_loss(loss, x)
        return loss

    def compute_output_shape(self, input_shape):
        return 1 # single scalar

class EncoderBlockBase(Layer):
    def __init__(self, scope, noise_level=0.05, **kwargs):
        self.scope = scope
        self.noise_level = noise_level
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.batchnorm = BatchNormalization(name=self.scope + '_batchnorm')       
        self.noise = GaussianNoise(stddev=self.noise_level, name=self.scope + '_noise')
        self.rescale = TrainableRescale(name=self.scope + '_rescale', activation='relu')

        super().build(input_shape)
    
    def call(self, x, add_noise):
        z_preac = x
        z = self.batchnorm(z_preac)
        # only add noise if thats the corruped path, but share weights with clean path
        if add_noise:
            z = self.noise(z)
        h = self.rescale(z)

        return [z_preac, z, h]
    
    def compute_output_shape(self, input_shape):
        return [input_shape] * 3

class EncoderLayerConv(EncoderBlockBase):
    def __init__(self, scope, filters, kernel_size, strides=None, noise_level=0.05, **kwargs):        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides if strides is not None else (1, 1)

        super().__init__(scope, noise_level, **kwargs)

    def build(self, input_shape):
        self.conv = Conv2D(self.filters, self.kernel_size, strides=self.strides, padding='same', name=self.scope + '_conv')
        super().build(input_shape)

    def call(self, x, add_noise=False):
        h_prev = x
        z_preac = self.conv(h_prev)
        return super().call(z_preac, add_noise=add_noise)

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(self.conv.output_shape)

class EncoderLayerPool(EncoderBlockBase):
    def __init__(self, scope, pool_size=(2, 2), strides=(2, 2), noise_level=0.05, **kwargs):
        self.pool_size = pool_size
        self.strides = strides
        super().__init__(scope, noise_level, **kwargs)

    def build(self, input_shape):
        self.pool = MaxPooling2D(self.pool_size, strides=self.strides, padding='same', name=self.scope + '_pool')
        super().build(input_shape)

    def call(self, x, add_noise=False):
        h_prev = x
        z_preac = self.pool(h_prev) 
        return super().call(z_preac, add_noise=add_noise)

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(self.pool.output_shape)

class EncoderBlock(EncoderBlockBase):
    """
    Block analog to a VGG block
    Conv - Conv - (Conv) - MaxPool
    """
    def __init__(self, scope, filters, kernel_size, num_convs, noise_level=0.05, trainable=True, use_bias=True, **kwargs):        
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_convs = num_convs
        self.trainable = trainable
        self.use_bias = use_bias

        super().__init__(scope, noise_level, **kwargs)

    def build(self, input_shape):
        self.convs = []
        for i in range(self.num_convs): 
            self.convs.append(Conv2D(self.filters, self.kernel_size, padding='same', name=self.scope + '_conv' + str(i + 1), 
                                     use_bias=self.use_bias, trainable=self.trainable))

        self.pool = MaxPooling2D((2, 2), strides=(2, 2), name=self.scope + '_pool')
        super().build(input_shape)

    def call(self, x, add_noise=False):
        h_prev = x
        # Calculate the preactivation of this layer
        
        z_preac = h_prev
        for conv in self.convs:
            z_preac = conv(z_preac)

        z_preac = self.pool(z_preac)    
        # The base class takes care of the added noise, batchnorm, ...
        return super().call(z_preac, add_noise=add_noise)

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(self.pool.output_shape)
    
class DecoderBlockBase(Layer):
    def __init__(self, scope, **kwargs):
        self.scope = scope
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.batchnorm = BatchNormalization(name=self.scope +'_batchnorm')
        self.denoise = Denoise(name=self.scope + '_denoise')
        self.debatchnorm = DeBatchNormalization(name=self.scope + '_debatchnorm')
        super().build(input_shape)

    def call(self, x):
        assert(len(x) == 3)
        u_prebn, z_noisy, z_preac = x        
        u = self.batchnorm(u_prebn)
        z_est = self.denoise([z_noisy, u])
        z_est_bn = self.debatchnorm([z_est, z_preac]) # use z_preac to get mu, sigma for debatchnorm

        return [u, z_est, z_est_bn]

    def compute_output_shape(self, input_shape):
        single_shape = input_shape[0] # 3 outputs with same shape as the first input
        return list([single_shape] * 3)

class DecoderLayerConv(DecoderBlockBase):
    def __init__(self, scope, filters, kernel_size, strides=None, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides if strides is not None else (1, 1)
        super().__init__(scope, **kwargs)

    def build(self, input_shape):
        self.tconv = Conv2DTranspose(self.filters, self.kernel_size, strides=self.strides, name=self.scope + '_tconv',
                                     padding='same', use_bias=False)
        super().build(input_shape)
    
    def call(self, x):
        assert(len(x) == 3)
        z_est_prev, z_noisy, z_preac = x
        u_prebn = self.tconv(z_est_prev)

        return super().call([u_prebn, z_noisy, z_preac])

    def compute_output_shape(self, input_shape):
        base_input_shape = [self.tconv.output_shape, input_shape[1], input_shape[2]]
        return super().compute_output_shape(base_input_shape)

class DecoderLayerUpsample(DecoderBlockBase):
    def __init__(self, scope, strides=None, **kwargs):
        self.strides = strides if strides is not None else (1, 1)
        super().__init__(scope, **kwargs)

    def build(self, input_shape):
        self.upsample = UpSampling2D(self.strides, name=self.scope + '_upsample')
        self.zeropad = ZeroPadding2D((1, 1), name=self.scope + '_zeropad')
        filters = input_shape[0][1] # channels of z_est_prev
        self.conv = Conv2D(filters, (3, 3), name=self.scope + '_conv', padding='valid')
        
        super().build(input_shape)
    
    def call(self, x):
        z_est_prev, z_noisy, z_preac = x
        u_prebn = self.upsample(z_est_prev)
        u_prebn = self.zeropad(u_prebn)
        u_prebn = self.conv(u_prebn)
        
        return super().call([u_prebn, z_noisy, z_preac])

    def compute_output_shape(self, input_shape):
        base_input_shape = [self.conv.output_shape, input_shape[1], input_shape[2]]
        return super().compute_output_shape(base_input_shape)

class DecoderBlock(DecoderBlockBase):
    def __init__(self, scope, filters, upsample=True, **kwargs):
        self.filters = filters
        self.do_upsample = upsample
        self.conv_pad = 'valid' if self.do_upsample else 'same' # make sure we keep dimensions
        super().__init__(scope, **kwargs)

    def build(self, input_shape):
        if self.do_upsample:
            self.upsample = UpSampling2D((2, 2), name=self.scope + '_upsample')
        
       
        self.conv = Conv2D(self.filters, (3, 3), name=self.scope + '_conv', padding='same')
        super().build(input_shape)
    
    def call(self, x):
        z_est_prev, z_noisy, z_preac = x
        if self.do_upsample:
            u_prebn = self.upsample(z_est_prev)

        u_prebn = self.conv(u_prebn)
        return super().call([u_prebn, z_noisy, z_preac])

    def compute_output_shape(self, input_shape):
        base_input_shape = [self.conv.output_shape, input_shape[1], input_shape[2]]
        return super().compute_output_shape(base_input_shape)

def seg_ladder_shallow(input_height, input_width, input_channels, n_classes, noise_level=0.05, lambdas=[10., 1., 1.]):
    """
    Ladder network implementation with convolutional layers
    """
    raise RuntimeWarning("This network currently does not learn.")

    z_clean = {} # clean latents
    z_noisy = {} # latents with added gaussian noise
    z_preac = {} # latents pre activation
    z_estim = {} # estimated (denoised) latents
    z_estbn = {} # batch normalized, estimated latents
    
    h_clean = {} # clean activations
    h_noisy = {} # noisy activations

    u = {} # projection vector

    l_loss = {} # loss from lateral connections

    # Prepare Input
    img_input = seg_input(input_height, input_width, input_channels)

    z_preac[0] = BatchNormalization(axis=1, name='input_batchnorm')(img_input)
    h_clean[0] = z_clean[0] = z_preac[0]
    h_noisy[0] = z_noisy[0] = GaussianNoise(stddev=0.25, name='input_noise')(z_preac[0]) # input noise bigger than elsewhere

    # Encoder Block 
    enc1 = EncoderBlock('enc1', 64, 3, num_convs=2, noise_level=10*noise_level) 
    z_preac[1], z_clean[1], h_clean[1] = enc1(h_clean[0])
    _, z_noisy[1], h_noisy[1] = enc1(h_noisy[0], add_noise=True)

    # Encoder Block 
    enc2 = EncoderBlock('enc2', 128, 3, num_convs=2, noise_level=noise_level) 
    z_preac[2], z_clean[2], h_clean[2] = enc2(h_clean[1])
    _, z_noisy[2], h_noisy[2] = enc2(h_noisy[1], add_noise=True)

    # Prepare for Decoder
    u[2] = BatchNormalization(name='conn_batchnorm')(h_noisy[2])
    z_estim[2] = Denoise(name='conn_denoise')([u[2], z_noisy[2]])
    z_estbn[2] = DeBatchNormalization(name='conn_debatchnorm')([z_estim[2], z_preac[2]])
    l_loss[2] = LateralConnectionLoss(name='conn_loss', weight=lambdas[2])([z_clean[2], z_estbn[2]])
    
    u[1], z_estim[1], z_estbn[1] = DecoderBlock('dec2', 64)([z_estim[2], z_noisy[1], z_preac[1]])
    l_loss[1] = LateralConnectionLoss(name='dec2_loss', weight=lambdas[1])([z_clean[1], z_estbn[1]])

    # Decoder Block
    u[0], z_estim[0], z_estbn[0] = DecoderBlock('dec1', input_channels)([z_estim[1], z_noisy[0], z_preac[0]])
    l_loss[0] = LateralConnectionLoss(name='dec1_loss', weight=lambdas[0])([z_clean[0], z_estbn[0]])
    
    # output for reconstruction (Output inactive, as the reconstruction cost is added as a regularizer in LateralConnectionLoss)    
    # z_estimate = z_estbn[0] # denoised input    
    # z_estimate = Conv2D(input_channels, (1, 1), padding='same', activation='sigmoid', name='reconstruction_head')(z_estimate)
    
    # Head for Supervised Learning
    o = h_clean[2]
    o = UpSampling2D((2, 2), name='up1_upsample')(o)
    o = Conv2D(128, (3, 3), padding='same', name='up1_conv1')(o)
    o = Conv2D(128, (3, 3), padding='same', name='up1_conv2')(o)
    o = BatchNormalization(name='up1_bn')(o)

    o = UpSampling2D((2, 2), name='up2_upsample')(o)
    o = Conv2D(64, (3, 3), padding='same', name='up2_conv1')(o)
    o = Conv2D(n_classes, (3, 3), padding='same', activation='softmax', name='segmentation_head')(o)

    o_shape = Model(img_input, o).output_shape
    output_height = o_shape[1]
    output_width = o_shape[2]

    return Model(img_input, o), output_height, output_width
