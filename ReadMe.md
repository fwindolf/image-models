# Universal-Models

This repository aims to provide different models and architectures to make learning from image based data easier.

# Models

There exist different models for image segmentation, recurrent image segmentation and ladder networks.

All models can be created via the `get_model()` method, where the name of the model is provided as the first argument.

## Image Segmentation


Image segmentation tries to predict non-overlapping areas of classes from rgb images. The training data thus consists of images as input, and categorical ground truth data (image based layers that contain 1s where the class that the layer represents is active, and 0s otherwise).

Networks are called with the following parameters:
| Parameter | Meaning |
| ----------| --------| 
| input_height | Height of the input image. |
| input_width  | Width of the input image. |
| input_channels | Number of channels in the input image (can also be the number of greyscal images stacked ontop of each other). |
| n_classes | The number of classes in the ground truth data. |

All networks return a tuple containing the model, the output_height and the output_width of the network.

Classically, networks for image segmentation consist of an encoder and a decoder. With every encoder-block, the dimensionality of the features gets reduced (say, internal representation of the image in features is shrinked, most of the time by a factor of 2). This leads to the network to create more and more abstract (high-level) features of the original image.

The following decoder will then increase dimensionality again to restore the original image dimensions, while retaining the important features for classes. This leads to the output of the network being a dense prediction of class labels.

All of the segmentation networks can also be used for prediction of greyscale images (eg, of the next frame in a sequence) by using greyscale images as input and ground truth data that is also greyscale. The models will then be created with sigmoid activation instead of softmax.

### TestNet

A simple network consisting of 3 convolutional encoding and decoding blocks with batchnormalizations throughout the network. It is big enough to do simple classification tasks and works very fast due to the small number of parameters.

### UNet

There exist two variants of UNet, a normal and a shallow one. The normal variant tries to stick to the architecture provided in the [paper](https://arxiv.org/abs/1505.04597), while the shallow variant aims to reduce the number of parameters for simpler tasks while retaining the idea behind UNet.

UNet saves feature representations in the encoder to re-use them in the decoder by concatenating them to the decoder features, making learning very fast whenever the output resembles the input in some way (eg. predict particles and their attributes - the particle position are very low level features that can be used for more accurate predictions in the last decoder layer).

### SegNet

Again, two variants of SegNet exist, a normal and a shallow one. The normal variant is implemented after the details given in the SegNet [paper](https://arxiv.org/abs/1511.00561). It uses blocks with simple chains of Conv-Conv-Conv-Pool or US-Conv-Conv-Conv for blocks, batch-normalizing the data after every operation on features.


## Recurrent Image Segmentation

For classification on sequence based data (videos, simulations, ...), image segmentation provides pretty accurate results. However, those can be improved by involving temporal data into the prediction process, making predictions more accurate, and also less 'jumpy' in sequence.

Networks are called with the following parameters:
| Parameter | Meaning |
| ----------| --------| 
| input_height | Height of the input image |
| input_width  | Width of the input image |
| input_channels | Number of channels in the input image (can also be the number of greyscal images stacked |ontop of each other)
| n_classes | The number of classes in the ground truth data |
| stateful  | Create a stateful network. This means that the network keeps the state between batches. Use `model.reset_states()` to clear state. |
| timesteps | The number of timesteps in the data. Can be left at None (default) for stateless networks. |
| batchsize | The number of batches used in training. Can be also left at None (default) for training stateless networks. |

Most of the models are implemented using TimeDistributed layers, which wrap other layers in a way that they can use input data with varying timesteps and apply to the wrapped layer to data of each timestep. This means the number of parameters does not change when varying the timesteps of the data.

The general architecture stays the same however, again using a encoder and decoder network (with layers wrapped in TimeDistributed). The connecting part between encoder and decoder (which didn't involve any operation on the features for normal image segementation) is realized using a convolutional LSTM block. This block keeps a hidden state over time, that might (or might not) store information about features from previous timesteps, which the network then could (or could not) use to increase accuracy of predictions (considering it was trained well).

The network will predict a sequence of the length (timesteps) of the input data. 

To use the model during training on sequences, don't use stateful networks. For inference however, that is what you want most of the time. To convert a model to stateful, the easiest way is to just save the weights with `model.save_weights(<filename>)`  and reload them into a network with the same image dimensions but statfulness activated (This involves setting the batchsize and the timesteps parameter). For prediction of sequences it is the easiest to set timesteps to 1 and batchsize to as high as possible (GPU memory is the bottlneck here) to make predictions faster.

### LSTM UNet

A network witht he same architecture as UNet, but with an added LSTM connection between the encoder and decoder. Makes use of the same concepts like UNet regarding using features from the encoder in the decoder, but additionally takes into account features from previous timesteps. It should be applied to the same kind of data than UNet, whenever the data is also sequence based.

### LSTM Shallow

Theoretically, a simple convolutional LSTM cell could be used to predict segmentation on images. However, this invovles a huge amount of parameters when operating on original image dimensions. Thus, the shallow LSTM uses convolutional downsampling on the image once, applies the LSTM cell and then does some convolutional upsampling. This leads to network with low receptive field (it cannot abstract features much, or use big features or features from far apart regions of the image to craft the predictions). It can be used when working on data that has only lokal features, and is sequence based.

### LSTM Full

The full LSTM works similar to the shallow LSTM, using also an encoder and decoder to reduce dimensionality a bit. Additionally, it applies stacked LSTM cells. That could help when working on data where 2nd order temporal information could be beneficial to make predicitons (eg. acceleration, ...), which is very hard for a single LSTM cell to learn.

### LSTM Next Frame

This architecture was used in a [paper](https://arxiv.org/abs/1609.06377) that tried to predict future frames in video sequences. It tried to tackle the problem that when feeding predictions back into the network, the predictions accumulate uncertainty over time and thus get more and more blurry. 
The Depth-to-space blocks used are based on the Tensorflow implementaiton wrapped in Keras layers and increase dimensionality by cutting the features in 4 parts in the channel dimensions and stitching them together (just like 4 panes in a window).
Theoretically that architecture should work well on predicting future frames in image sequences, but produces artifacts during training, as well as the prediction quality being overall not that great. Additional convolutional layers (or more filters) could help making it work better.

## Ladder

The ladder network described in this [paper](https://arxiv.org/abs/1507.02672) is considered a breakthrough in semi-supervised learning. It uses vasts amounts of images where ground truth data exists only sparsely, and can learn from both the labeling as well as the general structure of the images. However, it never really was able to produce breakthrough results in real-world applications.

The architecture combines a denoising autoencoder (which tries to reconstruct the original image from a noisy version) with denoising lateral connections (that enable feature transfer between layers) with supervised learning by taking into account a loss with respect to the ground truth in the most abstract layer whenever ground truth data are available.

The available implementation does enable meaningful learning, there seems to be a bug in it somewhere.

# Losses

As most of the data is biased towards a class or the background, the usual `categorical_crossentropy` usually does not enable fast learning. Thus the repository contains two other losses that take into account class distributions and are more suited for image segmentation.

A good rundown of different loss functions in this settings can be read in this [paper](https://arxiv.org/abs/1707.03237)

## Dice Loss

Dice loss tries to minimize the overlap of predicion and ground truth over the cardinality of both vectors. A weighting term (not described in the paper) can additionally reduce the class bias in the data.

It is based on the dice coefficient (which can be also used as a metric), that measures how good the overlap is, and converges to 1 for two classes.

$D= 2\frac{\Sigma_i^N p_i  g_i}{\Sigma_i^N p_i² + \Sigma_i^N g_i²}$ with a additional class weiths $w=\frac{1}{\Sigma_i^N g_i² + \epsilon}$

Combined with a binary crossentropy loss (bce_dice_loss), it enables fast learning eg for predicitons of future frames.

## Generalized Dice Loss

The variant for multiclass dice coefficient generalizes to formula for more than two classes, taking also into account the distribution over multiple classes. 

It can be combined with a crossentropy loss (ce_dice_loss) to do the same things dice loss does for more image segmentation

## Focal Loss

Another loss that is supposed to work well on image segmentation problems is the focal loss. It puts more learning effort onto hard (badly classified) examples and prevents easy negative examples from overwhelming the loss.

$F(p_i) = -(1-p_i)^\gamma log(p_i)$ with tunable $\gamma$

However, the implementation is not tested as of now.

