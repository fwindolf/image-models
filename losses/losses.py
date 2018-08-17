import keras.backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy

def dice_coeff(y_true, y_pred, smooth=1.):
    """
    Dice coefficient from http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf

    Converges towards one when the intersection between predicitions and labels equals to the
    sum of the labels, providing some equality measure of one-hot predictions that takes into
    account the cardinality of the labels. In other words, with highly unbiased datasets it 
    puts weight on the less frequent classes. 
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    """
    Loss based on the Dice coefficient.
    As the coefficient converges to 1, we need to substract the coefficient from 1 to get 
    a minimizable loss.
    """
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    """
    A combination of the binary crossentropy and the dice loss.
    Enables fast learning due to the BCE but with additional weighting to the less frequent
    class.
    """
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def gen_dice_coeff(y_true, y_pred, eps=1e-7, smooth=0):
    """
    Multiclass dice coefficient as described in https://arxiv.org/pdf/1707.03237.pdf

    w_l  is the weight for label set l, providing invariance to different proportions of label sets.
    r_ln is the label for the n-th data point for label set l.
    p_ln is the predicted probability for the n-th data point for label set l.
    """

    axes = tuple(range(0, len(y_pred.shape) - 1)) # everything but the channels
    w = 1./(K.square(K.sum(y_true, axis=axes)) + eps) # for stability
    
    num = K.sum(w * K.sum(y_true * y_pred, axis=axes)) 
    den = K.sum(w * K.sum(y_true + y_pred, axis=axes))

    score = (2. * num + smooth) / (den + eps + smooth)
    return score
    
    """
    # This does not work rn, it seems to only work for 1 class...
    
    axes = tuple(range(1, len(y_pred.shape) - 1)) # not over batchsize, channels
    w = 1./(K.square(K.sum(y_true, axis=axes)) + eps) # for stability

    # keep batch dimension intact, sum over channels
    num = K.sum(w * K.sum(y_true * y_pred, axis=axes), axis=-1) 
    den = K.sum(w * K.sum(y_true + y_pred, axis=axes), axis=-1)

    score = 2. * num / den
    return K.mean(score)
    """
    
def gen_dice_loss(y_true, y_pred):
    """
    Loss based on the Generalized Dice coefficient.
    As the coefficient converges to 1, we need to substract the coefficient from 1 to get 
    a minimizable loss.
    """
    loss = 1 - gen_dice_coeff(y_true, y_pred)
    return loss

def ce_dice_loss(y_true, y_pred):
    """
    A combination of the binary crossentropy and the dice loss.
    Enables fast learning due to the CE term but with additional weighting to the less frequent
    class.
    """
    loss = categorical_crossentropy(y_true, y_pred) + gen_dice_loss(y_true, y_pred)
    return loss

def focal_loss(y_true, y_pred, gamma=2):
    """
    Focal Loss from https://arxiv.org/pdf/1708.02002.pdf

    Puts more weigth on badly classified examples (y_pred < .5) in order
    to output highly accurate predictions
    """
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1. - eps)
    return -K.sum(K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred), axis=-1)