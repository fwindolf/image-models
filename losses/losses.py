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


def gen_dice_coeff(y_true, y_pred):
    """
    Multiclass dice coefficient as described in https://arxiv.org/pdf/1707.03237.pdf

    w_l  is the weight for label set l, providing invariance to different proportions of label sets.
    r_ln is the label for the n-th data point for label set l.
    p_ln is the predicted probability for the n-th data point for label set l.
    """
    w = 1./(K.square(K.sum(y_true, axis=(0, 1, 2))) + 1e-6) # for stability
    
    num = K.sum(w * K.sum(y_true * y_pred, axis=(0, 1, 2))) 
    den = K.sum(w * K.sum(y_true + y_pred, axis=(0, 1, 2)))

    score = 2. * num / den
    return score
    
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