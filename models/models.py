from universal_models.models.segmentation import *
from universal_models.models.ladder import *
from universal_models.models.rnn import *

def get_model(name, *args):
    """
    Create a model from name.
    Args:
        name: Name of the model.
        vars: The arguments to create the model with.
    Returns:
        A instance of the model or None if the name did not match any available model.
    """

    models = {
        'seg_test_net' : seg_test_net,
        'seg_unet_shallow' : seg_unet_shallow,
        'seg_unet' : seg_unet,
        'seg_net_shallow' : seg_net_shallow,
        'seg_net' : seg_net,
        'seg_ladder_shallow' : seg_ladder_shallow,
        'lstm_unet' : lstm_unet,
        'lstm_shallow' : lstm_shallow, 
        'lstm_full' : lstm_full,
        'lstm_next_frame' : lstm_next_frame,
    }

    try:
        model = models[name](*args)
        return model
    except Exception as e:
        print("Error while creating model:", e)
        return None
