from models.segmentation import seg_test_net

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
    }

    try:
        model = models[name](args)
        return model
    except:
        return None
