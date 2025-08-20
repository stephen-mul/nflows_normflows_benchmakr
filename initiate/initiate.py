from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from typing import Tuple

def initiate_object(
        object_path: str,
        *args,
        **kwargs,
) -> object:
    """
    Dynamically import and instantiate an object from a given path, with optional constructor arguments.
    
    Args:
        object_path (str): The path to the object, formatted as 'module.ClassName'.
        *args: Positional arguments to pass to the class constructor.
        **kwargs: Keyword arguments to pass to the class constructor.
    
    Returns:
        object: An instance of the specified class.
    """
    module_name, class_name = object_path.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(*args, **kwargs)

def initiate_config_objects(
        config: dict,
) -> Tuple[Dataset, LightningModule]:
    """
    Instantiate data and model objects from a configuration dictionary.
    
    Args:
        config (dict): Configuration dictionary containing 'data' and 'model' keys.
    
    Returns:
        tuple: A tuple containing the instantiated data object and model object.
    """
    data_module = initiate_object(
        config['data']['class'],
        **config['data'].get('params', {})
    )
    model = initiate_object(
        config['model']['class'], 
        **config['model'].get('params', {})
    )
    
    return data_module, model