import yaml

def config_loader(config_path: str) -> dict:
    """
    Function to load a config file
    Args:
        config_path (str): Path to the config file
    Returns:
        dict: Dictionary containing the config
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ### check if config includes "include" key and load it
    if "include" in config.keys():
        include_path = config["include"]
        with open(include_path, 'r') as f:
            include_config = yaml.load(f, Loader=yaml.FullLoader)
        # Merge include_config into config, giving precedence to include_config
        include_config.pop("include", None)  # avoid recursive include key pollution
        config = {**include_config, **config} # main config takes precedence
        ### remove include key from config
        config.pop("include", None)
    return config