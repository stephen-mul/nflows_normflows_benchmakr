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
        include_paths = config["include"] ## should be formatted as list of paths
        if not isinstance(include_paths, list):
            ### raise error
            raise ValueError("Include key must be a list of paths")
        for include_path in include_paths:
            with open(include_path, 'r') as f:
                include_config = yaml.load(f, Loader=yaml.FullLoader)
            # Merge include_config into config, giving precedence to include_config
            include_config.pop("include", None)  # avoid recursive include key pollution
            config = {**include_config, **config} # main config takes precedence
            ### remove include key from config
            config.pop("include", None)
    return config