import argparse

from utils.config_loader import config_loader
from initiate.initiate import initiate_config_objects
from lightning import Trainer

def parse_args():
    """
    Parse command line arguments. Gets config file path.
    """
    parser = argparse.ArgumentParser(description="NormFlows Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nflows.yaml",
        help="Path to the config file",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = config_loader(args.config)
    
    # Here you would typically initialize your model, data module, etc. using the loaded config
    # For example:
    # model_class = config['model']
    # data_module_class = config['data']
    # model = model_class(**config.get('model_params', {}))
    # data_module = data_module_class(**config.get('data_params', {}))
    
    data_module, model = initiate_config_objects(config)
    print("Initiate trainer")
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        logger=True,
    )
    print("Start training")
    trainer.fit(model, data_module)
    print("Training complete")
    

if __name__ == "__main__":
    main()
