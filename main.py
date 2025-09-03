import argparse
import time

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
    
    data_module, model = initiate_config_objects(config)

    print("Initiate trainer")
    trainer = Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        logger=config['trainer']['logger'],
    )
    print("Start training")
    start_time = time.time()
    trainer.fit(model, data_module)
    stop_time = time.time()
    print(f"Training time: {stop_time - start_time:.2f} seconds")
    print(f"Time per epoch: {(stop_time - start_time)/config['trainer']['max_epochs']:.2f} seconds")
    print("Training complete")

    print("Inference")
    model.inference(outtput_dir=data_module.plotting_dir)
    print("Inference complete")
    

if __name__ == "__main__":
    main()
