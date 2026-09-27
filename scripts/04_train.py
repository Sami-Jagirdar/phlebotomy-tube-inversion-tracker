"""Finetune yolo model on the tube inversion dataset"""

from inversion_tracker.config import load_yaml, REPO_ROOT
from ultralytics import YOLO
import argparse

def main():
    config = load_yaml("train.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=config.get("model"), help="Path to the model to finetune")
    parser.add_argument("--name", type=str, default=config.get("name"), help="Name of the experiment")
    parser.add_argument("--epochs", type=int, default=config.get("epochs"), help="Number of epochs to train for")
    parser.add_argument("--batch", type=int, default=config.get("batch"), help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=config.get("imgsz"), help="Image size for training")
    args = parser.parse_args()

    for key in ["model", "name", "epochs", "batch", "imgsz"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    
    model = YOLO(config.pop("model"))

    config['data'] = str(REPO_ROOT / config['data'])
    config['project'] = str(REPO_ROOT / config['project'])

    model.train(**config)


if __name__ == "__main__":
    main()

