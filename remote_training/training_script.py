import os

# This is a workaround to avoid Ultralytics trying to do multi-gpu training. Despite trying to set it as
# an env var, it still tries to do multi-gpu training. This is a workaround to avoid that.
# It's necessary to do this before importing ultralytics
os.environ["RANK"] = "-1"

import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO


class TrainingScript:
    def __init__(self) -> None:
        self.image_size = [int(size) for size in os.environ["IMAGE_SIZE"].replace(" ", "").split(",")]
        self.epochs = int(os.environ["EPOCHS"])
        self.model = os.environ["MODEL"]
        self.label_studio_token = os.environ["LABEL_STUDIO_TOKEN"]
        self.label_studio_project_url = os.environ["LABEL_STUDIO_PROJECT_URL"]
        self.images_bucket_path = os.environ["IMAGES_BUCKET_PATH"]
        self.base_path = os.getcwd()
        self.bucket_path = os.environ["BUCKET_PATH"]
        self.dataset_path = Path("dataset")
        self.number_folds = int(os.environ["NUMBER_OF_FOLDS"])
        self.save_path = Path(
            self.dataset_path / f"{datetime.date.today().isoformat()}_{self.number_folds}-Fold_Cross-val")
        self.accelerator_count = int(os.environ["ACCELERATOR_COUNT"])
        self.rank = os.environ["RANK"]
        self.training_results_path = self.save_path / "training_results"
        self.fold_datasets_path = self.save_path / "folds_datasets"

    def run(self):
        gpu_available = torch.cuda.is_available()
        print(f"Checking if GPU is available: {gpu_available}")
        if not gpu_available:
            exit(1)

        os.system("mkdir dataset")
        os.system(f'gsutil -m cp -r "{self.images_bucket_path}" {str(self.dataset_path)}')
        os.system(
            f"curl -X GET {self.label_studio_project_url}/export\?exportType\=YOLO -H 'Authorization: Token {self.label_studio_token}' --output 'annotations.zip'")
        os.system("unzip annotations -d dataset")

        with open(f"{self.dataset_path}/classes.txt", "r") as f:
            class_names = f.read().splitlines()
        data = {
            "names": class_names,
            "nc": len(class_names),
            "train": f"{self.base_path}/{self.dataset_path}/train",
            "val": f"{self.base_path}/{self.dataset_path}/val",
        }
        yaml_file_path = f"{self.dataset_path}/data.yaml"
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

        augmentations = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 4.0,
            'translate': 0.1,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }

        model = YOLO("yolov8n.pt")
        model.train(data=yaml_file_path, epochs=300, imgsz=[960, 540], rect=True, device='0', **augmentations)
        metrics = model.metrics.box
        results = pd.DataFrame({'p': metrics.p, 'r': metrics.r, 'map50': metrics.all_ap[:, 0], 'map50-95': metrics.maps})
        os.system(f'gsutil -m cp -r "{self.base_path}/runs" "{self.bucket_path}"')


if __name__ == "__main__":
    training_script = TrainingScript()
    training_script.run()
