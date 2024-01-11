import logging
import os

# This is a workaround to avoid Ultralytics trying to do multi-gpu training. Despite trying to set it as
# an env var, it still tries to do multi-gpu training. This is a workaround to avoid that.
# It's necessary to do this before importing ultralytics
os.environ["RANK"] = "-1"

import datetime
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold
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
        self._check_if_gpu_is_available()
        self._prevent_multi_gpu_training()

        folds_yamls = self._prepare_dataset()

        for k in range(self.number_folds):
            model = self._train_model_fold(folds_yamls, k)
            self._save_model_metrics(k, model)

        self._export_results()

    def _export_results(self):
        os.system(f'gsutil -m cp -r "{self.save_path}" "{self.bucket_path}"')

    def _save_model_metrics(self, k, model):
        metrics = model.metrics.box
        results = pd.DataFrame(
            {
                "p": metrics.p,
                "r": metrics.r,
                "map50": metrics.all_ap[:, 0],
                "map50-95": metrics.maps,
            }
        )
        results.to_csv(f"{self.training_results_path}/{self._fold_name(k)}/metrics.csv")

    def _train_model_fold(self, folds_yamls, k):
        dataset_yaml = folds_yamls[k]
        augmentations = self._augmentations()
        model = YOLO(self.model)
        model.train(
            data=dataset_yaml,
            epochs=self.epochs,
            imgsz=self.image_size,
            rect=(self.image_size[0] != self.image_size[1]),
            device=self._get_device(),
            project=str(self.training_results_path),
            name=self._fold_name(k),
            **augmentations,
        )
        return model

    def _augmentations(self):
        augmentations = {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 4.0,
            "translate": 0.1,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }
        return augmentations

    def _prepare_dataset(self):
        class_names, images, annotations = self._download_dataset()
        folds_yamls = self.create_k_folds(annotations, class_names, images)
        return folds_yamls

    def create_k_folds(self, annotations, class_names, images):
        file_names = [annotation.stem for annotation in annotations]
        class_indices = list(range(len(class_names)))
        labels_per_image = pd.DataFrame([], columns=class_indices, index=file_names)
        for label in annotations:
            label_counter = Counter()
            with open(label, "r") as lf:
                lines = lf.readlines()
            for line in lines:
                label_counter[int(line.split(" ")[0])] += 1
            labels_per_image.loc[label.stem] = label_counter
        labels_per_image = labels_per_image.fillna(0.0)
        k_fold_generator = KFold(n_splits=self.number_folds, shuffle=True, random_state=20)
        folds_indices = list(k_fold_generator.split(labels_per_image))
        folds = [self._fold_name(n) for n in range(self.number_folds)]
        folds_df = pd.DataFrame(index=file_names, columns=folds)
        for index, (train, val) in enumerate(folds_indices):
            folds_df[self._fold_name(index)].loc[labels_per_image.iloc[train].index] = "train"
            folds_df[self._fold_name(index)].loc[labels_per_image.iloc[val].index] = "val"
        fold_label_distribution = pd.DataFrame(index=folds, columns=class_indices)
        for n, (train_indices, val_indices) in enumerate(folds_indices):
            train_totals = labels_per_image.iloc[train_indices].sum()
            val_totals = labels_per_image.iloc[val_indices].sum()
            ratio = val_totals / (train_totals + 1e-7)
            fold_label_distribution.loc[self._fold_name(n)] = ratio
        folds_yamls = []
        for fold in folds_df.columns:
            fold_dir = self.fold_datasets_path / fold
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
            (fold_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

            dataset_yaml = fold_dir / f"dataset.yaml"
            folds_yamls.append(dataset_yaml)

            with open(dataset_yaml, "w") as ds_y:
                yaml.safe_dump(
                    {
                        "path": f"{self.base_path}/{fold_dir.as_posix()}",
                        "train": "train",
                        "val": "val",
                        "names": class_names,
                    },
                    ds_y,
                )
        for image, label in zip(images, annotations):
            for fold, train_or_val in folds_df.loc[image.stem].items():
                shutil.copy(image, self.fold_datasets_path / fold / train_or_val / "images" / image.name)
                shutil.copy(label, self.fold_datasets_path / fold / train_or_val / "labels" / label.name)
        folds_df.to_csv(self.fold_datasets_path / "kfold_datasplit.csv")
        fold_label_distribution.to_csv(self.fold_datasets_path / "kfold_label_distribution.csv")
        return folds_yamls

    def _download_dataset(self):
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.save_path.mkdir(parents=True, exist_ok=True)
        os.system(f'gsutil -m cp -r "{self.images_bucket_path}" {str(self.dataset_path)}')
        os.system(
            f"curl -X GET {self.label_studio_project_url}/export\?exportType\=YOLO -H 'Authorization: Token {self.label_studio_token}' --output 'annotations.zip'"
        )
        os.system(f"unzip -o annotations -d {str(self.dataset_path)}")
        with open(f"{self.dataset_path}/classes.txt", "r") as f:
            class_names = f.read().splitlines()
        yaml_data = {
            "names": class_names,
            "nc": len(class_names),
            "train": f"{self.base_path}/{self.dataset_path}/train",
            "val": f"{self.base_path}/{self.dataset_path}/val",
        }
        yaml_file_path = f"{self.dataset_path}/data.yaml"
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)
        annotations = sorted(self.dataset_path.rglob("*labels/*.txt"))
        images = []
        for extension in [".jpg", ".jpeg", ".png"]:
            images.extend(sorted((self.dataset_path / "images").rglob(f"*{extension}")))
        return class_names, images, annotations

    def _prevent_multi_gpu_training(self):
        logging.info(f"Checking RANK: {self.rank}")
        if self.rank != "-1":
            logging.error("Trying multi gpu training. Exiting.")
            exit(1)

    def _check_if_gpu_is_available(self):
        gpu_available = torch.cuda.is_available()
        logging.info(f"Checking if GPU is available: {gpu_available}")
        if self.accelerator_count > 0 and not gpu_available:
            logging.error(f"GPU is not available, accelerator count: {self.accelerator_count}")
            exit(1)

    def _fold_name(self, k):
        return f"fold_{k + 1}"

    def _get_device(self):
        return '0' if self._check_if_gpu_is_available() else 'cpu'


if __name__ == "__main__":
    training_script = TrainingScript()
    training_script.run()
