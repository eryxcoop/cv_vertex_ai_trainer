import datetime
import gc
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path

# import mlflow
import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold

# This is a workaround to avoid Ultralytics trying to do multi-gpu training. Despite trying to set it as
# an env var, it still tries to do multi-gpu training. This is a workaround to avoid that.
# It's necessary to do this before importing ultralytics
os.environ["RANK"] = "-1"
from ultralytics import YOLO
from label_studio_sdk.converter import Converter


class TrainingScript:
    def __init__(self) -> None:
        self.image_size = [int(size) for size in os.environ["IMAGE_SIZE"].replace(" ", "").split(",")]
        self.epochs = int(os.environ["EPOCHS"])
        self.model = os.environ["MODEL"]
        self.obb = os.environ["OBB"] == "True"
        self.label_studio_token = os.environ["LABEL_STUDIO_TOKEN"]
        self.label_studio_project_url = os.environ["LABEL_STUDIO_PROJECT_URL"]
        self.images_bucket_path = os.environ["IMAGES_BUCKET_PATH"]
        self.base_path = os.getcwd()
        self.bucket_path = os.environ["BUCKET_PATH"]
        self.dataset_path = Path("dataset")
        self.number_folds = int(os.environ["NUMBER_OF_FOLDS"])
        self.use_kfold = (os.environ["USE_KFOLD"] == "True")
        self.save_path = self._define_save_path()
        self.accelerator_count = int(os.environ["ACCELERATOR_COUNT"])
        self.rank = os.environ["RANK"]
        self.training_results_path = self.save_path / "training_results"
        self.fold_datasets_path = self.save_path / "folds_datasets"
        self.single_dataset_path = self.save_path / "single_dataset"
        self.mlflow_experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]

    def run(self):
        self._check_if_gpu_is_available()
        self._prevent_multi_gpu_training()

        class_names, images, annotations = self._download_dataset()
        images, annotations = self._match_images_and_annotations(images, annotations)
        datasets_path = self.fold_datasets_path if self.use_kfold else self.single_dataset_path

        if self.use_kfold:
            dataset_yaml_list = self._create_k_folds(annotations, class_names, images, datasets_path)
            for fold_number in range(len(dataset_yaml_list)):
                dataset_yaml = dataset_yaml_list[fold_number]
                model_name = self._fold_name(fold_number)
                model = self._train_model(dataset_yaml, model_name)
                self._save_model_metrics(model_name, model)
                self._clean_gpu_cache()  # This is necessary to avoid running out of memory
        else:
            dataset_yaml = self._create_single_dataset(annotations, class_names, images, datasets_path)
            model_name = "single_model"
            model = self._train_model(dataset_yaml, model_name)
            self._save_model_metrics(model_name, model)

        self._export_results()

    def _export_results(self):
        os.system(f'gsutil -m cp -r "{self.save_path}" "{self.bucket_path}"')

    def _save_model_metrics(self, fold_name, model):
        metrics = model.metrics.box
        results = pd.DataFrame(
            {
                "p": metrics.p,
                "r": metrics.r,
                "map50": metrics.ap50,
                "map50-95": metrics.ap,
            }
        )
        results.to_csv(f"{self.training_results_path}/{fold_name}/metrics.csv")

    def _train_model(self, dataset_yaml, model_name):
        augmentations = self._augmentations()
        model = YOLO(self.model)
        model.train(
            data=dataset_yaml,
            epochs=self.epochs,
            imgsz=self.image_size,
            rect=(self.image_size[0] != self.image_size[1]),
            device=self._get_device(),
            project=str(self.training_results_path),
            name=model_name,
            **augmentations,
        )
        # experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
        # runs_ordered_by_end_time = mlflow.search_runs([experiment.experiment_id], order_by=["end_time DESC"])
        # last_run_id = runs_ordered_by_end_time.loc[0, 'run_id']
        # model_uri = f"runs:/{last_run_id}/artifacts/weights/best.pt"
        # with mlflow.start_run(run_id=last_run_id):
        #    mlflow.log_params(self.__dict__)
        # mlflow.register_model(model_uri, "Single yolo model trained")
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

    def _create_k_folds(self, annotations, class_names, images, datasets_path):
        file_names = [annotation.stem for annotation in annotations]
        class_indices = list(range(len(class_names)))

        labels_per_image = self._build_df_with_labels_per_image(annotations, class_indices, file_names)

        # It's necessary to cast to list because a generator can only be used once, and we use it many times.
        k_fold_indices = list(KFold(n_splits=self.number_folds, shuffle=True, random_state=20).split(labels_per_image))

        fold_names = [self._fold_name(n) for n in range(self.number_folds)]
        folds_df = self._build_df_with_image_distribution_for_folds(k_fold_indices, labels_per_image, file_names,
                                                                    fold_names)
        fold_label_distribution = self._build_df_with_label_distribution_for_folds(fold_names, class_indices,
                                                                                   k_fold_indices, labels_per_image)
        folds_yamls = self._build_train_and_val_dataset_yaml(folds_df, class_names, datasets_path)
        self._build_datasets_folders(folds_df, datasets_path)
        self._copy_images_and_labels_to_datasets(annotations, datasets_path, folds_df, images)
        folds_df.to_csv(datasets_path / "kfold_datasplit.csv")
        fold_label_distribution.to_csv(datasets_path / "kfold_label_distribution.csv")
        return folds_yamls

    def _build_train_and_val_dataset_yaml(self, folds_df, class_names, datasets_path):
        folds_yamls = []
        for fold in folds_df.columns:
            fold_dir = datasets_path / fold
            fold_dir.mkdir(parents=True, exist_ok=True)
            dataset_yaml = fold_dir / f"dataset.yaml"
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
            folds_yamls.append(dataset_yaml)
        return folds_yamls

    def _build_datasets_folders(self, folds_df, datasets_path):
        for fold in folds_df.columns:
            fold_dir = datasets_path / fold
            (fold_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
            (fold_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    def _copy_images_and_labels_to_datasets(self, annotations, datasets_path, folds_df, images):
        for image, label in zip(images, annotations):
            for fold, train_or_val in folds_df.loc[image.stem].items():
                shutil.copy(image, datasets_path / fold / train_or_val / "images" / image.name)
                shutil.copy(label, datasets_path / fold / train_or_val / "labels" / label.name)

    def _create_single_dataset(self, annotations, class_names, images, datasets_path):
        folder_name = 'single_dataset'
        model_info_dir = datasets_path / folder_name
        model_info_dir.mkdir(parents=True, exist_ok=True)
        (model_info_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (model_info_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (model_info_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (model_info_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        dataset_yaml = model_info_dir / f"dataset.yaml"

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": f"{self.base_path}/{model_info_dir.as_posix()}",
                    "train": "train",
                    "val": "val",
                    "names": class_names,
                },
                ds_y,
            )
        for image, label in zip(images, annotations):
            shutil.copy(image, datasets_path / folder_name / 'train' / "images" / image.name)
            shutil.copy(label, datasets_path / folder_name / 'train' / "labels" / label.name)

        first_image = images[0]
        first_label = annotations[0]
        shutil.copy(first_image, datasets_path / folder_name / 'val' / "images" / first_image.name)
        shutil.copy(first_label, datasets_path / folder_name / 'val' / "labels" / first_label.name)
        return dataset_yaml

    def _download_dataset(self):
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.save_path.mkdir(parents=True, exist_ok=True)
        os.system(f'gsutil -m cp -r "{self.images_bucket_path}" {str(self.dataset_path)}')
        if self.obb:
            json_annotations_path = self.dataset_path / 'annotations.json'
            self._export_annotations_from_label_studio("JSON", json_annotations_path)

            print("Converting annotations to YOLO OBB format.")
            self._convert_annotations_into_yolo_obb(json_annotations_path, self.dataset_path)
        else:
            self._export_annotations_from_label_studio("YOLO", "annotations.zip")
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
        images_path = self.images_bucket_path.split("/")[-1]
        for extension in [".jpg", ".jpeg", ".png"]:
            images.extend(sorted((self.dataset_path / images_path).rglob(f"*{extension}")))
        return class_names, images, annotations

    def _export_annotations_from_label_studio(self, export_type, output_path):
        os.system(
            f"curl -X GET {self.label_studio_project_url}/export\?exportType\={export_type} -H 'Authorization: Token {self.label_studio_token}' --output '{str(output_path)}'"
        )

    def _convert_annotations_into_yolo_obb(self, json_annotations_path, output_dir_path):
        label_config_path = str(self.dataset_path / 'label_config.json')
        os.system(
            f"curl -X GET {self.label_studio_project_url}/validate -H 'Authorization: Token {self.label_studio_token}' --output '{label_config_path}'"
        )
        with open(f"{label_config_path}", "r") as f:
            raw_data = f.read()
            data_as_json = json.loads(raw_data)
            label_config = data_as_json['label_config']

        converter = Converter(config=label_config, project_dir='.')
        converter.convert_to_yolo(
            input_data=str(json_annotations_path),
            is_dir=False,
            output_dir=str(output_dir_path),
            is_obb=True)

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
        return gpu_available

    def _fold_name(self, k):
        return f"fold_{k + 1}"

    def _get_device(self):
        return '0' if self._check_if_gpu_is_available() else 'cpu'

    def _define_save_path(self):
        formatted_datetime = datetime.datetime.now().isoformat().replace('.', '').replace(':', '')
        if self.use_kfold:
            return Path(self.dataset_path / f"{formatted_datetime}_{self.number_folds}-Fold_Cross-val")
        else:
            return Path(self.dataset_path / f"{formatted_datetime}_Single_Model_Training")

    def _build_df_with_labels_per_image(self, annotations, class_indices, file_names):
        labels_per_image = pd.DataFrame([], columns=class_indices, index=file_names)
        for label in annotations:
            label_counter = Counter()
            with open(label, "r") as lf:
                lines = lf.readlines()
            for line in lines:
                label_counter[int(line.split(" ")[0])] += 1
            labels_per_image.loc[label.stem] = label_counter
        return labels_per_image.fillna(0.0)

    def _build_df_with_image_distribution_for_folds(self, k_fold_indices, labels_per_image, file_names, fold_names):
        folds_df = pd.DataFrame(index=file_names, columns=fold_names)

        for index, (train, val) in enumerate(k_fold_indices):
            folds_df[self._fold_name(index)].loc[labels_per_image.iloc[train].index] = "train"
            folds_df[self._fold_name(index)].loc[labels_per_image.iloc[val].index] = "val"
        return folds_df

    def _build_df_with_label_distribution_for_folds(self, fold_names, class_indices, k_fold_indices, labels_per_image):
        fold_label_distribution = pd.DataFrame(index=fold_names, columns=class_indices)
        for n, (train_indices, val_indices) in enumerate(k_fold_indices):
            train_totals = labels_per_image.iloc[train_indices].sum()
            val_totals = labels_per_image.iloc[val_indices].sum()
            ratio = val_totals / (train_totals + 1e-7)
            fold_label_distribution.loc[self._fold_name(n)] = ratio
        return fold_label_distribution

    def _match_images_and_annotations(self, images, annotations):
        annotations_stems = [annotation.stem for annotation in annotations]
        images_stems = [image.stem for image in images]
        images_with_annotation = [image for image in images if image.stem in annotations_stems]
        annotations_with_image = [annotation for annotation in annotations if annotation.stem in images_stems]
        return images_with_annotation, annotations_with_image

    def _clean_gpu_cache(self):
        if self._check_if_gpu_is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    training_script = TrainingScript()
    training_script.run()
