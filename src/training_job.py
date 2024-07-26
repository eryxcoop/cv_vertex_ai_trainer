from google.cloud import aiplatform


class TrainingJob:
    DEFAULT_JOB_NAME = "cv-vertex-ai-trainer"
    CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"

    def __init__(self, gc_project, gc_bucket, machine_type, accelerator_type, accelerator_count,
                 training_config) -> None:
        self.gc_project = gc_project
        self.gc_bucket = gc_bucket
        self.machine_type = machine_type
        self.accelerator_type = accelerator_type
        self.accelerator_count = accelerator_count
        self.training_config = training_config

        aiplatform.init(project=self.gc_project, staging_bucket=self.gc_bucket)

        self.job = aiplatform.CustomTrainingJob(
            display_name=self.DEFAULT_JOB_NAME,
            container_uri=self.CONTAINER_URI,
            requirements=self._load_requirements(),
            script_path="remote_training/training_script.py",
        )

    def run(self):
        self.job.run(
            replica_count=1,
            machine_type=self.machine_type,
            accelerator_type=self.accelerator_type,
            accelerator_count=self.accelerator_count,
            environment_variables=self._load_environment_variables()
        )

    def _load_environment_variables(self):
        training_config = self.training_config
        return {
            "IMAGE_SIZE": str(training_config.image_size),
            "EPOCHS": str(training_config.epochs),
            "MODEL": str(training_config.model),
            "OBB": str(training_config.obb),
            "LABEL_STUDIO_TOKEN": str(training_config.label_studio_token),
            "LABEL_STUDIO_PROJECT_URL": str(training_config.label_studio_project_url),
            "IMAGES_BUCKET_PATH": str(training_config.images_bucket_path),
            "BUCKET_PATH": str(training_config.bucket_path),
            "NUMBER_OF_FOLDS": str(training_config.number_of_folds),
            "ACCELERATOR_COUNT": str(training_config.accelerator_count),
            "USE_KFOLD": str(training_config.use_kfold),
            "MLFLOW_TRACKING_URI": str(training_config.mlflow_tracking_uri),
            "MLFLOW_EXPERIMENT_NAME": str(training_config.mlflow_experiment_name),
            "MLFLOW_MODEL_NAME": str(training_config.mlflow_model_name),
            "MLFLOW_RUN": str(training_config.mlflow_run),
            "MLFLOW_TRACKING_USERNAME": str(training_config.mlflow_tracking_username),
            "MLFLOW_TRACKING_PASSWORD": str(training_config.mlflow_tracking_password)
        }

    def _load_requirements(self) -> list[str]:
        with open("remote_training/train_requirements.txt", "r") as f:
            requirements = f.read().splitlines()
        return requirements
