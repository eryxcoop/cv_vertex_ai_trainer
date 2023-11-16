from google.cloud import aiplatform


class TrainingJob:
    DEFAULT_JOB_NAME = "cv-vertex-ai-trainer"
    CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"

    def __init__(self, gc_project, gc_bucket, machine_type, accelerator_type, accelerator_count) -> None:
        self.gc_project = gc_project
        self.gc_bucket = gc_bucket
        self.machine_type = machine_type
        self.accelerator_type = accelerator_type
        self.accelerator_count = accelerator_count

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
        )

    def _load_requirements(self) -> list[str]:
        with open("remote_training/train_requirements.txt", "r") as f:
            requirements = f.read().splitlines()
        return requirements
