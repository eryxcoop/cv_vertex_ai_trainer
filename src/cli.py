import argparse

from src.training_job import TrainingJob


class CLI:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Train a Yolov8 model with Vertex AI")
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument(
            "--gc-project",
            type=str,
            required=True,
            help="The name of the Google Cloud project",
        )
        self.parser.add_argument(
            "--gc-bucket",
            type=str,
            required=True,
            help="The name of the Google Cloud bucket",
        )
        self.parser.add_argument(
            "--machine-type",
            type=str,
            required=False,
            default="n1-highmem-16",
            help="The type of machine to use. Refer to: https://cloud.google.com/vertex-ai/docs/training/configure-comp"
            "ute#machine-types",
        )
        self.parser.add_argument(
            "--accelerator",
            type=str,
            required=False,
            default="NVIDIA_TESLA_T4",
            help="The type of accelerator to use. Refer to: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/M"
            "achineSpec#acceleratortype",
        )
        self.parser.add_argument(
            "--accelerator-count",
            type=int,
            required=False,
            default=1,
            help="The number of accelerators to use.",
        )

    def run(self):
        args = self.parser.parse_args()
        train_job = TrainingJob(
            gc_project=args.gc_project,
            gc_bucket=args.gc_bucket,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator,
            accelerator_count=args.accelerator_count,
        )
        train_job.run()


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
