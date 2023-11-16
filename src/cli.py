import argparse
import textwrap

import toml

from src.training_job import TrainingJob


class CLI:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="Train a Yolov8 model with Vertex AI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(self._show_config_example()),
        )
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument(
            "-c",
            "--config",
            type=str,
            required=True,
            help="The path to the config file.",
        )

    def run(self):
        args = self.parser.parse_args()
        config = self._parse_toml(args.config)
        train_job = TrainingJob(
            gc_project=config["google_cloud"]["project"],
            gc_bucket=config["google_cloud"]["bucket"],
            machine_type=config["vertex_ai_machine_config"]["machine_type"],
            accelerator_type=config["vertex_ai_machine_config"]["accelerator_type"],
            accelerator_count=config["vertex_ai_machine_config"]["accelerator_count"],
        )
        train_job.run()

    def _show_config_example(self) -> str:
        toml_file_example = "config_example.toml"
        config = self._parse_toml(toml_file_example)
        return f"Config example:\n\n{toml.dumps(config)}"

    def _parse_toml(self, file_path) -> dict:
        with open(file_path, "r") as f:
            config = toml.load(f)
        return config


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
