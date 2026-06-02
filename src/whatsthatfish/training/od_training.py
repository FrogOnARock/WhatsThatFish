import argparse
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from ultralytics import YOLO
from dotenv import load_dotenv
from ray import tune
import yaml
from enum import Enum
from dataclasses import dataclass
import shutil

from ..models.loaders.od_dataloader import CustomDetectionTrainer
load_dotenv()

logger = logging.getLogger(__name__)

# Maps YAML type strings to ray.tune distribution constructors.
# Add entries here when new distribution types are needed in param_space_config.yaml.
_TUNE_FNS = {
    "uniform": tune.uniform,
    "loguniform": tune.loguniform,
    "choice": tune.choice,
    "randint": tune.randint,
}


def load_param_space(path: Path, dataset: str) -> dict:
    """Load a ray.tune param space from the structured YAML config.

    Supported entry types:
      uniform / loguniform / randint  — require min + max
      choice                          — requires a list under 'values'
      fixed                           — requires a scalar under 'value';
                                        passed through as a constant (not sampled)
    Raises ValueError if the dataset key is not found in the config.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    space = raw.get(dataset)
    if space is None:
        raise ValueError(f"No param space defined for dataset '{dataset}' in {path}")

    result = {}
    for k, v in space.items():
        if v["type"] == "fixed":
            result[k] = v["value"]
        elif v["type"] == "choice":
            result[k] = tune.choice(v["values"])
        else:
            result[k] = _TUNE_FNS[v["type"]](float(v["min"]), float(v["max"]))
    return result


@dataclass
class Dataset(str, Enum):
    LILA = "lila"
    LC1 = "lc1"
    LC2 = "lc2"

@dataclass
class TrainType(str, Enum):
    FULL = "full"
    TUNE = "tune"

class ObjectDetectionTrain:

    # Input weights and output filename per dataset
    _WEIGHTS: dict[str, tuple[str, str]] = {
        "lila": ("yolo11l.pt",   "od_best.pt"),
        "lc1":  ("od_best.pt",   "lc1_best.pt"),
        "lc2":  ("lc1_best.pt",  "lc2_best.pt"),
    }

    def __init__(self,
                 dataset: Dataset,
                 train_type: TrainType,
                 config_path: str = Path(__file__).parent.parent / "config",
                 class_config: str = "class_config.yaml",
                 weights_path: str = Path(__file__).parent.parent / "weights",
                 param_tuning_path: str = "param_space_config.yaml",
                 restore_path: str = None):

        self.config_path = Path(config_path)
        self.dataset = dataset.value
        self.train_type = train_type.value
        self.train_config_path = self.config_path / f"{self.dataset}_train_config.yaml"
        self.class_config = self.config_path / class_config
        self.param_path = self.config_path / param_tuning_path
        self.weights_path = Path(weights_path)
        self.restore_path = restore_path
        self.input_weights, self.output_weights = self._WEIGHTS[self.dataset]
        self.tune_param_space = load_param_space(self.param_path, self.dataset)
        logger.info("ObjectDetectionTrain initialised: dataset=%s train_type=%s", self.dataset, self.train_type)

    def train_fn(self, config, epochs: int = 20, img_size: int = 640):
        logger.info("tune trial starting: %s", config)
        CustomDetectionTrainer.max_samples = 8000
        CustomDetectionTrainer.dataset = self.dataset
        model = YOLO(model=self.weights_path / self.input_weights)
        model.train(
            cfg=str(self.train_config_path),
            data=self.class_config,
            trainer=CustomDetectionTrainer,
            epochs=epochs,
            imgsz=img_size,
            lr0=config["lr0"],
            box=config["box"],
            cls=config["cls"],
            weight_decay=config["weight_decay"],
            dfl=config["dfl"],
            verbose=False,
        )


    def tune_model(self, num_samples: int = 5):

        failure_config = tune.FailureConfig(
            max_failures=3,
            fail_fast=False
        )

        if self.restore_path and Path(self.restore_path).exists():
            tuner = tune.Tuner.restore(
                self.restore_path,
                trainable=tune.with_resources(self.train_fn, {"gpu": 1}),
                param_space=self.tune_param_space,
                resume_unfinished=True,
                resume_errored=False,
                restart_errored=False
            )
        else:
            tuner = tune.Tuner(
                tune.with_resources(self.train_fn, {"gpu": 1}),
                tune_config=tune.TuneConfig(
                    metric="metrics/mAP50(B)",
                    mode="max",
                    num_samples=num_samples,
                ),
                param_space=self.tune_param_space,
                run_config=tune.RunConfig(
                    failure_config=failure_config,
                    name=f"{self.train_type}_experiment"
                )
            )

        results = tuner.fit()
        return results


    def train_final(self):
        logger.info("Starting full training run: dataset=%s weights=%s", self.dataset, self.input_weights)
        CustomDetectionTrainer.max_samples = None
        CustomDetectionTrainer.dataset = self.dataset
        model = YOLO(model=self.weights_path / self.input_weights)
        model.train(
            cfg=str(self.train_config_path),
            data=self.class_config,
            trainer=CustomDetectionTrainer,
            verbose=True,
        )
        best_pt = Path(model.trainer.best)
        dest = self.weights_path / self.output_weights
        dest.parent.mkdir(exist_ok=True)
        shutil.copy(best_pt, dest)
        logger.info("Saved best weights → %s", dest)

    def run(self):
        if self.train_type == "tune":
            self.tune_model()
        elif self.train_type == "full":
            self.train_final()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Train or tune the YOLO object detection model",
        epilog=(
            "Examples:\n"
            "  Hyperparameter tune on LILA:  --dataset lila --type tune\n"
            "  Full training run on LC1:     --dataset lc1 --type full\n"
            "  Full training run on LC2:     --dataset lc2 --type full\n"
            "  Resume a tune from a Ray checkpoint:\n"
            "    --dataset lila --type tune --restore-path /path/to/ray/results\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=[d.value for d in Dataset],
        required=True,
        metavar="DATASET",
        help=f"Dataset to train on. Choices: {', '.join(d.value for d in Dataset)}",
    )
    parser.add_argument(
        "--type",
        choices=[t.value for t in TrainType],
        required=True,
        metavar="TYPE",
        help=f"Run type. Choices: {', '.join(t.value for t in TrainType)}. 'tune' runs Ray hyperparameter search; 'full' trains with config defaults",
    )
    parser.add_argument(
        "--restore-path",
        default=None,
        metavar="PATH",
        help="Path to an existing Ray Tune experiment directory to resume (tune mode only)",
    )
    args = parser.parse_args()

    od_train = ObjectDetectionTrain(
        dataset=Dataset(args.dataset),
        train_type=TrainType(args.type),
        restore_path=args.restore_path,
    )
    od_train.run()





