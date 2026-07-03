"""The YOLO11 fish detector as one facade: .train(), .tune(), .predict().

`Detector` configures one curriculum stage (LILA → LC1 → LC2) and exposes the
same lifecycle verbs as the classifier side. Each stage reads the weights the
previous one produced (see _WEIGHTS) and emits its own best checkpoint. It wraps
the Ultralytics `YOLO` object, which already provides train/tune/predict — this
class just pins the per-stage config, weights and (for tuning) the param space.
"""

import argparse
import logging
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
from ultralytics import YOLO
from dotenv import load_dotenv
import yaml

from .loaders.od_dataloader import CustomDetectionTrainer

load_dotenv()

logger = logging.getLogger(__name__)


def load_param_space(path: Path, dataset: str) -> dict:
    """Load a ray.tune param space from the structured YAML config.

    Supported entry types:
      uniform / loguniform / randint  — require min + max
      choice                          — requires a list under 'values'
      fixed                           — requires a scalar under 'value';
                                        passed through as a constant (not sampled)
    Raises ValueError if the dataset key is not found in the config.
    ray.tune is imported lazily so a .predict()-only import stays slim.
    """
    from ray import tune

    tune_fns = {
        "uniform": tune.uniform,
        "loguniform": tune.loguniform,
        "choice": tune.choice,
        "randint": tune.randint,
    }

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
            result[k] = tune_fns[v["type"]](float(v["min"]), float(v["max"]))
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
    EVAL = "eval"
    EXPORT = "export"


class Detector:
    """Configure and run one detector stage (`lila`, `lc1`, or `lc2`).

    Resolves the per-dataset config, weights and param space up front, so the
    lifecycle methods stay thin. The LILA → LC1 → LC2 chaining is encoded in
    _WEIGHTS (input → output checkpoint).
    """

    # Input weights and output filename per dataset
    _WEIGHTS: dict[str, tuple[str, str]] = {
        "lila": ("yolo11l.pt", "od_best.pt"),
        "lc1": ("od_best.pt", "lc1_best.pt"),
        "lc2": ("lc1_best.pt", "lc2_best.pt"),
    }

    def __init__(
        self,
        dataset: Dataset = Dataset.LILA,
        config_path: str = Path(__file__).parent.parent / "config",
        class_config: str = "class_config.yaml",
        weights_path: str = Path(__file__).parent.parent / "weights",
        param_tuning_path: str = "param_space_config.yaml",
        restore_path: str = None,
    ):
        self.config_path = Path(config_path)
        self.dataset = dataset.value if isinstance(dataset, Dataset) else dataset
        self.train_config_path = self.config_path / f"{self.dataset}_train_config.yaml"
        self.class_config = self.config_path / class_config
        self.param_path = self.config_path / param_tuning_path
        self.weights_path = Path(weights_path)
        self.restore_path = restore_path
        self.input_weights, self.output_weights = self._WEIGHTS[self.dataset]
        logger.info("Detector initialised: dataset=%s", self.dataset)

    def train_fn(self, config, epochs: int = 20, img_size: int = 640):
        """One Ray Tune trial: train YOLO with a sampled hyperparameter set.

        Capped to max_samples=8000 so trials stay short, and reports back through
        Ultralytics' metrics (the tuner reads mAP50 from these). Not used by the
        full run, which trains on the whole dataset instead.
        """
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

    def tune(self, num_samples: int = 5):
        """Run (or resume) the Ray Tune search, maximizing validation mAP50.

        Each trial gets a full GPU; failures are tolerated up to a limit so one
        bad sample doesn't kill the sweep. With a restore_path it resumes an
        existing experiment's unfinished trials instead of starting fresh.
        ray.tune is imported lazily so .predict() callers don't pull it in.
        """
        from ray import tune

        param_space = load_param_space(self.param_path, self.dataset)
        failure_config = tune.FailureConfig(max_failures=3, fail_fast=False)

        if self.restore_path and Path(self.restore_path).exists():
            tuner = tune.Tuner.restore(
                self.restore_path,
                trainable=tune.with_resources(self.train_fn, {"gpu": 1}),
                param_space=param_space,
                resume_unfinished=True,
                resume_errored=False,
                restart_errored=False,
            )
        else:
            tuner = tune.Tuner(
                tune.with_resources(self.train_fn, {"gpu": 1}),
                tune_config=tune.TuneConfig(
                    metric="metrics/mAP50(B)",
                    mode="max",
                    num_samples=num_samples,
                ),
                param_space=param_space,
                run_config=tune.RunConfig(
                    failure_config=failure_config, name=f"{self.dataset}_experiment"
                ),
            )

        results = tuner.fit()
        return results

    def train(self):
        """Full training run on the whole dataset, using the post-tuning config.

        Trains from the stage's input weights and copies the best checkpoint to
        this stage's output name (e.g. lc1_best.pt) so the next stage can pick it
        up. No sample cap here.
        """
        logger.info(
            "Starting full training run: dataset=%s weights=%s",
            self.dataset,
            self.input_weights,
        )
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

        # Post-train promotion gate: evaluate on the fixed val split, then promote
        # to the store ONLY if it clears the targets and doesn't regress vs the
        # incumbent. Local store by default; inject a GCS-backed store on the VM.
        from .promotion import (
            PromotionStore,
            gate_and_promote,
            DETECTOR_FLOOR,
            DETECTOR_KEYS,
        )

        results, _ = self.evaluate(weights=dest)
        store = PromotionStore(self.weights_path / "promoted")
        gate_and_promote(
            f"detector_{self.dataset}",
            results,
            dest,
            DETECTOR_FLOOR,
            DETECTOR_KEYS,
            store,
        )
        return results

    def recall_at_conf(self, box, conf: float = 0.15) -> float:

        r_curve = box.r_curve
        if r_curve is None or r_curve.size == 0:
            raise RuntimeError(
                "Val set produced no ground-truth labels — cannot evaluate. "
                "Is inat_obj_detection_dataset populated / annotation IS NOT NULL non-empty?"
            )
        px = box.px
        idx = int(np.argmin(np.abs(px - conf)))
        return float(r_curve.mean(0)[idx])

    def evaluate(
        self,
        weights: Path = None,
        conf: float = 0.001,
        img_size: int = 640,
        device: str = None,
    ):
        """Validate this stage's detector on its DB-driven val split.

        Reuses the CustomDetectionValidator (same [0,1]-image handling as training)
        with the val-mode dataloader, so the split matches what training validated
        on.

        We retain conf = 0.001 to measure the entire curve, while indexing the curve to
        find the recall@0.5 at conf=0.15 to record the recall at our survey mode confidence.

        Returns the DetMetrics object from the validator. Read:
          metrics.box.map50  -> mAP@0.5
          metrics.box.map    -> mAP@0.5:0.95
          metrics.box.mr     -> mean Recall@0.5 at this conf
        """
        from copy import copy
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
        from .loaders.od_dataloader import CustomDetectionValidator, od_dataloader

        w = Path(weights) if weights else self.weights_path / self.output_weights
        model = YOLO(model=w, task="detect")

        val_loader = od_dataloader(
            dataset=self.dataset, mode="val", batch_size=16, max_samples=None
        )
        # `data` is needed even though we supply the dataloader: standalone
        # validation parses it for metadata only — nc/names (init_metrics) and
        # channels (warmup). The image paths (/dev/null) are never read because
        # self.dataloader is already set, so get_dataloader() is skipped.
        overrides = {
            "conf": conf,
            "imgsz": img_size,
            "data": str(self.class_config),
        }
        # `device` is left unset by default (select_device picks the GPU if
        # present). The INT8 release gate passes device="cpu" so the quantized
        # graph is certified on the SAME execution provider it serves on — INT8
        # kernels differ across CPU vs CUDA providers.
        if device is not None:
            overrides["device"] = device
        args = get_cfg(DEFAULT_CFG, overrides=overrides)
        save_dir = self.weights_path.parent / "runs" / "detect" / f"{self.dataset}_eval"
        save_dir.mkdir(parents=True, exist_ok=True)

        validator = CustomDetectionValidator(
            dataloader=val_loader, save_dir=save_dir, args=copy(args)
        )
        validator(model=model.model)
        box = validator.metrics.box
        results = {
            "mAP@0.5": box.map50,
            "Recall@0.5": self.recall_at_conf(box),
            "mAP@0.5:0.95": box.map,
        }
        targets = {"mAP@0.5": 0.75, "mAP@0.5:0.95": 0.50, "Recall@0.5": 0.90}
        passed = {k: results[k] >= targets[k] for k in targets}
        for k in targets:
            logger.info(
                "%s = %.4f (target %.2f) %s",
                k,
                results[k],
                targets[k],
                "PASS" if passed[k] else "FAIL",
            )

        return results, passed

    def export(self, weights=None, img_size: int = 640, int8: bool = False):
        """Export the detector to ONNX (FP32, NMS baked in), optionally emitting INT8.

        Dumb by design: it only *produces* artifacts. The release gate (Recall/mAP
        accuracy via `evaluate()`, ONNX parity, INT8 degradation) lives in the model
        retraining, the .pt model is gated based on it exceeding the requisite performance metrics.
        """
        w = Path(weights) if weights else self.weights_path / self.output_weights
        model = YOLO(model=w, task="detect")
        fp32_path = Path(
            model.export(
                format="onnx",
                nms=True,
                dynamic=True,
                simplify=True,
                opset=17,
                imgsz=img_size,
            )
        )
        int8_path = None
        if int8:
            from .onnx_utils import quantize_dynamic_int8

            int8_path = quantize_dynamic_int8(fp32_path)
        return {"fp32": fp32_path, "int8": int8_path}

    def predict(self, source, weights: Path = None, conf: float = 0.25):
        """Run detection on images/video, returning Ultralytics Results.

        Loads this stage's output weights by default (e.g. lc2_best.pt) — pass
        `weights` to override. For the single best-box contract the classifier
        handoff needs, see inference.bbox_inference.BoundingBoxInference.
        """
        w = Path(weights) if weights else self.weights_path / self.output_weights
        model = YOLO(model=w, task="detect")
        return model.predict(source, conf=conf, verbose=False)


def main(**kwargs):
    """CLI entry point: dispatch to a Ray Tune search or a full training run."""
    detector = Detector(
        dataset=Dataset(kwargs["dataset"]),
        restore_path=kwargs.get("restore_path"),
    )
    if kwargs.get("type") == "tune":
        detector.tune()
    elif kwargs.get("type") == "eval":
        _, _ = detector.evaluate(weights=kwargs.get("weights_path", None))
    elif kwargs.get("type") == "export":
        detector.export(int8=kwargs.get("int8", False))
    else:
        detector.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Train or tune the YOLO object detection model",
        epilog=(
            "Examples:\n"
            "  Hyperparameter tune on LILA:  --dataset lila --type tune\n"
            "  Full training run on LC1:     --dataset lc1 --type full\n"
            "  Full training run on LC2:     --dataset lc2 --type full\n"
            "  Evaluated training run on LC1: --dataset lc1 --type eval\n"
            "  Evaluated training run on LC2: --dataset lc2 --type eval\n"
            "  Export latest model to ONNX: --dataset lc1 --type export\n"
            "  Export to ONNX + INT8 quantized variant: --dataset lc1 --type export --int8\n"
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
        help=f"Run type. Choices: {', '.join(t.value for t in TrainType)}. 'tune' runs Ray hyperparameter search; 'full' trains with config defaults; "
        f"'eval' runs evaluation @ desired conf level; 'export' exports the model to ONNX distribution",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="On export, also emit a dynamic-INT8 quantized ONNX variant.",
    )
    parser.add_argument(
        "--restore-path",
        default=None,
        metavar="PATH",
        help="Path to an existing Ray Tune experiment directory to resume (tune mode only)",
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        metavar="WEIGHTS_PATH",
        help="Path to a model weights file to load (eval mode only)",
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        type=args.type,
        restore_path=args.restore_path,
        int8=args.int8,
        weights_path=args.weights_path,
    )
