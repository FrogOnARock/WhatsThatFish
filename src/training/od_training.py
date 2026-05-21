import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO
from ..models.od_dataloader import CustomDetectionTrainer
from dotenv import load_dotenv
from ray import tune


CONFIG_PATH = str(Path(__file__).parent.parent / "config" / "class_config.yaml")
TRAIN_CONFIG_PATH = str(Path(__file__).parent.parent / "config" / "train_config.yaml")
WEIGHTS_PATH = str(Path(__file__).parent.parent.parent / "yolo11l.pt")
RESTORE_PATH = "/home/frogonarock/ray_results/train_fn_2026-05-17_19-52-35"

load_dotenv()
def train_fn(config):
    CustomDetectionTrainer.max_samples = 8000
    model = YOLO(model=WEIGHTS_PATH)
    model.train(
        cfg=TRAIN_CONFIG_PATH,
        data=CONFIG_PATH,
        trainer=CustomDetectionTrainer,
        epochs=20,
        imgsz=640,
        lr0=config["lr0"],
        box=config["box"],
        cls=config["cls"],
        weight_decay=config["weight_decay"],
        dfl=config["dfl"],
        verbose=False,
    )


PARAM_SPACE = {
    "lr0": tune.loguniform(5e-3, 5e-2),
    "box": tune.uniform(5.0, 9.0),
    "cls": tune.uniform(0.8, 1.5),
    "weight_decay": tune.loguniform(1e-3, 1e-2),
    "dfl": tune.loguniform(0.5, 2.0),
}


def tune_model():
    if RESTORE_PATH and Path(RESTORE_PATH).exists():
        tuner = tune.Tuner.restore(
            RESTORE_PATH,
            trainable=tune.with_resources(train_fn, {"gpu": 1}),
            param_space=PARAM_SPACE,
            resume_unfinished=True,
            resume_errored=False,
            restart_errored=False
        )
    else:
        tuner = tune.Tuner(
            tune.with_resources(train_fn, {"gpu": 1}),
            tune_config=tune.TuneConfig(
                metric="metrics/mAP50(B)",
                mode="max",
                num_samples=8,
            ),
            param_space=PARAM_SPACE,
        )

    results = tuner.fit()
    return results


def train_final():
    import shutil
    CustomDetectionTrainer.max_samples = None
    model = YOLO(model=WEIGHTS_PATH)
    model.train(
        cfg=TRAIN_CONFIG_PATH,
        data=CONFIG_PATH,
        trainer=CustomDetectionTrainer,
        verbose=True,
    )
    best_pt = Path(model.trainer.best)
    dest = Path(__file__).parent.parent.parent / "weights" / "od_best.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best_pt, dest)
    print(f"Saved best weights → {dest}")


if __name__ == '__main__':
    # parameter_results = tune_model()
    # best_config = parameter_results.get_best_result(metric="metrics/mAP50(B)", mode="max").config
    # print("Best config:", best_config)
    train_final()





