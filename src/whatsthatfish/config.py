"""App configuration (loaded from YAML) and the shared logger factory.

Config is parsed once into frozen dataclasses and cached as a singleton, so
every module sees the same S3/GCS/YOLO settings without re-reading the file.
"""

from dataclasses import dataclass
import yaml
from pathlib import Path
import logging
import sys


@dataclass
class S3Config:
    """iNaturalist S3 access settings: base URL, bucket, and dataset/output path maps."""

    base_url: str
    bucket: str
    datasets: dict[str, str]
    output_paths: dict[str, str]


@dataclass
class GCSConfig:
    """GCS target settings: bucket name and the named prefixes (training/validation/etc.)."""

    bucket: str
    prefixes: dict[str, str]


@dataclass
class YoloConfig:
    """YOLO dataset paths keyed by split/curriculum stage."""

    data_paths: dict[str, str]


@dataclass
class AppConfig:
    """Bundles the S3 and GCS configs that the data pipelines need."""

    s3: S3Config
    gcs: GCSConfig

    @classmethod
    def from_yaml(cls, path: str = "config/data_config.yaml") -> "AppConfig":
        """Parse the data-config YAML into an AppConfig."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            s3=S3Config(**raw["s3"]),
            gcs=GCSConfig(**raw["gcs"]),
        )


@dataclass
class ModelConfig:
    """Holds the YOLO model/dataset config loaded from yolo_config.yaml."""

    yolo: YoloConfig

    @classmethod
    def from_yaml(cls, path: str = "config/yolo_config.yaml") -> "ModelConfig":
        """Parse the yolo-config YAML into a ModelConfig."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            yolo=YoloConfig(**raw["yolo_config"]),
        )


# Singleton — loaded once, imported everywhere
_config = None
_model_config = None


def get_config(path: str | None = None) -> AppConfig:
    """Return the cached AppConfig, loading it from the default path on first call."""
    global _config
    if _config is None:
        if path is None:
            path = str(Path(__file__).parent / "config" / "data_config.yaml")
        _config = AppConfig.from_yaml(path)
    return _config


def get_model_config(path: str | None = None) -> ModelConfig:
    """Return the cached ModelConfig, loading it from the default path on first call."""
    global _model_config
    if _model_config is None:
        if path is None:
            path = str(Path(__file__).parent / "config" / "yolo_config.yaml")
        _model_config = ModelConfig.from_yaml(path)
    return _model_config


def _get_logger(name: str):
    """Return a named logger that writes to both a per-name log file and stdout.

    Handlers are attached only once per name, so repeated calls reuse the same
    configured logger rather than duplicating output.
    """
    logging_path = Path(__file__).parents[1] / "logs" / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatting = "%(asctime)s - %(levelname)s - %(message)s"

    if logger.handlers:
        return logger

    # Console (stdout) is always attached — Cloud Run / Cloud Logging captures it.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(formatting))
    logger.addHandler(console_handler)

    # File logging is best-effort: valuable locally and on the training VM, but in
    # a non-root / read-only serving container the logs dir isn't writable — fall
    # back to stdout-only rather than crashing the app at import time.
    try:
        logging_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logging_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(formatting))
        logger.addHandler(file_handler)
    except OSError:
        logger.debug(
            "File logging disabled (%s not writable) — stdout only",
            logging_path.parent,
        )

    return logger
