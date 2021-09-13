import pytest
from argparse import Namespace
from lab.trainer.logger import Logger
import numpy as np
from pathlib import Path


log_dir = "./tests/test_logs"
i = 1
while (Path(log_dir) / f"experiment_{i}").exists():
    i += 1
logger = Logger(Path(log_dir) / f"experiment_{i}", num_runs=10)


@pytest.mark.parametrize("logger", [(logger)])
def test_matplotlib_output(logger: Logger) -> None:
    logger.set_tensorboard_logging(False)
    logger.set_json_logging(False)
    # Add random scalars to metrics.
    for t in range(25):
        for i in range(logger.num_runs):
            logger.add("scalar", np.random.random(), t, prefix="pytest", run_id=i)

    logger.to_matplotlib(logger._metrics, name="pytest_scalar")


@pytest.mark.parametrize("logger", [(logger)])
def test_tensorboard_output(logger: Logger) -> None:
    logger.set_tensorboard_logging(True)
    logger.set_json_logging(False)
    # Test scalar logging.
    for t in range(10):
        for i in range(logger.num_runs):
            logger.add("scalar", np.random.random(), t, prefix="pytest", run_id=i)

    # Test image logging.
    for t in range(5):
        for i in range(logger.num_runs):
            rgb_image = np.zeros((3, 64, 64))
            rgb_image[0] = (np.arange(0, 4096).reshape(64, 64) / 4096) * (1 / (1 + t))
            rgb_image[1] = 1 - np.arange(0, 4096).reshape(64, 64) / 4096
            bw_image = np.random.rand(64, 64)
            logger.add("bw_image", bw_image, t, prefix="pytest", run_id=i)
            logger.add("rbg_image", rgb_image, t, prefix="pytest", run_id=i)

    # Test video logging.
    pass


@pytest.mark.parametrize("logger", [(logger)])
def test_json_output(logger: Logger) -> None:
    logger.set_tensorboard_logging(False)
    logger.set_json_logging(True)
    # Test scalar logging (should appear in json).
    for t in range(10):
        for i in range(logger.num_runs):
            logger.add("scalar", np.random.random(), t, prefix="pytest", run_id=i)

    # Test image logging (should not influence json logging).
    for t in range(5):
        for i in range(logger.num_runs):
            rgb_image = np.zeros((3, 64, 64))
            rgb_image[0] = (np.arange(0, 4096).reshape(64, 64) / 4096) * (1 / (1 + t))
            rgb_image[1] = 1 - np.arange(0, 4096).reshape(64, 64) / 4096
            bw_image = np.random.rand(64, 64)
            logger.add("bw_image", bw_image, t, prefix="pytest", run_id=i)
            logger.add("rbg_image", rgb_image, t, prefix="pytest", run_id=i)


if __name__ == "__main__":
    test_matplotlib_output(logger)
