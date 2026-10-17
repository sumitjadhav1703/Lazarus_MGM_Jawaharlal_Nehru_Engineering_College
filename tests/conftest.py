"""Configure native runtimes before optional neural tests are collected."""

import os
import sys

if sys.platform == "darwin":
    # Mixed Torch/XGBoost OpenMP runtimes can crash; verified both import orders.
    # https://github.com/pytorch/pytorch/issues/155201
    os.environ.setdefault("OMP_NUM_THREADS", "1")
