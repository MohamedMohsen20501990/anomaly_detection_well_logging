import logging
import sys

formatter = logging.Formatter("%(asctime)s :%(name)s | %(levelname)s | %(message)s", datefmt="%Y-%B-%d %H:%M:%S")

logger = logging.getLogger(name="app_log")
logger.setLevel("DEBUG")
logger.propagate=False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)