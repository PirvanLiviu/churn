import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
MODELS = ROOT / "models"

RAW_CSV = DATA / "raw.csv"
PREPROCESSED_CSV = DATA / "preprocessed.csv"
MODEL_FILE = MODELS / "model_v1.ubj"

# src/ for the training modules, the project root for the api package
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@contextmanager
def cwd(path):
    # the src modules use paths like "../data/...", so they only work when run from inside src/
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def import_fresh(name, work_dir=SRC):
    # modules in src do their work at import time, so re-import them from the right directory
    sys.modules.pop(name, None)
    with cwd(work_dir):
        return importlib.import_module(name)
