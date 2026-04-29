import importlib.util
import os


def _load_branching_module(module_filename, module_name):
    branching_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Branching"))
    module_path = os.path.join(branching_dir, module_filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_branching_spprc = _load_branching_module("SPPRC.py", "branching_SPPRC")
SPPRC = _branching_spprc.SPPRC
