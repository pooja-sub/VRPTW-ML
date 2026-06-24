import importlib.util
import os


def _load_expanded_pricing_module(module_filename, module_name):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    expanded_dir = os.path.join(root_dir, "BranchingExpandedPricing")
    module_path = os.path.join(expanded_dir, module_filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_expanded_module = _load_expanded_pricing_module("expanded_pricing.py", "branching_expanded_pricing")
ExpandedGraphPricing = _expanded_module.ExpandedGraphPricing
