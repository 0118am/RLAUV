"""Lazy imports for notebook-only numerical and plotting dependencies."""

import importlib


class LazyDependency:
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self._module = None

    def __getattr__(self, name: str):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)


np = LazyDependency("numpy")
pd = LazyDependency("pandas")
