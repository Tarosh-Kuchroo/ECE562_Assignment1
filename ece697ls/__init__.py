"""Compatibility shim package `ece697ls` that re-exports code from `ece662`.

Some notebooks and student code import from `ece697ls` while the actual
implementation lives in the `ece662` directory in this repository. This
module makes `import ece697ls...` work by delegating to `ece662`.
"""
# Expose the same public API as ece662 by manipulating the import system.
from importlib import import_module
import sys

# Ensure the package is importable as a module
__all__ = []

def _reexport(subpkg):
    """Import a subpackage from ece662 and copy its public names into this
    package's namespace. This keeps the original modules intact while
    allowing imports like `from ece697ls.classifiers.fc_net import *` to work.
    """
    mod = import_module(f"ece662.{subpkg}")
    # Re-export public names
    for name in getattr(mod, "__all__", [n for n in dir(mod) if not n.startswith("_")]):
        globals()[name] = getattr(mod, name)
        __all__.append(name)

# Lazy re-exports for common subpackages. Individual wrapper modules will
# import from ece662 directly when needed; this file keeps the package
# present so `import ece697ls` succeeds.
