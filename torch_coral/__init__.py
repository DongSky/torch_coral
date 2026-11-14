"""Modern PyTorch front end for the legacy Coral Edge TPU inference hardware."""
__version__ = '0.3.0a9'


def __getattr__(name):
    if name in __all__:
        import sys
        if sys.version_info < (3, 12):
            raise RuntimeError('The PyTorch frontend requires Python 3.12+. Use torch_coral_runtime.DirectSession in the legacy SDK environment.')
    if name == 'compile':
        from .compiler import compile
        return compile
    if name in ('CoralModule', 'load'):
        from . import runtime
        return getattr(runtime, name)
    if name == 'make_backend':
        from .backend import make_backend
        return make_backend
    raise AttributeError(name)

__all__ = ["compile", "load", "CoralModule", "make_backend"]
