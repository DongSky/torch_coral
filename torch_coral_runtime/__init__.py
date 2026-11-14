"""In-process Coral inference, usable with the original Python 3.9 SDK."""
from .direct import DirectSession, normalize_options

__all__ = ["DirectSession", "normalize_options"]
