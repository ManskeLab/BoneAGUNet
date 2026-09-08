"""BoneAGUNet public API."""

__all__ = ["run"]
__version__ = "0.1.0"


def run(*args, **kwargs):
    """Load the imaging stack lazily, keeping config/install commands lightweight."""
    from .pipeline import run as _run
    return _run(*args, **kwargs)
