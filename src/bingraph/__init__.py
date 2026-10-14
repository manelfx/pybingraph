"""Binary inspection with a shared package, CLI and build version."""

from .version import get_version_info

__version__, __commit__ = get_version_info()

__all__ = ["__version__", "__commit__"]
