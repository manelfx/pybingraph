from __future__ import annotations
from functools import lru_cache, wraps
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable

from loguru import logger
import cxxfilt
from angr.rust.utils.demangler import demangle as rust_demangle


MODULE_NAME = "bingraph"


def time_it(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)  # Preserves the original function's name and docstring
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()  # Highest resolution clock available

        result = func(*args, **kwargs)  # Execute the actual function

        end_time = perf_counter()
        execution_time = end_time - start_time

        # Log the result
        function_name = getattr(func, "__name__", type(func).__name__)
        logger.info(
            f"Function '{function_name}' executed in {execution_time:.4f} seconds"
        )
        return result

    return wrapper


def resolve_under_root(root: Path, relpath: str) -> Path:
    """
    Resolve a file's relative path under the given root directory.

    This function ensures that the provided relative path does not start with a slash,
    resolves the absolute path under the provided root, and verifies that the resolved path
    does not escape the root directory. It raises:
      - ValueError if the path is absolute or escapes the root directory.
      - FileNotFoundError if the resolved path does not exist.

    Args:
        root (Path): The root directory.
        relpath (str): The file path relative to the root.

    Returns:
        Path: The resolved absolute file path under the root.
    """
    if relpath.startswith("/"):
        raise ValueError("filepath must be relative to root")
    candidate = (root / relpath).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise ValueError("filepath escapes root directory")
    if not candidate.exists():
        raise FileNotFoundError(f"Binary not found: {candidate}")
    return candidate


@lru_cache(maxsize=512)
def demangle(name: str) -> str:
    """Decode C++ and recognizable Rust symbols, retaining unsupported names.

    Reuse angr's Rust support for legacy escapes and v0 symbols. Legacy Rust
    shares C++'s prefix, so require its hash suffix before selecting that parser.
    Preserve ELF symbol versions independently of the mangled base name.
    """
    if not name:
        return ""
    base, separator, version = name.partition("@")
    suffix = separator + version
    legacy = re.fullmatch(r"((?:_ZN|__ZN|ZN).*17h[0-9a-fA-F]{16}E)(\..*)?", base)
    if base.startswith(("_R", "__R")) or (legacy is not None):
        rust_name = legacy[1] if legacy else base
        clone = (legacy[2] or "") if legacy else ""
        try:
            result = rust_demangle(rust_name)
        except (KeyError, OverflowError):
            # Malformed escape sequences must not break a display-only option.
            result = rust_name
        if result != rust_name:
            return result + clone + suffix
    try:
        return cxxfilt.demangle(base) + suffix
    except cxxfilt.InvalidName:
        return name
