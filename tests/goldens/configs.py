"""Named graph configurations and one strict, cross-suite environment filter.

Each suite supplies its own defaults. An explicit list selects only the named
configurations, preserving order and ignoring known configurations belonging to
the other graph type. Validate against the whole registry so typos cannot make
a suite silently disappear. DDG None overrides deliberately exercise API defaults.
"""

from dataclasses import dataclass
import os

CONFIGS_ENV = "BINGRAPH_GOLDEN_CONFIGS"


@dataclass(frozen=True)
class GoldenConfig:
    """Existing CFG overrides; keep field defaults stable for summary baselines."""

    name: str
    cfg_mode: str
    cfg_exits: str = "jump"
    comments: bool = False
    demangle: bool = False


@dataclass(frozen=True)
class DDGConfig:
    """Only non-None fields become request overrides, never frozen defaults."""

    name: str
    memory: str | None = None
    dependencies: str | None = None


CFG_CONFIGS = [
    GoldenConfig(name="cfg_mode_none", cfg_mode="none"),
    GoldenConfig(name="cfg_mode_custom", cfg_mode="custom"),
    GoldenConfig(name="cfg_exits_always", cfg_mode="custom", cfg_exits="always"),
    GoldenConfig(name="cfg_comments_on", cfg_mode="custom", comments=True),
    GoldenConfig(
        name="cfg_demangle_on", cfg_mode="custom", comments=True, demangle=True
    ),
]
DDG_CONFIGS = [
    DDGConfig("ddg_default"),
    DDGConfig("ddg_memory_may", memory="may"),
    DDGConfig("ddg_dependencies_raw", dependencies="raw"),
]
DEFAULT_CFG_CONFIGS = [next(c for c in CFG_CONFIGS if c.name == "cfg_mode_custom")]
DEFAULT_DDG_CONFIGS = [DDG_CONFIGS[0]]


def selected_configs(configs, defaults):
    """Select this suite's configurations without activating unrelated defaults."""
    raw = os.getenv(CONFIGS_ENV, "").strip()
    if not raw:
        return list(defaults)
    names = list(dict.fromkeys(name.strip() for name in raw.split(",") if name.strip()))
    available = {config.name: config for config in CFG_CONFIGS + DDG_CONFIGS}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"{CONFIGS_ENV} contains unknown config(s): {', '.join(unknown)}. "
            f"Valid values: {', '.join(sorted(available))}"
        )
    local = {config.name: config for config in configs}
    return [local[name] for name in names if name in local]
