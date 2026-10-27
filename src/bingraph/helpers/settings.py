from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, cast
from loguru import logger
from pydantic import AfterValidator, Field, field_validator, BaseModel
from pydantic_settings import (
    BaseSettings,
    CliImplicitFlag,
    SettingsConfigDict,
    CliSubCommand,
    CliApp,
)


# CFG mode
CfgMode = Literal["none", "custom"]
CfgExits = Literal["never", "jump", "always"]
MemoryPolicy = Literal["always", "may", "must"]


def normalize_dependencies(value: str) -> str:
    """Validate a nonempty selection and canonicalize case, order and duplicates.

    API, CLI and Python rendering share the same rule, so an invalid filter
    cannot silently hide dependencies or fall back to a different selection.
    """
    kinds = ("raw", "war", "waw")
    selected = {part.strip().lower() for part in value.split(",")}
    if not selected <= set(kinds):
        raise ValueError("Dependencies must be comma-separated raw, war, or waw")
    return ",".join(kind for kind in kinds if kind in selected)


DependencySelection = Annotated[str, AfterValidator(normalize_dependencies)]


class GlobalSettings(BaseSettings):
    """Configuration settings for the application.

    Settings model allows a .bingraphenv config file.
    Also, CLI arguments are implicitly parsed, as well as env variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="BINGRAPH_",
        env_file=".bingraphenv",
        env_nested_delimiter="__",  # for parsing from env (e.g., BINGRAPH_SERVER__HOST)
        cli_parse_args=True,  # automatic parsing of CLI options
        cli_implicit_flags=True,  # allows using "--no-x" bool flag modes
        cli_kebab_case=True,  # CLI options should be shown as kebab-case
    )

    root: Path = Field(..., description="The root directory for binaries")
    cfg_mode: CfgMode = Field(
        "custom", description="Bounded custom construction or unmodified CFGFast"
    )
    cfg_exits: CfgExits = Field(
        "jump", description="Which CFG exits to render outside the function"
    )
    ddg_memory: MemoryPolicy = Field(
        "must",
        description="DDG memory edges: always, may alias, or must overlap",
    )
    ddg_dependencies: DependencySelection = Field(
        "raw,war,waw", description="DDG edges to display: comma-separated raw, war, waw"
    )
    comments: CliImplicitFlag[bool] = Field(
        True, description="Appends comments to instructions when available"
    )
    demangle: CliImplicitFlag[bool] = Field(
        False, description="Demangles displayed symbol names without changing analysis"
    )
    dfs_rank: CliImplicitFlag[bool] = Field(
        False, description="Improves CFG layout on loop-back"
    )

    log_level: str = Field(
        "INFO", description="Flag to define level for logging messages"
    )
    debug: CliImplicitFlag[bool] = Field(False, description="Flag to enable debug mode")

    @field_validator("root", mode="before")
    def validate_root(cls, v):
        """Makes sure provided root directory exists."""

        if not Path(v).exists():
            raise ValueError(f"root directory {v} does not exist")
        return Path(v)

    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Makes sure log level is valid value."""

        upper_v = v.upper()
        try:
            logger.level(upper_v)
        except ValueError as exc:
            raise ValueError("Must be a valid Loguru level") from exc
        return upper_v


class ServerSettings(BaseModel):
    """Server mode settings."""

    host: str = Field("127.0.0.1", description="The host IP address for the server")
    port: int = Field(8000, description="The port number to run the server")

    def cli_cmd(self) -> None:
        pass


class ClientSettings(BaseModel):
    """Server mode settings."""

    endpoint: str = Field(..., description="Target API endpoint")
    payload: str | None = Field(None, description="JSON Payload data")

    filepath: str = Field(
        ...,
        description="Path to target binary (relative to root directory for binaries)",
    )
    function: str | None = Field(
        None, description="Function address (in int or 0x hex format)"
    )
    address: str | None = Field(None, description="DDG block start address")
    size: int | None = Field(None, description="DDG block size in bytes")
    memory: MemoryPolicy | None = Field(
        None, description="Override the DDG memory policy"
    )
    dependencies: DependencySelection | None = Field(
        None, description="Override DDG edges: comma-separated raw, war, waw"
    )
    format: str = Field("dot", description="Textual graph output: svg, dot, raw, xdot")

    def cli_cmd(self) -> None:
        pass


_settings: Settings | None = None


class Settings(GlobalSettings):
    """Root settings model."""

    server: CliSubCommand[ServerSettings]
    client: CliSubCommand[ClientSettings]

    def cli_cmd(self) -> None:
        """
        Pydantic executes this method immediately after successfully parsing the CLI.
        We intercept execution here to save the instance to our global singleton.
        """
        global _settings
        _settings = self  # Save this specific instantiated runtime object

        # Hand off execution to the typed subcommand
        CliApp.run_subcommand(self)


def get_settings() -> Settings:
    """Returns a cached, globally shared Settings instance,
    but defers CLI parsing until explicitly called.
    """
    global _settings
    if _settings is None:
        # Pydantic Settings consumes this private constructor option at runtime,
        # but its generated type signature does not expose it to static checkers.
        _settings = cast(Any, Settings)(_cli_parse_args=True)
    return _settings
