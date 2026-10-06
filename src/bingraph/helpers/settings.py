from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast
from loguru import logger
from pydantic import Field, field_validator, BaseModel
from pydantic_settings import (
    BaseSettings,
    CliImplicitFlag,
    SettingsConfigDict,
    CliSubCommand,
    CliApp,
)


# CFG mode
CfgMode = Literal["none", "custom", "extract"]
CfgExits = Literal["never", "jump", "always"]


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
        "custom", description="CFG reconstruction mode and fallback strategy"
    )
    cfg_exits: CfgExits = Field(
        "jump", description="Which CFG exits to render outside the function"
    )
    comments: CliImplicitFlag[bool] = Field(
        True, description="Appends comments to instructions when available"
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
    format: str = Field("dot", description="Output format for graphs")

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
