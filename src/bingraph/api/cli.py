from __future__ import annotations
import argparse
import json
import sys


def main() -> None:
    """Print the package version or run the configured server/client command.

    Handle the standalone version query before loading application dependencies
    or validating settings: inspecting an installed version needs no binary root,
    server configuration or angr initialization. All other arguments retain the
    existing Pydantic CLI parsing and application behavior.
    """
    if sys.argv[1:] == ["--version"]:
        from bingraph import __version__

        print(f"bingraph {__version__}")
        return

    from fastapi.testclient import TestClient
    from fastapi.routing import APIRoute
    from pydantic_settings import CliApp, CliSettingsSource
    import uvicorn

    from bingraph.helpers import get_settings, Settings
    from bingraph import __version__
    from .app import create_app

    parser = argparse.ArgumentParser(description=Settings.__doc__)
    parser.add_argument(
        "--version", action="version", version=f"bingraph {__version__}"
    )
    source = CliSettingsSource(Settings, root_parser=parser)
    CliApp.run(Settings, cli_settings_source=source)
    settings = get_settings()

    if settings.server:
        # If we are debugging the server, mute uvicorn's redundant error dumps
        # so they don't clobber our clean Loguru console outputs.
        log_config = uvicorn.config.LOGGING_CONFIG
        if settings.debug:
            log_config["loggers"]["uvicorn.error"]["level"] = "CRITICAL"

        uvicorn.run(
            "bingraph.api.app:create_app" if settings.debug else create_app(),
            host=settings.server.host,
            port=settings.server.port,
            reload=settings.debug,
            factory=settings.debug,
            log_config=log_config,
        )
    else:
        client_settings = settings.client
        if client_settings is None:
            raise RuntimeError(
                "Client settings are required when server mode is disabled"
            )

        app = create_app()
        client = TestClient(app, raise_server_exceptions=settings.debug)

        # build correct endpoint path
        api_endpoint = client_settings.endpoint
        if not api_endpoint.startswith("/"):
            api_endpoint = f"/{api_endpoint}"
        if not api_endpoint.startswith("/api"):
            api_endpoint = f"/api{api_endpoint}"
        route_paths = [
            route.path for route in app.routes if isinstance(route, APIRoute)
        ]
        assert api_endpoint in route_paths, f"Invalid endpoint {api_endpoint}"

        # build query parameters
        # all client settings attributes except endpoint and payload are considered parameters
        query_params = {
            key: value
            for key, value in vars(client_settings).items()
            if value is not None and key not in ["endpoint", "payload"]
        }

        # build payload info (if needed)
        body_data = client_settings.payload
        if body_data:
            try:
                body_data = json.loads(body_data)
            except json.JSONDecodeError:
                # Fall back to raw string if it's not JSON
                pass

        # TODO: as of today, no payloads allowed, only GET endpoints considered
        # json=body_data if isinstance(body_data, dict) else None,
        # data=body_data if isinstance(body_data, str) else None
        response = client.get(api_endpoint, params=query_params)
        print(json.dumps(response.json()))


if __name__ == "__main__":
    main()
