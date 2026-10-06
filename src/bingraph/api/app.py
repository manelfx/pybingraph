from __future__ import annotations
from http import HTTPStatus
from pathlib import Path
from typing import Callable, Generic, TypeVar
import sys

from angr import Project
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.routing import APIRoute
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException
from loguru import logger

from bingraph.helpers import (
    CfgExits,
    CfgMode,
    get_settings,
    resolve_under_root,
    MODULE_NAME,
)
from bingraph.core import (
    load_project,
    list_function_symbols,
    render_cfg,
    FunctionSymbol,
)


ItemT = TypeVar("ItemT")


class ItemsList(BaseModel, Generic[ItemT]):
    """Generic API wrapper for responses that expose a list of items."""

    items: list[ItemT]


def create_app() -> FastAPI:

    settings = get_settings()

    class RouteInterceptor(APIRoute):
        """
        Custom routing gatekeeper that captures exceptions directly at the execution
        boundary, bypassing Starlette's noisy middleware streaming panics.
        """

        def get_route_handler(self) -> Callable:
            original_handler = super().get_route_handler()

            async def custom_route_handler(request: Request) -> Response:
                try:
                    return await original_handler(request)
                except Exception as exc:
                    # CLI Client Debug Mode Branch
                    if settings.debug and settings.client:
                        # Let it leak naturally out to cli.py's TestClient
                        # The thing is that TestClient works different than alive server,
                        # that's why in this case we don't need to log stack trace.
                        raise exc

                    # normalize response status codes
                    if isinstance(exc, (ValueError, FileNotFoundError)):
                        status_code = HTTPStatus.BAD_REQUEST
                    elif isinstance(exc, StarletteHTTPException):
                        status_code = exc.status_code
                    else:
                        status_code = HTTPStatus.INTERNAL_SERVER_ERROR

                    if settings.debug:
                        # here, this is server mode for sure.
                        # log the exception beautifully with Loguru exactly ONCE
                        logger.exception(
                            f"Error [{status_code}] across {request.url.path}: {str(exc)}"
                        )

                        # In debug server mode, raise the error *after* logging with Loguru
                        # so the interactive yellow webpage can still load in the browser.
                        raise exc

                    # from here, we are in non-debug mode
                    logger.error(
                        f"Error [{status_code}] across {request.url.path}: {str(exc)}"
                    )

                    # format UI routes vs API paths smoothly
                    if request.url.path.startswith("/api"):
                        return JSONResponse(
                            status_code=status_code, content={"error": str(exc)}
                        )

                    # show error message in browser.
                    # note this implies a non-/api endpoint, since client mode doesn't allow to query those.
                    return HTMLResponse(
                        status_code=status_code,
                        content=f"<html><body><h1>Error {status_code}</h1><p>{str(exc)}</p></body></html>",
                    )

            return custom_route_handler

    # Enable debug mode on the FastAPI app context if set via CLI
    app = FastAPI(
        title="bingraph",
        debug=settings.server is not None and settings.debug,
    )

    # Force the app router to process all endpoints through our interceptor class
    app.router.route_class = RouteInterceptor

    # Setting templates directory
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    # logging setup
    logger.remove()
    logger.add(sys.stderr, filter=MODULE_NAME, level=settings.log_level)
    logger.info(f"Server initialized with settings {settings.model_dump()}")

    # Add the global crash hook inside the app factory
    def execution_crash_hook(exctype, value, traceback):
        if issubclass(exctype, KeyboardInterrupt):
            sys.__excepthook__(exctype, value, traceback)
            return
        # Render the raw uncaught panic through Loguru
        logger.opt(exception=(exctype, value, traceback)).critical(
            "Unhandled application panic encountered:"
        )

    sys.excepthook = execution_crash_hook

    # SYMTAB endpoints #########################################################

    def _get_project(filepath: str) -> Project:

        resolved = resolve_under_root(settings.root, filepath)
        return load_project(resolved)

    def _get_symbols(filepath: str) -> list[FunctionSymbol]:

        project = _get_project(filepath)
        return list_function_symbols(project)

    @app.get("/symtab", response_class=HTMLResponse)
    def symtab(request: Request, filepath: str = Query(...)) -> HTMLResponse:

        return templates.TemplateResponse(
            request,
            "symtab.html",
            {"symbols": _get_symbols(filepath), "filepath": filepath},
        )

    @app.get("/api/symtab", response_model=ItemsList[FunctionSymbol])
    def api_symtab(
        request: Request, filepath: str = Query(...)
    ) -> ItemsList[FunctionSymbol]:

        return ItemsList(items=_get_symbols(filepath))

    # Control flow graph endpoints #############################################

    def _resolve_faddr(faddr: str) -> int:
        """Make sure function addr is valid."""

        try:
            return int(faddr, 16) if faddr.startswith("0x") else int(faddr)
        except ValueError as exc:
            raise ValueError("Invalid function address") from exc

    def _render_cfg(
        filepath: str,
        function: str,
        dfs: bool | None = None,
        comments: bool | None = None,
        mode: CfgMode | None = None,
        exits: CfgExits | None = None,
        format: str = "svg",
    ) -> str:
        func_addr = _resolve_faddr(function)
        project = _get_project(filepath)
        dfs = settings.dfs_rank if dfs is None else dfs
        comments = settings.comments if comments is None else comments
        cfg_mode = mode or settings.cfg_mode
        cfg_exits = exits or settings.cfg_exits
        return render_cfg(
            project,
            func_addr,
            dfs,
            comments,
            cfg_mode,
            cfg_exits,
            format,
        )

    @app.get("/cfg")
    def cfg(
        request: Request,
        filepath: str = Query(...),
        function: str = Query(...),
        dfs: bool | None = Query(None),
        comments: bool | None = Query(None),
        mode: CfgMode | None = Query(None),
        exits: CfgExits | None = Query(None),
    ) -> Response:
        """Endpoint to return the CFG of a specified function as an SVG image."""

        svg = _render_cfg(
            filepath,
            function,
            dfs,
            comments,
            mode,
            exits,
            format="svg",
        )
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/api/cfg", response_model=dict[str, str])
    def api_cfg(
        request: Request,
        filepath: str = Query(...),
        function: str = Query(...),
        format: str = Query(...),
        dfs: bool | None = Query(None),
        comments: bool | None = Query(None),
        mode: CfgMode | None = Query(None),
        exits: CfgExits | None = Query(None),
    ) -> dict[str, str]:
        """Endpoint to return the CFG of a specified function."""

        cfg = _render_cfg(filepath, function, dfs, comments, mode, exits, format)
        return {"graph": cfg}

    return app
