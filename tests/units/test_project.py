"""Fast tests for CFGFast preconditions applied by the project layer."""

from types import SimpleNamespace

from capstone import CS_GRP_CALL, CS_OP_IMM
import pytest

from bingraph.core import project as project_module


def test_cfg_cache_uses_resolved_mode(monkeypatch) -> None:
    """Default and explicit requests for the same mode share one graph."""

    settings = SimpleNamespace(cfg_mode="custom")
    monkeypatch.setattr(project_module, "get_settings", lambda: settings)
    monkeypatch.setattr(project_module, "KnowledgeBase", lambda _: object())
    calls = []

    def build(_project, _kb, _addr):
        calls.append("custom")
        return SimpleNamespace(mode="custom")

    monkeypatch.setattr(project_module, "build_custom_cfg", build)
    fast = object()
    monkeypatch.setattr(project_module, "_get_fast_cfg", lambda *_args: fast)
    project_module.get_cfg.cache_clear()
    project = object()
    try:
        custom = project_module.get_cfg(project, 0x1000)
        assert project_module.get_cfg(project, 0x1000, "custom") is custom
        settings.cfg_mode = "none"
        assert project_module.get_cfg(project, 0x1000) is fast
        assert project_module.get_cfg(project, 0x1000, "custom") is custom
        assert project_module.get_cfg(project, 0x1000, "none") is fast
        assert calls == ["custom"]
    finally:
        project_module.get_cfg.cache_clear()


def test_unsupported_cfg_mode_does_not_run_an_analysis(monkeypatch) -> None:
    """Reject invalid mode names before invoking either construction strategy."""

    monkeypatch.setattr(project_module, "KnowledgeBase", lambda _: object())
    monkeypatch.setattr(
        project_module,
        "_get_fast_cfg",
        lambda *_args: pytest.fail("Invalid mode invoked CFGFast"),
    )
    monkeypatch.setattr(
        project_module,
        "build_custom_cfg",
        lambda *_args: pytest.fail("Invalid mode invoked the custom builder"),
    )
    project_module.get_cfg.cache_clear()
    try:
        with pytest.raises(ValueError, match="Unsupported cfg mode"):
            project_module.get_cfg(object(), 0x1000, "invalid")
    finally:
        project_module.get_cfg.cache_clear()


def _terminal_call(addr: int, target: int) -> SimpleNamespace:
    """Create the Capstone instruction shape for a direct five-byte call."""

    return SimpleNamespace(
        address=addr,
        size=5,
        groups=(CS_GRP_CALL,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=target),),
    )


def _project(memory_load) -> SimpleNamespace:
    """Create the minimum loader shape needed by the terminal-call check."""

    return SimpleNamespace(
        loader=SimpleNamespace(memory=SimpleNamespace(load=memory_load))
    )


def test_terminal_unmapped_direct_call_is_predeclared_nonreturning(
    monkeypatch,
) -> None:
    """Avoid a fake return when the final direct call cannot return in-image."""

    function_addr = 0x1000
    function_size = 0x20
    callee_addr = 0x2000
    monkeypatch.setattr(
        project_module,
        "decode_raw_capstone_insns",
        lambda *_args: (_terminal_call(0x101B, callee_addr),),
    )
    project = _project(lambda _addr, _size: (_ for _ in ()).throw(KeyError()))

    assert (
        project_module._terminal_unmapped_return_callee(
            project,
            function_addr,
            function_size,
            {callee_addr},
        )
        == callee_addr
    )


def test_terminal_call_with_mapped_return_keeps_normal_inference(monkeypatch) -> None:
    """Do not override CFGFast when the function has a readable continuation."""

    function_addr = 0x1000
    function_size = 0x20
    callee_addr = 0x2000
    monkeypatch.setattr(
        project_module,
        "decode_raw_capstone_insns",
        lambda *_args: (_terminal_call(0x101B, callee_addr),),
    )
    project = _project(lambda _addr, _size: b"\x00")

    assert (
        project_module._terminal_unmapped_return_callee(
            project,
            function_addr,
            function_size,
            {callee_addr},
        )
        is None
    )


def test_terminal_call_to_unknown_target_keeps_normal_inference(monkeypatch) -> None:
    """Do not predeclare unknown direct targets as non-returning functions."""

    function_addr = 0x1000
    function_size = 0x20
    callee_addr = 0x2000
    monkeypatch.setattr(
        project_module,
        "decode_raw_capstone_insns",
        lambda *_args: (_terminal_call(0x101B, callee_addr),),
    )
    project = _project(lambda _addr, _size: (_ for _ in ()).throw(KeyError()))

    assert (
        project_module._terminal_unmapped_return_callee(
            project,
            function_addr,
            function_size,
            set(),
        )
        is None
    )
