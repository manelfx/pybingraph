"""Stateless VEX vocabulary and local definitions shared by CFG consumers.

These helpers do not lift blocks, follow predecessor edges or infer targets.
Callers retain their own memory, ABI, work-budget and presentation policies.
"""

from types import MappingProxyType
from typing import Any

import pyvex


# Supported scalar reads, not a bound on vector loads or pointer widths.
STATIC_INTEGER_BYTES = frozenset({1, 2, 4, 8})
X86_CONDITION_HELPERS = MappingProxyType(
    {
        "AMD64": "amd64g_calculate_condition",
        "X86": "x86g_calculate_condition",
    }
)
X86_CONDITION_HELPER_NAMES = frozenset(X86_CONDITION_HELPERS.values())


def temporary_definitions(vex: Any) -> dict[int, Any]:
    """Return local WrTmp expressions without resolving or evaluating them.

    This intentionally excludes LoadG destinations. Analyses requiring read
    positions or provenance must keep their richer maps, not substitute this
    context-free view for reaching-definition facts.
    """

    return {
        stmt.tmp: stmt.data
        for stmt in vex.statements
        if isinstance(stmt, pyvex.stmt.WrTmp)
    }


def is_call_like_jumpkind(jumpkind: object) -> bool:
    """Identify VEX calls/system transfers, not fake returns or internal labels."""

    return jumpkind == "Ijk_Call" or (
        isinstance(jumpkind, str) and jumpkind.startswith("Ijk_Sys_")
    )
