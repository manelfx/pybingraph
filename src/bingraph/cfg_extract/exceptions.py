"""Recover proven ELF Itanium/DWARF exceptional call destinations.

The normal extractor deliberately does not infer exception flow from opaque
cleanup code. ELF binaries with an LSDA provide a stronger source of truth:
each call-site record names the instruction range that can unwind and its
landing pad. This module reads only that narrowly scoped metadata and returns
no result whenever its encoding cannot be interpreted exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from angr import Project
from elftools.common.exceptions import ELFError
from elftools.dwarf.callframe import FDE
from elftools.elf.elffile import ELFFile

from bingraph.cfg.models import FunctionBounds


_DW_EH_PE_OMIT = 0xFF
_DW_EH_PE_ULEB128 = 0x01
_DW_EH_PE_UDATA2 = 0x02
_DW_EH_PE_UDATA4 = 0x03
_DW_EH_PE_UDATA8 = 0x04


@dataclass(frozen=True)
class ExceptionalCallSite:
    """One LSDA-proven call instruction range and its landing pad."""

    start_addr: int
    end_addr: int
    landing_pad_addr: int


@dataclass(frozen=True)
class _FunctionExceptionSites:
    """Link-time exceptional transfers belonging to one FDE range."""

    start_addr: int
    end_addr: int
    call_sites: tuple[ExceptionalCallSite, ...]


class _UnsupportedLSDA(ValueError):
    """Raised internally when a record cannot be decoded without guessing."""


def _read_uleb128(data: bytes, offset: int, end: int) -> tuple[int, int]:
    """Read one bounded unsigned LEB128 value."""

    value = 0
    shift = 0
    while offset < end:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            break
    raise _UnsupportedLSDA("truncated or oversized ULEB128")


def _read_call_site_value(
    data: bytes,
    offset: int,
    end: int,
    encoding: int,
    *,
    byteorder: Literal["little", "big"],
) -> tuple[int, int]:
    """Read one LSDA call-site-table value with no relative interpretation."""

    if encoding == _DW_EH_PE_ULEB128:
        return _read_uleb128(data, offset, end)

    widths = {
        _DW_EH_PE_UDATA2: 2,
        _DW_EH_PE_UDATA4: 4,
        _DW_EH_PE_UDATA8: 8,
    }
    width = widths.get(encoding)
    if width is None or offset + width > end:
        raise _UnsupportedLSDA(f"unsupported call-site encoding {encoding:#x}")
    return int.from_bytes(data[offset : offset + width], byteorder), offset + width


def _parse_lsda_call_sites(
    data: bytes,
    offset: int,
    *,
    function_addr: int,
    byteorder: Literal["little", "big"],
) -> tuple[ExceptionalCallSite, ...]:
    """Parse the standard LSDA call-site table at ``offset``.

    The first implementation accepts the common Itanium layout whose landing
    pad base is omitted, making FDE ``initial_location`` the base. It rejects
    other pointer encodings rather than treating a guessed address as code.
    Type/action records are intentionally not interpreted: a non-zero landing
    pad is already sufficient to prove that an unwind transfer reaches it.
    """

    end = len(data)
    if offset + 3 > end:
        raise _UnsupportedLSDA("truncated LSDA header")

    lpstart_encoding = data[offset]
    offset += 1
    if lpstart_encoding != _DW_EH_PE_OMIT:
        raise _UnsupportedLSDA("explicit LSDA landing-pad base")

    ttype_encoding = data[offset]
    offset += 1
    if ttype_encoding != _DW_EH_PE_OMIT:
        # The type-table offset is relative to the field after this ULEB128.
        # We do not need the type table to recover control-flow destinations.
        _, offset = _read_uleb128(data, offset, end)

    call_site_encoding = data[offset]
    offset += 1
    if call_site_encoding & 0xF0:
        raise _UnsupportedLSDA("relative call-site encoding")
    call_site_length, offset = _read_uleb128(data, offset, end)
    table_end = offset + call_site_length
    if table_end > end:
        raise _UnsupportedLSDA("truncated LSDA call-site table")

    call_sites: list[ExceptionalCallSite] = []
    while offset < table_end:
        start, offset = _read_call_site_value(
            data, offset, table_end, call_site_encoding, byteorder=byteorder
        )
        length, offset = _read_call_site_value(
            data, offset, table_end, call_site_encoding, byteorder=byteorder
        )
        landing_pad, offset = _read_call_site_value(
            data, offset, table_end, call_site_encoding, byteorder=byteorder
        )
        _, offset = _read_uleb128(data, offset, table_end)
        if landing_pad == 0 or length == 0:
            continue
        call_sites.append(
            ExceptionalCallSite(
                start_addr=function_addr + start,
                end_addr=function_addr + start + length,
                landing_pad_addr=function_addr + landing_pad,
            )
        )

    if offset != table_end:
        raise _UnsupportedLSDA("misaligned LSDA call-site table")
    return tuple(call_sites)


@lru_cache(maxsize=64)
def _exception_sites_by_elf(binary: str) -> tuple[_FunctionExceptionSites, ...]:
    """Parse every LSDA in one file once and retain link-time addresses."""

    with Path(binary).open("rb") as stream:
        elf = ELFFile(stream)
        if elf.header["e_type"] not in {"ET_EXEC", "ET_DYN"}:
            return ()
        exception_table = elf.get_section_by_name(".gcc_except_table")
        if exception_table is None:
            return ()

        section_addr = int(exception_table["sh_addr"])
        section_data = exception_table.data()
        byteorder = "little" if elf.little_endian else "big"
        records: list[_FunctionExceptionSites] = []
        for entry in elf.get_dwarf_info().EH_CFI_entries():
            if not isinstance(entry, FDE) or entry.lsda_pointer is None:
                continue
            start_addr = int(entry["initial_location"])
            end_addr = start_addr + int(entry["address_range"])
            lsda_offset = int(entry.lsda_pointer) - section_addr
            if not 0 <= lsda_offset < len(section_data):
                continue
            try:
                call_sites = _parse_lsda_call_sites(
                    section_data,
                    lsda_offset,
                    function_addr=start_addr,
                    byteorder=byteorder,
                )
            except _UnsupportedLSDA:
                continue
            if call_sites:
                records.append(
                    _FunctionExceptionSites(start_addr, end_addr, call_sites)
                )
        return tuple(records)


def exceptional_call_sites_for_function(
    project: Project, bounds: FunctionBounds
) -> tuple[ExceptionalCallSite, ...]:
    """Return exact LSDA exceptional transfers for one linked ELF function.

    All addresses returned are rebased to the project's loader address space.
    Unsupported metadata and non-file-backed objects deliberately return no
    records, leaving the extractor's ordinary direct-flow result unchanged.
    """

    try:
        obj = project.loader.find_object_containing(bounds.addr)
        binary = getattr(obj, "binary", None)
        if obj is None or not isinstance(binary, (str, Path)):
            return ()
        linked_base = int(obj.linked_base)
        mapped_base = int(obj.mapped_base)
        address_delta = mapped_base - linked_base

        link_time_addr = bounds.addr - address_delta
        fde = next(
            (
                record
                for record in _exception_sites_by_elf(str(Path(binary).resolve()))
                if record.start_addr <= link_time_addr < record.end_addr
            ),
            None,
        )
        if fde is None:
            return ()
        return tuple(
            ExceptionalCallSite(
                start_addr=site.start_addr + address_delta,
                end_addr=site.end_addr + address_delta,
                landing_pad_addr=site.landing_pad_addr + address_delta,
            )
            for site in fde.call_sites
        )
    except (ELFError, AssertionError, KeyError, OSError, TypeError, ValueError):
        return ()
