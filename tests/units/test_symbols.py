"""Fast tests for symbol-table normalization used by CFG reconstruction."""

from types import SimpleNamespace
import pytest

from bingraph.helpers.symbols import is_arm_mapping_symbol, list_function_symbols


@pytest.mark.parametrize(
    "arch", ["ARMEL", "ARMHF", "ARMfuture", "AARCH64", "AMD64", "MIPS32"]
)
@pytest.mark.parametrize(
    "name,mapping",
    [
        ("$a", True),
        ("$d.42", True),
        ("$t.", True),
        ("$x.suffix", True),
        ("$table_loop", False),
        ("$a_suffix", False),
        (None, False),
        ("", False),
    ],
)
def test_mapping_symbols_require_arm_and_an_exact_marker(arch, name, mapping):
    assert is_arm_mapping_symbol(arch, name) is (
        mapping and arch.startswith(("ARM", "AARCH64"))
    )


def _symbol(
    addr: int,
    name: str,
    size: int,
    *,
    is_function: bool = True,
) -> SimpleNamespace:
    """Create the loader-symbol shape used by symbol normalization."""

    return SimpleNamespace(
        rebased_addr=addr,
        name=name,
        size=size,
        is_function=is_function,
        is_import=False,
    )


class _Project:
    """Hashable project stand-in compatible with the symbol-list cache."""

    def __init__(
        self,
        symbols: list[SimpleNamespace],
        *,
        sections: dict[int, SimpleNamespace] | None = None,
    ) -> None:
        """Expose loader symbols through the minimal angr-project shape."""

        sections = sections or {}
        self.loader = SimpleNamespace(
            main_object=SimpleNamespace(
                symbols=symbols,
                find_section_containing=lambda addr: sections.get(addr),
            )
        )


def test_symbol_listing_sorts_deduplicates_and_infers_sizes() -> None:
    """Infer a zero-sized function span from the next distinct symbol address."""

    project = _Project(
        [
            _symbol(0x1020, "later", 4),
            _symbol(0x1000, "entry", 0),
            _symbol(0x1000, "entry", 0),
            _symbol(0x1010, "not_a_function", 9, is_function=False),
        ]
    )
    list_function_symbols.cache_clear()

    symbols = list_function_symbols(project)

    assert [(symbol.addr, symbol.name, symbol.size) for symbol in symbols] == [
        (0x1000, "entry", 0x20),
        (0x1020, "later", 4),
    ]


def test_symbol_listing_drops_unresolved_trailing_zero_sized_symbol() -> None:
    """Avoid returning a CFG candidate whose function bound cannot be inferred."""

    project = _Project([_symbol(0x1000, "last", 0)])
    list_function_symbols.cache_clear()

    assert list_function_symbols(project) == []


def test_symbol_listing_caps_an_inferred_size_at_its_section_end() -> None:
    """Avoid extending a zero-sized symbol into the next executable section."""

    project = _Project(
        [_symbol(0x1000, "init", 0), _symbol(0x1200, "text", 4)],
        sections={0x1000: SimpleNamespace(vaddr=0x1000, memsize=0x20)},
    )
    list_function_symbols.cache_clear()

    symbols = list_function_symbols(project)

    assert [(symbol.name, symbol.size) for symbol in symbols] == [
        ("init", 0x20),
        ("text", 4),
    ]


def test_symbol_listing_preserves_zero_sized_aliases_at_one_entry() -> None:
    """Infer one shared span for aliases before the next distinct address."""

    project = _Project(
        [
            _symbol(0x1000, "alias", 0),
            _symbol(0x1000, "entry", 0),
            _symbol(0x1080, "next", 4),
        ]
    )
    list_function_symbols.cache_clear()

    symbols = list_function_symbols(project)

    assert [(symbol.name, symbol.size) for symbol in symbols] == [
        ("alias", 0x80),
        ("entry", 0x80),
        ("next", 4),
    ]
