from typing import Any, Union

STYLE_CLASSIC = {
    "COLOR_SCHEME": {
        "EDGECOLOR_CONDITIONAL_TRUE": {"color": "green"},
        "EDGECOLOR_CONDITIONAL_FALSE": {"color": "red"},
        "EDGECOLOR_UNCONDITIONAL": {"color": "blue"},
        "EDGECOLOR_NEXT": {"color": "blue", "style": "dashed"},
        "EDGECOLOR_INDIRECT": {"color": "orange"},
        "EDGECOLOR_UNRESOLVED_INDIRECT": {"color": "orange", "style": "dashed"},
        "EDGECOLOR_CALL": {"color": "gray"},
        "EDGECOLOR_RET": {"color": "gray", "style": "dotted"},
        "EDGECOLOR_FAKE_RET": {"color": "gray", "style": "dashed"},
        "EDGECOLOR_EXCEPTION": {"color": "gray", "style": "dotted"},
        "EDGECOLOR_UNKNOWN": {"color": "purple"},
    }
}

STYLE_THICK = {
    "COLOR_SCHEME": {
        "EDGECOLOR_CONDITIONAL_TRUE": {"color": "green", "penwidth": "2"},
        "EDGECOLOR_CONDITIONAL_FALSE": {"color": "red", "penwidth": "2"},
        "EDGECOLOR_UNCONDITIONAL": {"color": "blue", "penwidth": "2"},
        "EDGECOLOR_NEXT": {"color": "blue", "style": "dashed", "penwidth": "2"},
        "EDGECOLOR_INDIRECT": {"color": "orange", "penwidth": "2"},
        "EDGECOLOR_UNRESOLVED_INDIRECT": {
            "color": "orange",
            "style": "dashed",
            "penwidth": "2",
        },
        "EDGECOLOR_CALL": {"color": "gray", "penwidth": "2"},
        "EDGECOLOR_RET": {"color": "gray", "style": "dotted", "penwidth": "2"},
        "EDGECOLOR_FAKE_RET": {"color": "gray", "style": "dashed", "penwidth": "2"},
        "EDGECOLOR_EXCEPTION": {"color": "gray", "style": "dotted", "penwidth": "2"},
        "EDGECOLOR_UNKNOWN": {"color": "purple", "penwidth": "2"},
    }
}

STYLE_BLACK = {
    "COLOR_SCHEME": {
        "EDGECOLOR_CONDITIONAL_TRUE": {"color": "black"},
        "EDGECOLOR_CONDITIONAL_FALSE": {"color": "black"},
        "EDGECOLOR_UNCONDITIONAL": {"color": "black"},
        "EDGECOLOR_NEXT": {"color": "black", "style": "dashed"},
        "EDGECOLOR_INDIRECT": {"color": "black"},
        "EDGECOLOR_UNRESOLVED_INDIRECT": {"color": "black", "style": "dashed"},
        "EDGECOLOR_CALL": {"color": "gray"},
        "EDGECOLOR_RET": {"color": "gray", "style": "dotted"},
        "EDGECOLOR_FAKE_RET": {"color": "gray", "style": "dashed"},
        "EDGECOLOR_EXCEPTION": {"color": "gray", "style": "dotted"},
        "EDGECOLOR_UNKNOWN": {"color": "purple"},
    }
}

STYLE_DARK = {
    "COLOR_SCHEME": {
        "EDGECOLOR_CONDITIONAL_TRUE": {"color": "#006400"},
        "EDGECOLOR_CONDITIONAL_FALSE": {"color": "#8b0000"},
        "EDGECOLOR_UNCONDITIONAL": {"color": "#00008b"},
        "EDGECOLOR_NEXT": {"color": "#00008b", "style": "dashed"},
        "EDGECOLOR_INDIRECT": {"color": "#ff8c00"},
        "EDGECOLOR_UNRESOLVED_INDIRECT": {"color": "#ff8c00", "style": "dashed"},
        "EDGECOLOR_CALL": {"color": "#a9a9a9"},
        "EDGECOLOR_RET": {"color": "#a9a9a9", "style": "dotted"},
        "EDGECOLOR_FAKE_RET": {"color": "#a9a9a9", "style": "dashed"},
        "EDGECOLOR_EXCEPTION": {"color": "#a9a9a9", "style": "dotted"},
        "EDGECOLOR_UNKNOWN": {"color": "#850A93"},
    }
}

STYLE_LIGHT = {
    "COLOR_SCHEME": {
        "EDGECOLOR_CONDITIONAL_TRUE": {"color": "#ADFF2F"},
        "EDGECOLOR_CONDITIONAL_FALSE": {"color": "#F08080"},
        "EDGECOLOR_UNCONDITIONAL": {"color": "#87CEFA"},
        "EDGECOLOR_NEXT": {"color": "#87CEFA", "style": "dashed"},
        "EDGECOLOR_INDIRECT": {"color": "#FFD700"},
        "EDGECOLOR_UNRESOLVED_INDIRECT": {"color": "#FFD700", "style": "dashed"},
        "EDGECOLOR_CALL": {"color": "#C0C0C0"},
        "EDGECOLOR_RET": {"color": "#C0C0C0", "style": "dotted"},
        "EDGECOLOR_FAKE_RET": {"color": "#C0C0C0", "style": "dashed"},
        "EDGECOLOR_EXCEPTION": {"color": "#C0C0C0", "style": "dotted"},
        "EDGECOLOR_UNKNOWN": {"color": "#BE69B9"},
    }
}


class Style:
    def __init__(self, st: Any) -> None:
        self.style = st

    def make_edge(self, edge: Any, edge_type: str) -> None:
        edge_attrs = self.style["COLOR_SCHEME"]["EDGECOLOR_" + edge_type.upper()]
        for k, v in edge_attrs.items():
            edge.pydot.set(k, v)


_style = Style(STYLE_CLASSIC)


def set_style(c: Union[str, Any]) -> None:
    global _style
    if type(c) is str:
        if c == "classic":
            set_style(STYLE_CLASSIC)
        elif c == "thick":
            set_style(STYLE_THICK)
        elif c == "black":
            set_style(STYLE_BLACK)
        elif c == "dark":
            set_style(STYLE_DARK)
        elif c == "light":
            set_style(STYLE_LIGHT)
        else:
            raise KeyError("Style '%s' not defined" % c)
    else:
        _style = Style(c)


def get_style() -> Style:
    return _style
