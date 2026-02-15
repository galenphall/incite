"""inCite menu bar application.

Requires the ``rumps`` package (macOS only).
Install with: pip install incite[tray]
"""


def __getattr__(name: str):
    if name == "InCiteTray":
        from incite.tray.app import InCiteTray

        return InCiteTray
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["InCiteTray"]
