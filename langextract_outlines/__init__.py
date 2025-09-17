"""LangExtract provider plugin for Outlines."""

from langextract_outlines.provider import OutlinesProvider

__all__ = ["OutlinesProvider"]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
