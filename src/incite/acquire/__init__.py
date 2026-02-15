"""PDF acquisition pipeline for inCite.

Acquires PDFs from free sources (Unpaywall, arXiv) and institutional
library proxies (EZproxy via Playwright browser automation).
"""

from incite.acquire.config import ProxyConfig, load_proxy_config, save_proxy_config
from incite.acquire.unpaywall import UnpaywallClient, UnpaywallResult

__all__ = [
    "ProxyConfig",
    "UnpaywallClient",
    "UnpaywallResult",
    "load_proxy_config",
    "save_proxy_config",
]
