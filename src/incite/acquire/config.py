"""Proxy configuration management.

Stores proxy settings in ~/.incite/config.json under the "proxy" key,
reusing the existing config infrastructure from webapp/state.py.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProxyConfig:
    """User's library proxy configuration."""

    proxy_type: str = ""  # "ezproxy_prefix" | "ezproxy_suffix" | "vpn"
    proxy_url: str = ""  # e.g. "https://proxy.lib.umich.edu/login?url="
    proxy_suffix: str = ""  # e.g. ".proxy.lib.umich.edu"
    institution_name: str = ""  # For display only
    test_doi: str = "10.1038/nature12373"  # Known-paywalled DOI to verify access
    session_dir: Path = field(default_factory=lambda: Path.home() / ".incite" / "proxy_session")

    @property
    def is_configured(self) -> bool:
        """Check if proxy is configured (has a type set)."""
        return bool(self.proxy_type)

    @property
    def test_url(self) -> str:
        """Construct a test URL using the configured proxy and test DOI."""
        target = f"https://doi.org/{self.test_doi}"
        if self.proxy_type == "ezproxy_prefix" and self.proxy_url:
            return self.proxy_url + target
        elif self.proxy_type == "ezproxy_suffix":
            # For suffix mode, we'd need to resolve the DOI first
            return target
        return target


# Presets for common institutions.
INSTITUTION_PRESETS: dict[str, ProxyConfig] = {
    "umich": ProxyConfig(
        proxy_type="ezproxy_prefix",
        proxy_url="https://proxy.lib.umich.edu/login?url=",
        institution_name="University of Michigan",
        test_doi="10.1038/nature12373",
    ),
}


def load_proxy_config() -> ProxyConfig:
    """Load proxy configuration from ~/.incite/config.json.

    Returns:
        ProxyConfig (may be unconfigured if no proxy settings saved).
    """
    from incite.webapp.state import get_config

    config = get_config()
    proxy_data = config.get("proxy", {})

    if not proxy_data:
        return ProxyConfig()

    return ProxyConfig(
        proxy_type=proxy_data.get("type", ""),
        proxy_url=proxy_data.get("url", ""),
        proxy_suffix=proxy_data.get("suffix", ""),
        institution_name=proxy_data.get("institution", ""),
        test_doi=proxy_data.get("test_doi", "10.1038/nature12373"),
        session_dir=Path(
            proxy_data.get("session_dir", str(Path.home() / ".incite" / "proxy_session"))
        ),
    )


def save_proxy_config(proxy_config: ProxyConfig) -> None:
    """Save proxy configuration to ~/.incite/config.json.

    Merges into the existing config (doesn't overwrite other settings).

    Args:
        proxy_config: The proxy configuration to save.
    """
    from incite.webapp.state import get_config, save_config

    config = get_config()
    config["proxy"] = {
        "type": proxy_config.proxy_type,
        "url": proxy_config.proxy_url,
        "suffix": proxy_config.proxy_suffix,
        "institution": proxy_config.institution_name,
        "test_doi": proxy_config.test_doi,
        "session_dir": str(proxy_config.session_dir),
    }
    save_config(config)
