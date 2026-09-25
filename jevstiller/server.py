"""The drop-in proxy as a library (the `server` extra): `create_app` builds the ASGI app that `jevstiller serve`
runs, for embedding it in your own server or tests. See docs/configuration.md ("Library")."""
from ._server import KeyRegistry, ProxySettings, create_app, load_salt

__all__ = ["KeyRegistry", "ProxySettings", "create_app", "load_salt"]
