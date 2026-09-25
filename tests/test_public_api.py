"""The public API (P7.7): what a minor version keeps compatible (docs/compatibility.md).

The public API is the `__all__` of the modules in PUBLIC_MODULES: the names, the signatures, and the fields of the
public dataclasses. Everything under `jevstiller._*` is internal. If this test fails because you changed the public
API on purpose, regenerate the snapshot and say what changed in CHANGELOG.md:

    python tests/test_public_api.py --update
"""
from __future__ import annotations

import dataclasses
import difflib
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

import jevstiller

PUBLIC_MODULES = ["jevstiller", "jevstiller.server", "jevstiller.encoders", "jevstiller.teachers",
                  "jevstiller.teachers.jev"]
SNAPSHOT = Path(__file__).with_name("public_api.txt")


def _sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def _describe_class(path: str, cls: type) -> list[str]:
    out = [f"class {path}{_sig(cls)}"]
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            out.append(f"    field {f.name}: {f.type}")
    elif getattr(cls, "_is_protocol", False):
        for name, t in sorted(getattr(cls, "__annotations__", {}).items()):
            out.append(f"    attribute {name}: {t}")
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if isinstance(member, property):
            out.append(f"    property {name}")
        elif isinstance(member, (staticmethod, classmethod)):
            out.append(f"    {type(member).__name__} {name}{_sig(member.__func__)}")
        elif inspect.isfunction(member):
            out.append(f"    def {name}{_sig(member)}")
    return out


def describe() -> list[str]:
    lines: list[str] = []
    for modname in PUBLIC_MODULES:
        mod = importlib.import_module(modname)
        lines.append(f"module {modname}")
        for name in sorted(mod.__all__):
            obj = getattr(mod, name)
            path = f"{modname}.{name}"
            if modname != "jevstiller" and getattr(jevstiller, name, None) is obj:
                lines.append(f"  {path} -> jevstiller.{name}")
            elif inspect.isclass(obj):
                lines += ["  " + line for line in _describe_class(path, obj)]
            elif callable(obj):
                lines.append(f"  def {path}{_sig(obj)}")
            elif isinstance(obj, (dict, list, tuple, str, int, float)):
                lines.append(f"  {path}: {type(obj).__name__}")
            else:                                        # a type alias: its repr is the same on every Python
                lines.append(f"  {path} = {obj!r}")
    return lines


def test_public_api_matches_the_snapshot():
    now = describe()
    before = SNAPSHOT.read_text().splitlines()
    diff = "\n".join(difflib.unified_diff(before, now, "public_api.txt", "now", lineterm=""))
    assert now == before, (f"The public API changed:\n{diff}\n\nIf that's intended: python tests/test_public_api.py "
                           "--update, and describe the change in CHANGELOG.md.")


def test_every_module_is_public_or_underscored():
    """A new module without a leading underscore would look public: list it in PUBLIC_MODULES, or rename it."""
    found = [m.name for m in pkgutil.walk_packages(jevstiller.__path__, "jevstiller.")
             if not any(part.startswith("_") for part in m.name.split(".")[1:])]
    assert sorted(found) == sorted(PUBLIC_MODULES[1:])


def test_everything_in_all_exists():
    for modname in PUBLIC_MODULES:
        mod = importlib.import_module(modname)
        assert all(hasattr(mod, name) for name in mod.__all__), modname


if __name__ == "__main__" and sys.argv[1:] == ["--update"]:
    SNAPSHOT.write_text("\n".join(describe()) + "\n")
    print(f"wrote {SNAPSHOT}")
