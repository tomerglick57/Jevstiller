"""CITATION.cff must name the version being released: the release workflow checks the tag against pyproject,
and this check ties the citation file to the same number, so "Cite this repository" and Zenodo never lag."""

import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def test_citation_version_matches_pyproject():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    cff = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(version)}$", cff, re.M), (
        f"CITATION.cff must say 'version: {version}' (and a matching date-released)"
    )
    assert re.search(r"^date-released: \d{4}-\d{2}-\d{2}$", cff, re.M)
