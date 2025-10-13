from __future__ import annotations
from typing import Optional, Dict, Any, Set
from pathlib import Path
import os
import json
import importlib.resources as ir

ENV_LEXICON = "SAYSIYAT_LEXICON"
ENV_AFFIXES = "SAYSIYAT_AFFIXES"


def _pkg_data_path(filename: str) -> Path:
    with ir.as_file(ir.files("saysiyat_textkit") / "data" / filename) as p:
        return Path(p)


def resolve_resource(kind: str, user_path: Optional[str]) -> Path:
    """Resolve a path for 'lexicon' or 'affixes' with precedence:
    1) explicit user_path if exists
    2) env var (saysiyat_LEXICON / saysiyat_AFFIXES)
    3) package data default
    """
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
    env = os.getenv(ENV_LEXICON if kind == "lexicon" else ENV_AFFIXES)
    if env:
        p = Path(env)
        if p.exists():
            return p
    return _pkg_data_path("lexicon.tsv" if kind == "lexicon" else "affixes.json")


def load_lexicon(path: Optional[str] = None) -> Set[str]:
    p = resolve_resource("lexicon", path)
    forms = set()
    with open(p, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts:
                forms.add(parts[0])
    return forms


def load_affixes(path: Optional[str] = None) -> Dict[str, Any]:
    p = resolve_resource("affixes", path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
