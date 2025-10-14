from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
from importlib import resources
import json
import pandas as pd
from pathlib import Path


def _default_path(package_rel: str) -> Path:
    # Access packaged data at runtime
    with resources.as_file(resources.files("saysiyat_textkit.data") / package_rel) as p:
        return p


def load_lexicon(path: Optional[str] = None) -> pd.DataFrame:
    """Load lexicon TSV with columns: form, lemma, freq, seg, pos, gloss.

    If path is None, load packaged default.

    """
    p = Path(path) if path else _default_path("lexicon.tsv")
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    required = {"form", "lemma", "freq", "seg", "POS", "gloss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lexicon missing columns: {missing}")
    return df


def load_affixes(path: Optional[str] = None) -> Dict[str, Any]:
    """Load affixes JSON (voice markers & affix types). If None, use default."""
    p = Path(path) if path else _default_path("affixes.json")
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    # normalize structure
    data.setdefault("voice_markers", {})
    data.setdefault("affix_types", {})
    return data
