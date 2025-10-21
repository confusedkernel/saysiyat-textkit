from __future__ import annotations
from typing import Optional, Dict, Any, Set, Union
from pathlib import Path
import os
import json
import pandas as pd
import importlib.resources as ir

ENV_LEXICON = "SAYSIYAT_LEXICON"
ENV_AFFIXES = "SAYSIYAT_AFFIXES"


def _tqdm(iterable, disable=False, desc: str = ""):
    """
    Safe wrapper for tqdm that falls back gracefully if tqdm is not available.
    
    Args:
        iterable: The iterable to wrap
        disable: Whether to disable the progress bar
        desc: Description to show in the progress bar
        
    Returns:
        Either a tqdm-wrapped iterable or the original iterable
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, disable=disable, desc=desc)
    except ImportError:
        return iterable
    

def _pkg_data_path(filename: str) -> Path:
    """Get path to packaged data file."""
    with ir.as_file(ir.files("stk") / "data" / filename) as p:
        return Path(p)


def resolve_resource(kind: str, user_path: Optional[str]) -> Path:
    """Resolve a path for 'lexicon' or 'affixes' with precedence:
    1) explicit user_path if exists
    2) env var (SAYSIYAT_LEXICON / SAYSIYAT_AFFIXES)
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


def load_lexicon(
    path: Optional[Union[str, Path]] = None, 
    return_type: str = "dataframe"
) -> Union[pd.DataFrame, Dict[str, Dict[str, Any]], Set[str]]:
    """
    Load lexicon with flexible return types.
    
    Args:
        path: Path to lexicon file, or None for default
        return_type: One of "dataframe", "dict", or "set"
            - "dataframe": Returns pd.DataFrame with all columns
            - "dict": Returns dict[form -> {lemma, freq, seg, pos, gloss}]
            - "set": Returns set of forms only
    
    Returns:
        DataFrame, dict, or set depending on return_type
    """
    if path is None:
        # Use packaged default
        lexicon_path = _pkg_data_path("lexicon.tsv")
    else:
        lexicon_path = Path(path)
    
    # Read as DataFrame first (most flexible)
    df = pd.read_csv(lexicon_path, sep="\t", dtype=str, keep_default_na=False)
    
    # Validate required columns (POS is optional, others are required)
    required = {"form", "lemma", "freq", "seg", "gloss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lexicon missing columns: {missing}")
    
    # Handle POS column variants (optional)
    if "POS" not in df.columns and "pos" not in df.columns:
        df["POS"] = ""  # Add empty POS column if neither exists
    elif "pos" in df.columns and "POS" not in df.columns:
        df["POS"] = df["pos"]  # Rename pos to POS for consistency
    
    if return_type == "dataframe":
        return df
    elif return_type == "set":
        return set(df["form"].astype(str))
    elif return_type == "dict":
        # Convert to the dict format used in normalization.py
        lex = {}
        for _, row in df.iterrows():
            form = str(row["form"]).strip()
            if not form:
                continue
                
            lemma = str(row["lemma"]).strip() or form
            
            # Handle freq: keep "" if empty; else parse to int
            freq_raw = str(row["freq"]).strip()
            if freq_raw == "":
                freq = ""
            else:
                try:
                    freq = int(float(freq_raw))
                except Exception:
                    freq = ""
            
            seg = str(row["seg"]).strip()
            pos = str(row["POS"]).strip()
            gloss = str(row["gloss"]).strip()
            
            lex[form] = {
                "lemma": lemma,
                "freq": freq,
                "seg": seg,
                "pos": pos,
                "gloss": gloss,
            }
        return lex
    else:
        raise ValueError(f"Invalid return_type: {return_type}. Must be 'dataframe', 'dict', or 'set'")


def load_affixes(path: Optional[str] = None) -> Dict[str, Any]:
    """Load affixes JSON (voice markers & affix types). If None, use default."""
    affixes_path = resolve_resource("affixes", path)
    with open(affixes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Normalize structure
    data.setdefault("voice_markers", {})
    data.setdefault("affix_types", {})
    return data


def get_affix_sets(affixes_data: Optional[Dict[str, Any]] = None) -> tuple[set, set, set]:
    """
    Extract prefix, suffix, and infix sets from affixes JSON data.
    Returns (prefixes, suffixes, infixes) as sets of strings.
    """
    if affixes_data is None:
        affixes_data = load_affixes()
    
    prefixes, suffixes, infixes = set(), set(), set()
    
    if "affix_types" in affixes_data:
        for affix_key, meta in affixes_data["affix_types"].items():
            affix_type = str((meta or {}).get("type", "")).lower()
            clean_key = str(affix_key).strip("-")  # Remove leading/trailing dashes
            
            if affix_type == "prefix":
                prefixes.add(clean_key)
            elif affix_type == "suffix":
                suffixes.add(clean_key)
            elif affix_type == "infix":
                infixes.add(clean_key)
            else:
                # Fallback: infer from dash patterns
                if affix_key.startswith("-") and affix_key.endswith("-"):
                    infixes.add(clean_key)
                elif affix_key.endswith("-"):
                    prefixes.add(clean_key)
                elif affix_key.startswith("-"):
                    suffixes.add(clean_key)
                else:
                    prefixes.add(clean_key)  # Default to prefix
    
    return prefixes, suffixes, infixes


def get_voice_mappings(affixes_data: Optional[Dict[str, Any]] = None) -> tuple[dict, dict, dict]:
    """
    Extract voice feature mappings from affixes JSON data.
    Returns (voice_prefix, voice_infix, voice_suffix) dictionaries.
    Each maps affix -> (VOICE, POL) tuple.
    """
    if affixes_data is None:
        affixes_data = load_affixes()
    
    voice_prefix, voice_infix, voice_suffix = {}, {}, {}
    
    if "affix_types" in affixes_data:
        for affix_key, meta in affixes_data["affix_types"].items():
            if not meta or "features" not in meta:
                continue
                
            features = meta["features"]
            voice = features.get("VOICE")
            pol = features.get("POL")
            
            if voice and pol:
                clean_key = str(affix_key).strip("-")
                affix_type = str(meta.get("type", "")).lower()
                
                if affix_type == "prefix":
                    voice_prefix[clean_key] = (voice, pol)
                elif affix_type == "infix":
                    voice_infix[clean_key] = (voice, pol)
                elif affix_type == "suffix":
                    voice_suffix[clean_key] = (voice, pol)
    
    return voice_prefix, voice_infix, voice_suffix