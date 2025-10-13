from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import regex as re
import pandas as pd

from .config import load_lexicon, load_affixes

PUNCT_RE = re.compile(r"^\p{P}+$")
NUM_RE = re.compile(r"^\p{N}+$")


def _tag_from_affixes(segments: str, affixes: Dict[str, Any]) -> Optional[str]:
    # Check voice markers on segment ends
    vm = affixes.get("voice_markers", {})
    segs = (segments or "").split("+")
    for seg in segs:
        for marker, meta in vm.items():
            if seg.endswith(marker):
                return meta.get("label", "AFF")
    return None


def tag_token(
    token: str,
    segments: Optional[str],
    lex_lookup: Dict[str, Tuple[str, str, str]],
    affixes: Dict[str, Any],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Return (tag, lemma, gloss). Priority:

    1) punctuation/number

    2) lexicon POS (also returns lemma/gloss)

    3) affix-based label (e.g., FOC)

    4) default WORD

    """
    if PUNCT_RE.fullmatch(token):
        return "PUNCT", None, None
    if NUM_RE.fullmatch(token):
        return "NUM", None, None
    if token in lex_lookup:
        pos, lemma, gloss = lex_lookup[token]
        return pos or "WORD", lemma or None, gloss or None
    aff_tag = _tag_from_affixes(segments or token, affixes)
    if aff_tag:
        return aff_tag, None, None
    return "WORD", None, None


def tag_tokens(
    tokens: List[str],
    segments: Optional[List[str]] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> List[str]:
    lex = load_lexicon(
        lexicon_path) if lexicon_path or lexicon_path is None else None
    aff = load_affixes(affixes_path) if affixes_path or affixes_path is None else {
        "voice_markers": {}, "affix_types": {}}
    lex_lookup: Dict[str, Tuple[str, str, str]] = {}
    if isinstance(lex, pd.DataFrame):
        for _, row in lex.iterrows():
            lex_lookup[row["form"]] = (row.get("pos", "WORD"), row.get(
                "lemma", ""), row.get("gloss", ""))
    if segments is None:
        segments = tokens
    tags = []
    for t, s in zip(tokens, segments):
        tag, _, _ = tag_token(t, s, lex_lookup, aff)
        tags.append(tag)
    return tags


def tag_file_tsv(
    infile: Path,
    outfile: Path,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> None:
    df = pd.read_csv(infile, sep="\t", dtype=str, keep_default_na=False)
    if "token" not in df.columns:
        raise ValueError("Input TSV must have a 'token' column.")
    segs = df["segments"].tolist(
    ) if "segments" in df.columns else df["token"].tolist()

    # build lex+aff and also add lemma/gloss columns when available
    lex = load_lexicon(
        lexicon_path) if lexicon_path or lexicon_path is None else None
    aff = load_affixes(affixes_path) if affixes_path or affixes_path is None else {
        "voice_markers": {}, "affix_types": {}}
    lex_lookup = {}
    if isinstance(lex, pd.DataFrame):
        lex_lookup = {row["form"]: (row.get("pos", "WORD"), row.get(
            "lemma", ""), row.get("gloss", "")) for _, row in lex.iterrows()}

    tags, lemmas, glosses = [], [], []
    for tok, seg in zip(df["token"].tolist(), segs):
        tag, lemma, gloss = tag_token(tok, seg, lex_lookup, aff)
        tags.append(tag)
        lemmas.append(lemma or "")
        glosses.append(gloss or "")

    df["tag"] = tags
    # include lemma/gloss if not present
    if "lemma" not in df.columns:
        df["lemma"] = lemmas
    if "gloss" not in df.columns:
        df["gloss"] = glosses

    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, sep="\t", index=False)


def tag_tokens_with_lex(
    tokens: List[str],
    segments: Optional[List[str]] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Return [(tag, lemma, gloss), ...] using the same logic as tag_file_tsv."""
    lex = load_lexicon(
        lexicon_path) if lexicon_path or lexicon_path is None else None
    aff = load_affixes(affixes_path) if affixes_path or affixes_path is None else {
        "voice_markers": {}, "affix_types": {}}
    lex_lookup: Dict[str, Tuple[str, str, str]] = {}
    if isinstance(lex, pd.DataFrame):
        for _, row in lex.iterrows():
            lex_lookup[row["form"]] = (row.get("pos", "WORD"), row.get(
                "lemma", ""), row.get("gloss", ""))
    if segments is None:
        segments = tokens
    out: List[Tuple[str, Optional[str], Optional[str]]] = []
    for t, s in zip(tokens, segments):
        out.append(tag_token(t, s, lex_lookup, aff))
    return out
