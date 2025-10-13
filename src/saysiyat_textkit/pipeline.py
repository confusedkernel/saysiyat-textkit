from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from .normalization import normalize_text, tokenize_keep_clitic
from .segmentation import segment_tokens
from .tagging import tag_tokens_with_lex


def _normalize_and_tokenize_lines(text: str) -> List[Tuple[int, int, str]]:
    """Normalize each non-empty line, tokenize with notebook-parity rules,
    and return (sent_id, token_id, token) triples."""
    lines = [s for s in (text.splitlines() or [text]) if s.strip()]
    rows: List[Tuple[int, int, str]] = []
    for i, sent in enumerate(lines, start=1):
        norm = normalize_text(sent, lowercase=True, keep_diacritics=True)
        toks = tokenize_keep_clitic(norm)
        for j, tok in enumerate(toks, start=1):
            rows.append((i, j, tok))
    return rows


def _coerce_segments_to_spaces(seg: object) -> str:
    """
    Accepts:
      - list/tuple of pieces -> ' '.join(pieces)
      - string with '+' separators -> replace '+' with ' '
      - plain string -> return as-is
      - None -> ''
    Ensures the tagging step receives consistent 'space-joined' segmentation.
    """
    if seg is None:
        return ""
    if isinstance(seg, (list, tuple)):
        return " ".join(str(s) for s in seg if str(s))
    s = str(seg)
    # If user/model produced "a+b+c", normalize to spaces
    if "+" in s and " " not in s:
        return " ".join(p for p in s.split("+") if p)
    # Collapse any accidental multiple spaces
    return " ".join(s.split())


def run_pipeline_from_text(
    text: str,
    morfessor_model_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run normalization → segmentation → tagging and return a DataFrame with
    columns: sent_id, token_id, token, segments, tag, lemma, gloss.

    - If lexicon_path/affixes_path are None, the packaged defaults are used.
    - segment_tokens may return lists of morphemes or a joined string; both are handled.
    """
    rows = _normalize_and_tokenize_lines(text)
    df = pd.DataFrame(rows, columns=["sent_id", "token_id", "token"])

    # Morfessor-driven segmentation (uses packaged lex/affixes by default)
    segs = segment_tokens(
        df["token"].tolist(),
        morfessor_model_path=morfessor_model_path,
        lexicon_path=lexicon_path,     # None => packaged
        affixes_path=affixes_path,     # None => packaged
    )

    # Normalize all segment outputs to space-joined strings
    df["segments"] = [ _coerce_segments_to_spaces(s) for s in segs ]

    # Tagging (+ lemma/gloss) using packaged lex/affixes by default
    tlg = tag_tokens_with_lex(
        df["token"].tolist(),
        df["segments"].tolist(),
        lexicon_path=lexicon_path,     # None => packaged
        affixes_path=affixes_path,     # None => packaged
    )

    # tag_tokens_with_lex should return a list of (tag, lemma, gloss)
    # Make this robust even if it's empty or mismatched length.
    if not tlg or len(tlg) != len(df):
        tags = [""] * len(df)
        lemmas = [""] * len(df)
        glosses = [""] * len(df)
    else:
        tags, lemmas, glosses = zip(*tlg)
    df["tag"] = list(tags)
    df["lemma"] = [ (l or "") for l in lemmas ]
    df["gloss"] = [ (g or "") for g in glosses ]

    # Ensure stable dtypes/order
    return df[["sent_id", "token_id", "token", "segments", "tag", "lemma", "gloss"]]


def run_pipeline_file(
    infile: Path,
    outfile: Path,
    morfessor_model_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> None:
    """
    Read a raw text file (one sentence per line), run the full pipeline, and write a TSV.
    """
    text = Path(infile).read_text(encoding="utf-8")
    df = run_pipeline_from_text(
        text,
        morfessor_model_path=morfessor_model_path,
        lexicon_path=lexicon_path,
        affixes_path=affixes_path,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, sep="\t", index=False)
