# src/saysiyat_textkit/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
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
    """Return a space-joined segmentation string from list/tuple/'+'/plain/None."""
    if seg is None:
        return ""
    if isinstance(seg, (list, tuple)):
        return " ".join(str(s) for s in seg if str(s))
    s = str(seg)
    if "+" in s and " " not in s:
        s = " ".join(p for p in s.split("+") if p)
    return " ".join(s.split())


def run_pipeline_from_text(
    text: str,
    morfessor_model_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalize → segment → tag.
    Columns: sent_id, token_id, token, segments, corr_segments, corr_token, tag, lemma, gloss.

    - If lexicon_path/affixes_path are None, packaged defaults are used.
    - Uses corrected outputs (corr_token/corr_segments) for tagging.
    """
    rows = _normalize_and_tokenize_lines(text)
    df = pd.DataFrame(rows, columns=["sent_id", "token_id", "token"])

    # Segment (token API). It may return either:
    #  (A) dict with 'segments', 'corr_segments', 'corr_token'
    #  (B) list of segment strings (legacy/alternate)
    seg_out = segment_tokens(
        df["token"].astype(str).tolist(),
        morfessor_model_path=morfessor_model_path,
        lexicon_path=lexicon_path,     # None => packaged
        affixes_path=affixes_path,     # None => packaged
        allow_train_if_missing=True,
        with_correction=True,          # we want corr_* for tagging
        show_progress=False,
        save_model_path=None,
        pretty_affixes=True,
    )

    if isinstance(seg_out, dict):
        base_segments = [ _coerce_segments_to_spaces(s) for s in seg_out.get("segments", []) ]
        corr_segments = [ _coerce_segments_to_spaces(s) for s in seg_out.get("corr_segments", []) ]
        corr_token    = [ str(t) for t in seg_out.get("corr_token", []) ]
        # If lengths mismatch for any reason, fall back to originals safely
        if len(base_segments) != len(df): base_segments = [""] * len(df)
        if len(corr_segments) != len(df): corr_segments = base_segments
        if len(corr_token)    != len(df): corr_token    = df["token"].astype(str).tolist()
    else:
        # flat list; no correction info provided
        base_segments = [ _coerce_segments_to_spaces(s) for s in seg_out ]
        corr_segments = base_segments[:]
        corr_token    = df["token"].astype(str).tolist()

    df["segments"] = base_segments
    df["corr_segments"] = corr_segments
    df["corr_token"] = corr_token

    # Tagging (+ lemma/gloss) — use corrected token + corrected segments
    tlg = tag_tokens_with_lex(
        df["corr_token"].astype(str).tolist(),
        df["corr_segments"].astype(str).tolist(),
        lexicon_path=lexicon_path,     # None => packaged
        affixes_path=affixes_path,     # None => packaged
    )

    # tag_tokens_with_lex should return list[(tag, lemma, gloss)]
    if not tlg or len(tlg) != len(df):
        tags    = [""] * len(df)
        lemmas  = [""] * len(df)
        glosses = [""] * len(df)
    else:
        tags, lemmas, glosses = zip(*tlg)

    df["tag"]   = list(tags)
    df["lemma"] = [ (l or "") for l in lemmas ]
    df["gloss"] = [ (g or "") for g in glosses ]

    return df[[
        "sent_id", "token_id", "token",
        "segments", "corr_segments", "corr_token",
        "tag", "lemma", "gloss"
    ]]


def run_pipeline_file(
    infile: Path,
    outfile: Path,
    morfessor_model_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> None:
    """Read raw text (one sentence per line), run full pipeline, write TSV."""
    text = Path(infile).read_text(encoding="utf-8")
    df = run_pipeline_from_text(
        text,
        morfessor_model_path=morfessor_model_path,
        lexicon_path=lexicon_path,
        affixes_path=affixes_path,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, sep="\t", index=False)
