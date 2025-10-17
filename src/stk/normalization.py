from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import unicodedata as ud
import regex as re

from .utils import load_lexicon

def normalize_text(
    s: str,
    lowercase: bool = True,
    keep_diacritics: bool = True,
    apostrophes=("'", "’", "ʼ", "ꞌ", "`"),
    hyphens=("-", "–", "—"),
    glottal: str = "'",     # U+02BC (default unified apostrophe)
    # chars to NEVER downcase (both cases preserved)
    preserve_case_chars=("S",),
) -> str:
    # 1) Unicode compose
    s = ud.normalize("NFC", s)

    # 2) Unify apostrophes and dashes
    for a in apostrophes:
        s = s.replace(a, glottal)
    for h in hyphens[1:]:
        s = s.replace(h, hyphens[0])

    # 3) Lowercase everything EXCEPT preserved chars
    if lowercase:
        preserve = set(preserve_case_chars)
        out_chars = []
        for ch in s:
            if ch in preserve or ch.upper() in preserve:
                out_chars.append(ch)
            else:
                out_chars.append(ch.lower())
        s = "".join(out_chars)

    # 4) Optionally strip diacritics
    if not keep_diacritics:
        s = ''.join(
            ch for ch in ud.normalize('NFD', s)
            if ud.category(ch) != 'Mn'
        )
        s = ud.normalize('NFC', s)

    # 5) Whitespace tidy
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Segmentation rule:
# - keep '=' as its own token (so clitic boundary is explicit)
# - allow letters and U+02BC (') as part of words
TOKEN_RE_KEEP_CLITIC = re.compile(
    r"[A-Za-z\u00C0-\u024F\u02BC]+|=|[0-9]+|[-]+|[^\s]"
)

def tokenize_keep_clitic(s: str) -> List[str]:
    return [m.group(0) for m in TOKEN_RE_KEEP_CLITIC.finditer(s)]


tokenize = tokenize_keep_clitic


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter (newline or after .!?;:)."""
    text = text.replace("\r\n", "\n").strip()
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    return [p for p in parts if p]



def build_symspell_from_lexicon(lexicon: Dict[str, Dict[str, Any]],
                                max_edit_distance: int = 2,
                                prefix_length: int = 7):
    try:
        from symspellpy.symspellpy import SymSpell
    except Exception as e:
        raise RuntimeError(
            "symspellpy is required for build_symspell_from_lexicon(). "
            "Add `symspellpy` to your environment."
        ) from e

    import tempfile
    import os
    ss = SymSpell(max_dictionary_edit_distance=max_edit_distance,
                  prefix_length=prefix_length)

    # write a tiny temp dictionary: form \t freq
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
        for form, rec in lexicon.items():
            # empty freq becomes 0 in the file; SymSpell treats as low frequency
            tmp.write(f"{form}\t{rec.get('freq', 0) or 0}\n")
        tmp_path = tmp.name

    ss.load_dictionary(tmp_path, term_index=0, count_index=1, separator="\t")
    os.remove(tmp_path)
    return ss


def rerank_candidates(tokens: List[str],
                      i: int,
                      symspell,
                      lexicon: Dict[str, Dict[str, Any]],
                      kenlm_model=None,
                      ft_model=None,
                      alpha_ft: float = 0.2) -> str:
    """
    Return the best spelling for tokens[i] using SymSpell candidates
    with optional KenLM / FastText context (both optional).
    If the original is in-lexicon, keep it.
    """
    original = tokens[i]
    # do not touch separators / non-words
    if original == "=" or not re.match(r"^[A-Za-z\u00C0-\u024F\u02BC-]+$", original):
        return original

    if original in lexicon:
        return original

    # collect close candidates incl. the original
    from symspellpy.symspellpy import Verbosity
    cands = [original]
    for sug in symspell.lookup(original, Verbosity.CLOSEST, max_edit_distance=2):
        if sug.term not in cands:
            cands.append(sug.term)

    if not cands:
        return original

    best, best_score = original, float("-inf")
    for cand in cands:
        tmp = tokens[:]
        tmp[i] = cand
        score = 0.0

        if ft_model is not None and hasattr(ft_model, "wv"):
            try:
                vec_c = ft_model.wv[cand] if cand in ft_model.wv else None
                if vec_c is not None:
                    ctx = []
                    if i-1 >= 0 and tokens[i-1] != "=" and tokens[i-1] in ft_model.wv:
                        ctx.append(ft_model.wv[tokens[i-1]])
                    if i+1 < len(tokens) and tokens[i+1] != "=" and tokens[i+1] in ft_model.wv:
                        ctx.append(ft_model.wv[tokens[i+1]])
                    if ctx:
                        import numpy as np
                        ctx_vec = np.mean(ctx, axis=0)
                        denom = (np.linalg.norm(vec_c) *
                                 np.linalg.norm(ctx_vec) + 1e-8)
                        sim = float(vec_c.dot(ctx_vec) / denom)
                        score += alpha_ft * sim
            except Exception:
                pass

        if score > best_score:
            best, best_score = cand, score

    return best


def normalize_tokenize_align_correct(
    lines: List[str],
    lexicon: Dict[str, Dict[str, Any]],
    kenlm_model=None,
    ft_model=None
) -> List[Dict[str, Any]]:
    """
    For each raw line:
      - normalize (rich normalize_text)
      - tokenize with '=' kept as its own token
      - SymSpell correction (closest) with dictionary guard
      - collect lexicon-provided segment suggestions (if any)
    Returns a list of dicts with orth/norm/tokens/norm_tokens/corrected_tokens/lexicon_seg.
    """
    symspell = build_symspell_from_lexicon(lexicon)
    outputs: List[Dict[str, Any]] = []

    for line in lines:
        orth = line.rstrip("\n")
        norm = normalize_text(orth, lowercase=True, keep_diacritics=True)
        tokens = [m.group(0) for m in re.finditer(r"[A-Za-z\u00C0-\u024F\u02BC=]+|[0-9]+|[-]+|[^\s]", orth)]
        norm_tokens = tokenize_keep_clitic(norm)

        corrected = norm_tokens[:]
        for i in range(len(corrected)):
            corrected[i] = rerank_candidates(
                corrected, i, symspell, lexicon,
                kenlm_model=kenlm_model, ft_model=ft_model
            )

        lex_seg = [lexicon.get(t, {}).get("seg", "") for t in corrected]

        outputs.append({
            "orth": orth,
            "norm": norm,
            "tokens": tokens,
            "norm_tokens": norm_tokens,
            "corrected_tokens": corrected,
            "lexicon_seg": lex_seg
        })
    return outputs


def normalize_correct_file_to_tsv(
    infile: Path,
    lexicon_path: Path | None,   # None: use packaged default
    outfile: Path,
) -> None:
    lines = Path(infile).read_text(encoding="utf-8").splitlines()
    lexicon = load_lexicon(lexicon_path, return_type="dict")
    out = normalize_tokenize_align_correct(
        lines, lexicon, kenlm_model=None, ft_model=None)

    rows = []
    for rec in out:
        rows.append({
            "orth": rec["orth"],
            "norm": rec["norm"],
            "tokens": " ".join(rec["tokens"]),
            "norm_tokens": " ".join(rec["norm_tokens"]),
            "corrected_tokens": " ".join(rec["corrected_tokens"]),
            "lexicon_seg": " | ".join(s if s else "_" for s in rec["lexicon_seg"]),
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, sep="\t", index=False)
