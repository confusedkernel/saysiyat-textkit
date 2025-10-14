from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import regex as re

# Reuse your packaged loaders
from .normalization import load_lexicon
from .config import load_affixes       # supports None => built-in data/affixes.json

# ---------------------------------------------------------------------
# Affix inventory & helpers (VOICE table from your earlier spec)
# ---------------------------------------------------------------------
# Minimal, notebook-matching sets (you can also derive from affixes.json if you prefer)
PREFIXES_DEFAULT = {"m", "ma", "si"}                 # AF(I): m-, ma- ; IF(I): si-
INFIXES_DEFAULT  = {"om"}                            # AF(I): -om-
SUFFIXES_DEFAULT = {"en", "an", "i", "ani"}          # PF(I): -en ; LOC(I): -an ; PF(II): -i ; IF(II): -ani

VOICE_PREFIX = {"m": ("AF", "I"), "ma": ("AF", "I"), "si": ("IF", "I")}
VOICE_INFIX  = {"om": ("AF", "I")}
VOICE_SUFFIX = {"en": ("PF", "I"), "an": ("LOC", "I"), "i": ("PF", "II"), "ani": ("IF", "II")}

DERIV_PREFIX = set()  # e.g. {"ka","pa"} if you later want derivational-only labels

def _affix_sets_from_json(aff: Dict[str, Any]) -> tuple[set, set, set]:
    """Optional: read sets from affixes.json if present; else fall back to DEFAULTs."""
    if not aff or "affix_types" not in aff:
        return set(PREFIXES_DEFAULT), set(SUFFIXES_DEFAULT), set(INFIXES_DEFAULT)

    prefixes, suffixes, infixes = set(), set(), set()
    for k, meta in aff["affix_types"].items():
        t = str((meta or {}).get("type", "")).lower()
        s = str(k)
        if t == "prefix":
            prefixes.add(s.rstrip("-"))
        elif t == "suffix":
            suffixes.add(s.lstrip("-"))
        elif t == "infix":
            infixes.add(s.strip("-"))
        else:
            if s.startswith("-") and s.endswith("-"): infixes.add(s.strip("-"))
            elif s.endswith("-"):                     prefixes.add(s[:-1])
            elif s.startswith("-"):                   suffixes.add(s[1:])
            else:                                     prefixes.add(s)
    # Always make sure our defaults are included
    prefixes |= PREFIXES_DEFAULT
    suffixes |= SUFFIXES_DEFAULT
    infixes  |= INFIXES_DEFAULT
    return prefixes, suffixes, infixes

def _lexicon_to_maps(lex_any: Any) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Accept dict[str, dict] or a DataFrame. Return (lemma_map, gloss_map, pos_map)
    with LOWERCASED keys.
    """
    if isinstance(lex_any, dict):
        lemma_map = {k.lower(): str(v.get("lemma", "") or k) for k, v in lex_any.items()}
        gloss_map = {k.lower(): str(v.get("gloss", "") or "") for k, v in lex_any.items()}
        pos_map   = {k.lower(): str(v.get("pos",   "") or "") for k, v in lex_any.items()}
        return lemma_map, gloss_map, pos_map

    # DataFrame branch
    if isinstance(lex_any, pd.DataFrame):
        df = lex_any.copy()
        # normalize column names
        cols = {c.lower(): c for c in df.columns}
        form_col  = cols.get("form")  or cols.get("token_corr") or cols.get("token")
        lemma_col = cols.get("lemma")
        gloss_col = cols.get("gloss") or cols.get("zh")

        def series_or_empty(c):
            return df[c].astype(str) if c in df.columns else pd.Series([""] * len(df))

        if not form_col:
            # nothing usable
            return {}, {}, {}

        form  = series_or_empty(form_col).fillna("").astype(str)
        lemma = series_or_empty(lemma_col).fillna("").astype(str) if lemma_col else pd.Series([""] * len(df))
        gloss = series_or_empty(gloss_col).fillna("").astype(str) if gloss_col else pd.Series([""] * len(df))
        pos   = series_or_empty(pos_col).fillna("").astype(str)   if pos_col   else pd.Series([""] * len(df))

        lemma_map = {}
        gloss_map = {}
        pos_map   = {}
        for f, l, g, p in zip(form, lemma, gloss, pos):
            k = str(f).lower()
            lemma_map[k] = str(l) if str(l).strip() else f
            gloss_map[k] = str(g) if str(g).strip() else ""
            pos_map[k]   = str(p) if str(p).strip() else ""
        return lemma_map, gloss_map, pos_map

    # Unknown type
    return {}, {}, {}


# ---------------------------------------------------------------------
# Tiny heuristic segmenter (notebook parity; tagging-time only)
# ---------------------------------------------------------------------
def _heuristic_segment_for_tagging(tok: str,
                                   prefixes: set,
                                   suffixes: set,
                                   infixes: set) -> List[str]:
    """Return raw morphemes for tagging; no pretty formatting."""
    t = tok.strip().strip("'").lower()
    if not t:
        return [tok]

    # 1) single infix first (x-om-y)
    if "om" in t and len(t) > 3 and "om" in infixes:
        i = t.find("om")
        pre, post = t[:i], t[i+2:]
        parts = []
        if pre:  parts.append(pre)
        parts.append("om")
        if post: parts.append(post)
        return parts

    # 2) greedy longest prefix
    for p in sorted(prefixes, key=len, reverse=True):
        if t.startswith(p) and len(t) > len(p) + 1:
            return [p, t[len(p):]]

    # 3) greedy longest suffix
    for s in sorted(suffixes, key=len, reverse=True):
        if t.endswith(s) and len(t) > len(s) + 1:
            return [t[:-len(s)], s]

    return [tok]

def _tag_segment(seg: str,
                 prefixes: set,
                 suffixes: set,
                 infixes: set) -> Tuple[str, str]:
    """
    Return (LABEL, FEATURE).
    LABEL ∈ {PREF, INFX, SUFF, STEM, CLITIC, PUNC}
    FEATURE may encode VOICE and POL when applicable (e.g., "VOICE=AF;POL=I").
    """
    s = seg.strip().lower()
    if seg == "=":
        return "CLITIC", ""
    if re.fullmatch(r"[.,;:!?()\-]+", seg):
        return "PUNC", ""

    if s in prefixes:
        if s in VOICE_PREFIX:
            v, p = VOICE_PREFIX[s]
            return "PREF", f"VOICE={v};POL={p}"
        if s in DERIV_PREFIX:
            return "PREF", "DER"
        return "PREF", ""

    if s in infixes:
        if s in VOICE_INFIX:
            v, p = VOICE_INFIX[s]
            return "INFX", f"VOICE={v};POL={p}"
        return "INFX", ""

    if s in suffixes:
        if s in VOICE_SUFFIX:
            v, p = VOICE_SUFFIX[s]
            return "SUFF", f"VOICE={v};POL={p}"
        return "SUFF", ""

    return "STEM", ""

# ---------------------------------------------------------------------
# Core API — function returns (pos, lemma, gloss) per corr_token
# ---------------------------------------------------------------------
def tag_tokens_with_lex(
    corr_tokens: List[str],
    segments: Optional[List[str]] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    lex_any = load_lexicon(lexicon_path)
    lemma_map, gloss_map, pos_map = _lexicon_to_maps(lex_any)

    prefixes, suffixes, infixes = _affix_sets_from_json(load_affixes(affixes_path))

    out: List[Tuple[str, str, str]] = []
    for tok in corr_tokens:
        base = (tok or "").strip()
        key = base.lower()

        lex_pos = (pos_map.get(key, "") or "").strip()

        if lex_pos:
            # Word has POS in lexicon - use it directly, no guessing
            lemma = (lemma_map.get(key) or base)
            gloss = (gloss_map.get(key) or "")
            out.append((lex_pos, lemma, gloss))
            continue

        # No POS in lexicon → heuristic analysis
        segs = _heuristic_segment_for_tagging(base, prefixes, suffixes, infixes)

        # Compose gloss parts & decide coarse POS
        gloss_parts: List[str] = []
        for s in segs:
            lab, feat = _tag_segment(s, prefixes, suffixes, infixes)
            if lab == "STEM":
                gloss_parts.append(gloss_map.get(s.lower(), s))
            elif "VOICE=" in feat:
                m = re.search(r"VOICE=([A-Z]+)", feat)
                if m:
                    gloss_parts.append(f"{s} (VOICE: {m.group(1)})")
            elif lab in {"PREF", "INFX", "SUFF"}:
                gloss_parts.append(s)

        if base.startswith("="):
            pos_guess = "CLIT"
        elif any(_tag_segment(s, prefixes, suffixes, infixes)[0] in {"PREF", "INFX", "SUFF"} for s in segs):
            pos_guess = "V"
        else:
            pos_guess = "X"

        lemma = lemma_map.get(key, base)
        gloss = "＋".join(gloss_parts)
        out.append((pos_guess, lemma, gloss))

    return out


# ---------------------------------------------------------------------
# File-level API (CLI helper)
# ---------------------------------------------------------------------
def tag_file_tsv(
    infile: Path,
    outfile: Path,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> None:
    df = pd.read_csv(infile, sep="\t", dtype=str, keep_default_na=False)
    if "corr_token" not in df.columns:
        raise ValueError(f"{infile} must have a 'corr_token' column. Got: {list(df.columns)}")

    if "sent_id" not in df.columns:
        df = df.copy()
        df["sent_id"] = [f"s{1 + i // 50:03d}" for i in range(len(df))]

    # Load resources
    lex_any = load_lexicon(lexicon_path)
    lemma_map, gloss_map, pos_map = _lexicon_to_maps(lex_any)
    prefixes, suffixes, infixes = _affix_sets_from_json(load_affixes(affixes_path))

    rows = []
    for sent_id, sub in df.groupby("sent_id", sort=False):
        for tok in sub["corr_token"].astype(str):
            tok_disp = tok
            if not tok or tok.isspace():
                continue

            key = tok.lower()
            lex_pos = (pos_map.get(key, "") or "").strip()

            if lex_pos:
                # Word has POS in lexicon - use it directly
                lemma = lemma_map.get(key, tok)
                gloss = gloss_map.get(key, "")
                pos = lex_pos
                
                # Skip segmentation for nouns (and any other word with a POS)
                if lex_pos.upper() in ["N", "NOUN"]:
                    # No segmentation for nouns
                    segs = [tok]
                    seg_tags = [f"{tok}:STEM"]
                    focus_str = ""
                else:
                    # For other POS, keep word intact (no segmentation)
                    segs = [tok]
                    seg_tags = [f"{tok}:STEM"]
                    focus_str = ""
            else:
                # No POS in lexicon → heuristic analysis only
                segs = _heuristic_segment_for_tagging(tok, prefixes, suffixes, infixes)
                seg_tags: List[str] = []
                focus_vals = set()
                gloss_parts: List[str] = []

                for s in segs:
                    lab, feat = _tag_segment(s, prefixes, suffixes, infixes)
                    seg_tags.append(f"{s}:{lab}{('['+feat+']') if feat else ''}")
                    if feat.startswith("VOICE="):
                        m = re.search(r"VOICE=([A-Z]+)", feat)
                        if m:
                            focus_vals.add(m.group(1))
                    g = ""
                    if lab == "STEM":
                        g = gloss_map.get(s.lower(), s)
                    elif "VOICE" in feat:
                        m = re.search(r"VOICE=([A-Z]+)", feat)
                        if m:
                            g = f"{s} (VOICE: {m.group(1)})"
                    elif lab in {"PREF","INFX","SUFF"}:
                        g = s
                    if g:
                        gloss_parts.append(g)

                gloss = "＋".join(gloss_parts)
                focus_str = "|".join(sorted(focus_vals)) if focus_vals else ""
                lemma = lemma_map.get(key, tok)

                # POS guessing only when no lexicon POS exists
                if tok.startswith("="):
                    pos = "CLIT"
                elif any(tag.split(":")[1].startswith(("PREF","INFX","SUFF")) for tag in seg_tags):
                    pos = "V"
                else:
                    pos = "X"

            rows.append({
                "sent_id": sent_id,
                "token_orig": tok_disp,
                "token_corr": tok,
                "segs": " ".join(segs),
                "seg_tags": " ".join(seg_tags),
                "focus": focus_str,
                "lemma": lemma,
                "pos": pos,
                "gloss": gloss,
            })

    out = pd.DataFrame(rows)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outfile, sep="\t", index=False)