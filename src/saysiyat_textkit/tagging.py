from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import regex as re

# Reuse your packaged loaders
from .normalization import load_lexicon  # supports None => built-in data/lexicon.tsv
from .config import load_affixes         # supports None => built-in data/affixes.json

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
    segments: Optional[List[str]] = None,      # optional/ignored; tagging relies on corr_token
    lexicon_path: Optional[str] = None,        # None => packaged lexicon
    affixes_path: Optional[str] = None,        # None => packaged affixes
) -> List[Tuple[str, str, str]]:
    """
    Notebook-parity tagger:
      - look up lemma/gloss/POS by corrected token (corr_token) from lexicon
      - otherwise fall back to heuristic affix-based split to infer coarse POS and compose gloss
    Returns a list of (pos, lemma, gloss) aligned with corr_tokens.
    """
    # Load lexicon (must include columns: form, lemma, pos, gloss)
    # load_lexicon returns dict[str, {lemma, freq, seg, pos, gloss}] in our package
    ldict: Dict[str, Dict[str, Any]] = load_lexicon(lexicon_path)
    # Normalize keys to lowercase for lookup robustness
    lemma_map = {k.lower(): v.get("lemma") or k for k, v in ldict.items()}
    gloss_map = {k.lower(): v.get("gloss", "") for k, v in ldict.items()}
    pos_map   = {k.lower(): v.get("pos", "")   for k, v in ldict.items()}

    # Affixes
    prefixes, suffixes, infixes = _affix_sets_from_json(load_affixes(affixes_path))

    out: List[Tuple[str, str, str]] = []
    for tok in corr_tokens:
        base = tok or ""
        base_lc = base.lower()

        # 1) Dictionary direct hit → use it
        if base_lc in pos_map and str(pos_map[base_lc]).strip():
            pos   = str(pos_map[base_lc]) or ""
            lemma = str(lemma_map.get(base_lc, base)) or base
            gloss = str(gloss_map.get(base_lc, "")) or ""
            out.append((pos, lemma, gloss))
            continue

        # 2) Heuristic segmentation → compose features & gloss
        segs = _heuristic_segment_for_tagging(base, prefixes, suffixes, infixes)
        pos_guess = ""  # X, V, CLIT, etc.
        focus_vals = set()
        gloss_parts: List[str] = []

        for s in segs:
            lab, feat = _tag_segment(s, prefixes, suffixes, infixes)

            # collect focus values from features
            if feat.startswith("VOICE="):
                m = re.search(r"VOICE=([A-Z]+)", feat)
                if m:
                    focus_vals.add(m.group(1))

            # gloss composition
            g = ""
            if lab == "STEM":
                g = gloss_map.get(s.lower(), s)
            elif "VOICE" in feat:
                m = re.search(r"VOICE=([A-Z]+)", feat)
                if m:
                    g = f"{s} (VOICE: {m.group(1)})"
            elif lab in {"PREF", "INFX", "SUFF"}:
                g = s
            if g:
                gloss_parts.append(g)

        # POS fallback
        if not pos_guess:
            if base.startswith("="):
                pos_guess = "CLIT"
            elif any(_tag_segment(s, prefixes, suffixes, infixes)[0] in {"PREF","INFX","SUFF"} for s in segs):
                pos_guess = "V"
            else:
                pos_guess = "X"

        lemma = lemma_map.get(base_lc, base)
        gloss = "＋".join(gloss_parts)
        out.append((pos_guess, lemma, gloss))

    return out

# ---------------------------------------------------------------------
# File-level API (CLI helper) — notebook-parity I/O
# ---------------------------------------------------------------------
def tag_file_tsv(
    infile: Path,
    outfile: Path,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> None:
    """
    Read a TSV (normally your segmentation output) and write a tagged table:
      input requires at least: corr_token  (sent_id optional; fabricated if missing)
      output columns: sent_id, token_orig, token_corr, segs, seg_tags, focus, lemma, pos, gloss
    """
    df = pd.read_csv(infile, sep="\t", dtype=str, keep_default_na=False)
    if "corr_token" not in df.columns:
        raise ValueError(f"{infile} must have a 'corr_token' column. Got: {list(df.columns)}")

    if "sent_id" not in df.columns:
        df = df.copy()
        df["sent_id"] = [f"s{1 + i // 50:03d}" for i in range(len(df))]  # naive sentence grouping

    # Load resources
    ldict: Dict[str, Dict[str, Any]] = load_lexicon(lexicon_path)
    prefixes, suffixes, infixes = _affix_sets_from_json(load_affixes(affixes_path))

    # Build lex maps (lowercased keys)
    lemma_map = {k.lower(): v.get("lemma") or k for k, v in ldict.items()}
    gloss_map = {k.lower(): v.get("gloss", "") for k, v in ldict.items()}
    pos_map   = {k.lower(): v.get("pos", "")   for k, v in ldict.items()}

    rows = []
    for sent_id, sub in df.groupby("sent_id", sort=False):
        for tok in sub["corr_token"].astype(str):
            tok_disp = tok
            if not tok or tok.isspace():
                continue
            base_lc = tok.lower()

            # Dictionary direct hit
            if base_lc in pos_map and str(pos_map[base_lc]).strip():
                lemma = lemma_map.get(base_lc, tok)
                gloss = gloss_map.get(base_lc, "")
                pos   = pos_map.get(base_lc, "")
                segs = [tok]
                seg_tags = [f"{tok}:STEM"]
                focus_str = ""
            else:
                # Heuristic segmentation for tagging-time analysis
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
                lemma = lemma_map.get(base_lc, tok)

                # Coarse POS fallback
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
