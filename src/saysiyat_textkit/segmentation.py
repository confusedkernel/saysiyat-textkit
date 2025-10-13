# src/saysiyat_textkit/segmentation.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import collections
import pandas as pd
import regex as re

# packaged loaders (None => built-in data/*)
from .normalization import normalize_text, tokenize_keep_clitic, load_lexicon
from .config import load_affixes

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _tqdm(iterable, disable=False, desc: str = ""):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, disable=disable, desc=desc)
    except Exception:
        return iterable

def _affix_sets(aff: Dict[str, Any]) -> Tuple[set, set, set]:
    """Build sets of prefixes, suffixes, infixes from affixes.json 'affix_types'."""
    prefixes, suffixes, infixes = set(), set(), set()
    for key, meta in (aff.get("affix_types") or {}).items():
        k = str(key)
        t = str((meta or {}).get("type", "")).lower()
        if   t == "prefix": prefixes.add(k.rstrip("-"))
        elif t == "suffix": suffixes.add(k.lstrip("-"))
        elif t == "infix":  infixes.add(k.strip("-"))
        else:
            # Heuristic if type missing
            if k.startswith("-") and k.endswith("-"): infixes.add(k.strip("-"))
            elif k.endswith("-"):                     prefixes.add(k[:-1])
            elif k.startswith("-"):                   suffixes.add(k[1:])
            else:                                     prefixes.add(k)
    return prefixes, suffixes, infixes

def _heuristic_split(token: str, prefixes: set, suffixes: set, infixes: set) -> List[str]:
    """Affix-first greedy split used for weak supervision and rare fallbacks."""
    segs = [token]
    # single infix split (longest-first)
    for inf in sorted(infixes, key=len, reverse=True):
        if len(segs) == 1 and inf in token[1:-1]:
            L, R = token.split(inf, 1)
            segs = [L, inf, R]
            break
    # greedy prefix
    changed = True
    while changed:
        changed = False
        for pre in sorted(prefixes, key=len, reverse=True):
            if segs and segs[0].startswith(pre) and len(segs[0]) > len(pre) + 1:
                segs = [pre, segs[0][len(pre):]] + segs[1:]
                changed = True
                break
    # greedy suffix
    changed = True
    while changed:
        changed = False
        for suf in sorted(suffixes, key=len, reverse=True):
            if segs and segs[-1].endswith(suf) and len(segs[-1]) > len(suf) + 1:
                segs = segs[:-1] + [segs[-1][:-len(suf)], suf]
                changed = True
                break
    return [s for s in segs if s]

# -----------------------------------------------------------------------------
# Morfessor train/load
# -----------------------------------------------------------------------------
def _train_morfessor_semisupervised(
    tokens: List[str],
    lex_df: Optional[pd.DataFrame],
    prefixes: set, suffixes: set, infixes: set,
    show_progress: bool = True,
):
    import morfessor
    pat = re.compile(r"^[A-Za-z\u00C0-\u024F\u02BC-]+$")

    # Weighted data: lexicon freq if present, else token counts
    if isinstance(lex_df, pd.DataFrame) and {"form", "freq"} <= set(lex_df.columns):
        forms = lex_df["form"].astype(str)
        freqs = pd.to_numeric(lex_df["freq"], errors="coerce").fillna(1).astype(int)
        tally = collections.Counter()
        for w, f in zip(forms, freqs):
            if pat.match(w):
                tally[w] += int(f)
        data = [(c, w) for w, c in tally.items()]
    else:
        cnt = collections.Counter(w for w in tokens if pat.match(str(w)))
        data = [(c, w) for w, c in cnt.items()]

    # Weak-supervision annotations
    annotations: Dict[str, List[Tuple[str, ...]]] = {}
    if isinstance(lex_df, pd.DataFrame) and "form" in lex_df.columns:
        source = lex_df["form"].astype(str).unique()
    else:
        # top frequent tokens if no lexicon
        source = [w for _, w in sorted(data, key=lambda t: -t[0])[:2000]]

    for w in _tqdm(source, disable=not show_progress, desc="Affix annotations"):
        hs = _heuristic_split(w, prefixes, suffixes, infixes)
        if len(hs) > 1:
            annotations[w] = [tuple(hs)]

    # Train
    io = morfessor.MorfessorIO()
    model = morfessor.BaselineModel()
    model.load_data(data)
    if annotations:
        model.set_annotations(annotations)
    model.train_batch()
    return model

def _load_morfessor_model(path: str):
    from morfessor import io as mfio
    io = mfio.MorfessorIO()
    return io.read_binary_model_file(path)

# -----------------------------------------------------------------------------
# SymSpell helpers
# -----------------------------------------------------------------------------
def _build_symspell_from_lex_df(lex_df: pd.DataFrame):
    from symspellpy.symspellpy import SymSpell
    import tempfile, os
    ss = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        for _, row in lex_df.iterrows():
            form = str(row.get("form", "")).strip()
            if not form:
                continue
            try:
                f = int(float(row.get("freq", "") or 0))
            except Exception:
                f = 0
            tmp.write(f"{form}\t{f}\n")
        path = tmp.name
    ss.load_dictionary(path, term_index=0, count_index=1, separator="\t")
    os.remove(path)
    return ss

def _pick_best_freq_then_distance(suggestions, lex_set, form2freq):
    cands = []
    for s in suggestions:
        if s.term in lex_set:
            cands.append((s.term, form2freq.get(s.term, int(getattr(s, "count", 1))), s.distance))
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[1], x[2], x[0]))  # freq desc, dist asc
    return cands[0]

def _correct_segment(seg: str, ss, lex_set, form2freq, affix_set) -> str:
    if seg in lex_set or seg in affix_set or len(seg) > 3:
        return seg
    from symspellpy.symspellpy import Verbosity
    md = 1 if len(seg) <= 4 else 2
    sugs = ss.lookup(seg, Verbosity.ALL, max_edit_distance=md)
    best = _pick_best_freq_then_distance(sugs, lex_set, form2freq)
    return best[0] if best else seg

def _safe_whole_token_snap(orig: str, cand: str, ss, lex_set, form2freq,
                           max_ed=2, min_ratio=10.0, max_dist=2) -> str:
    from symspellpy.symspellpy import Verbosity
    pool = {}
    for q in (orig, cand):
        if not q:
            continue
        for s in ss.lookup(q, Verbosity.ALL, max_edit_distance=max_ed):
            if s.term not in lex_set:
                continue
            freq = form2freq.get(s.term, int(getattr(s, "count", 1)))
            tup = (freq, s.distance)
            prev = pool.get(s.term)
            if prev is None or (tup[1] < prev[1]) or (tup[1] == prev[1] and tup[0] > prev[0]):
                pool[s.term] = tup
    if orig in lex_set:
        pool.setdefault(orig, (form2freq.get(orig, 1), 0))
    if len(orig) <= 4 or not pool:
        return orig
    winner, (w_f, w_d) = sorted(pool.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0]))[0]
    o_f = pool.get(orig, (0, 0))[0]
    return winner if (winner != orig and w_f >= max(1, o_f) * min_ratio and w_d <= max_dist) else orig

# -----------------------------------------------------------------------------
# Public API (token-level)
# -----------------------------------------------------------------------------
def segment_tokens(
    tokens: List[str],
    morfessor_model_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,     # None => packaged
    affixes_path: Optional[str] = None,     # None => packaged
    allow_train_if_missing: bool = True,
    with_correction: bool = True,
    show_progress: bool = False,
    save_model_path: Optional[str] = None,
) -> List[str]:
    """
    Segment a list of tokens. Returns space-joined segments per token.
      1) Use lexicon.seg if available (space or '+' accepted).
      2) Else Morfessor viterbi_segment (model loaded or trained by default).
      3) Optional: SymSpell per-segment correction + safe whole-token snap.
    """
    # Load resources
    lex_any = load_lexicon(lexicon_path) if (lexicon_path or lexicon_path is None) else None
    if isinstance(lex_any, dict):
        lex_df = pd.DataFrame([
            {"form": k, "freq": v.get("freq", 0), "seg": v.get("seg", "")}
            for k, v in lex_any.items()
        ])
    else:
        lex_df = lex_any

    aff = load_affixes(affixes_path) if (affixes_path or affixes_path is None) else {"affix_types": {}}
    prefixes, suffixes, infixes = _affix_sets(aff)
    affix_set = set(prefixes) | set(suffixes) | set(infixes)

    # Morfessor model
    if morfessor_model_path:
        model = _load_morfessor_model(morfessor_model_path)
    else:
        if not allow_train_if_missing:
            raise RuntimeError("Morfessor model required (allow_train_if_missing=False).")
        model = _train_morfessor_semisupervised(tokens, lex_df, prefixes, suffixes, infixes, show_progress=show_progress)
        if save_model_path:
            from morfessor import io as mfio
            mfio.MorfessorIO().write_binary_model_file(str(save_model_path), model)

    # Lexicon overrides
    seg_override: Dict[str, str] = {}
    if isinstance(lex_df, pd.DataFrame) and {"form", "seg"} <= set(lex_df.columns):
        for _, row in lex_df.iterrows():
            f = str(row["form"])
            s = str(row.get("seg", "") or "")
            if s.strip():
                # allow "a b c" or "a+b+c"; store as space-joined
                seg_override[f] = " ".join(s.replace("+", " ").split())

    # Base segmentation
    pat_word = re.compile(r"^[A-Za-z\u00C0-\u024F\u02BC-]+$")
    base_parts: List[List[str]] = []
    for tok in _tqdm(tokens, disable=not show_progress, desc="Morfessor segment"):
        if tok in seg_override and seg_override[tok]:
            parts = seg_override[tok].split()
        elif pat_word.match(tok):
            parts, _ = model.viterbi_segment(tok)
        else:
            parts = [tok]
        base_parts.append(parts)

    # Optional correction
    if with_correction and isinstance(lex_df, pd.DataFrame):
        ss = _build_symspell_from_lex_df(lex_df)
        lex_set = set(lex_df["form"].astype(str))
        form2freq = {}
        for _, row in lex_df.iterrows():
            try:
                form2freq[str(row["form"])] = int(float(row.get("freq", "") or 0))
            except Exception:
                form2freq[str(row["form"])] = 0

        out_segs: List[str] = []
        for tok, parts in _tqdm(list(zip(tokens, base_parts)), disable=not show_progress, desc="Correction"):
            fixed = [_correct_segment(p, ss, lex_set, form2freq, affix_set) for p in parts]
            cand = "".join(fixed)
            snapped = _safe_whole_token_snap(tok, cand, ss, lex_set, form2freq)
            # Return the *corrected parts* as the segmentation; token snap only affects token surface
            out_segs.append(" ".join(fixed))
        return out_segs

    # No correction → join base parts
    return [" ".join(p) for p in base_parts]

# -----------------------------------------------------------------------------
# File API (CLI)
# -----------------------------------------------------------------------------
def segment_file_tsv(
    infile: Path,
    outfile: Path,
    morfessor_model_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,     # None => packaged
    affixes_path: Optional[str] = None,     # None => packaged
    allow_train_if_missing: bool = True,
    with_correction: bool = True,
    show_progress: bool = True,
    save_model_path: Optional[str] = None,
) -> None:
    """
    Accepts:
      (A) per-token TSV with 'token'
      (B) sentence-level TSV with one of: 'corrected_tokens'/'norm_tokens'/'tokens'
      (C) raw TXT (one sentence per line)
    Writes TSV with EXACTLY: sent_id, token, segments, corr_segments, corr_token
    """
    head = Path(infile).read_text(encoding="utf-8", errors="ignore")[:2048]
    looks_like_tsv = ("\t" in head)

    def _choose_token_col(df):
        for c in ("corrected_tokens", "norm_tokens", "tokens"):
            if c in df.columns:
                return c
        return None

    # Build per-token view (sent_id, token_id, token)
    if looks_like_tsv:
        df = pd.read_csv(infile, sep="\t", dtype=str, keep_default_na=False)
        if "token" in df.columns:
            if {"sent_id", "token_id"} <= set(df.columns):
                per_tok = df[["sent_id", "token_id", "token"]].copy()
            else:
                per_tok = pd.DataFrame({
                    "sent_id": [f"s{i+1:03d}" for i in range(len(df))],
                    "token_id": list(range(1, len(df) + 1)),
                    "token": df["token"].tolist(),
                })
        else:
            col = _choose_token_col(df)
            if col is None:
                looks_like_tsv = False  # fall back to raw text parse
            else:
                if "sent_id" not in df.columns:
                    df = df.copy()
                    df["sent_id"] = [f"s{i+1:03d}" for i in range(len(df))]
                rows = []
                for _, row in df.iterrows():
                    sid = row["sent_id"]
                    toks = str(row[col]).split()
                    for j, t in enumerate(toks, 1):
                        rows.append((sid, j, t))
                per_tok = pd.DataFrame(rows, columns=["sent_id", "token_id", "token"])

    if not looks_like_tsv:
        # raw text
        lines = Path(infile).read_text(encoding="utf-8").splitlines()
        rows = []
        for i, line in enumerate(lines, 1):
            norm = normalize_text(line, lowercase=True, keep_diacritics=True)
            toks = tokenize_keep_clitic(norm)
            for j, t in enumerate(toks, 1):
                rows.append((f"s{i:03d}", j, t))
        per_tok = pd.DataFrame(rows, columns=["sent_id", "token_id", "token"])

    tokens = per_tok["token"].astype(str).tolist()

    # Run segmentation (base + optional correction)
    # We'll compute both uncorrected and corrected to fill both columns.
    base_only = segment_tokens(
        tokens,
        morfessor_model_path=morfessor_model_path,
        lexicon_path=lexicon_path,
        affixes_path=affixes_path,
        allow_train_if_missing=allow_train_if_missing,
        with_correction=False,
        show_progress=show_progress,
        save_model_path=save_model_path,
    )
    corrected = segment_tokens(
        tokens,
        morfessor_model_path=morfessor_model_path,  # reuse same path/model decision
        lexicon_path=lexicon_path,
        affixes_path=affixes_path,
        allow_train_if_missing=allow_train_if_missing,
        with_correction=True,
        show_progress=False,   # already trained above; correction only
        save_model_path=None,
    )

    # For corr_token we need the snap winner; easiest is to rebuild with snap info.
    # Re-run a light pass to compute corr_token from corrected segments + SymSpell
    corr_token = tokens[:]  # default: original
    try:
        # Only possible if lexicon is available (SymSpell needs it)
        lex_any = load_lexicon(lexicon_path) if (lexicon_path or lexicon_path is None) else None
        if isinstance(lex_any, dict):
            lex_df = pd.DataFrame([
                {"form": k, "freq": v.get("freq", 0), "seg": v.get("seg", "")}
                for k, v in lex_any.items()
            ])
        else:
            lex_df = lex_any

        if isinstance(lex_df, pd.DataFrame):
            ss = _build_symspell_from_lex_df(lex_df)
            lex_set = set(lex_df["form"].astype(str))
            form2freq = {}
            for _, row in lex_df.iterrows():
                try:
                    form2freq[str(row["form"])] = int(float(row.get("freq", "") or 0))
                except Exception:
                    form2freq[str(row["form"])] = 0

            prefixes, suffixes, infixes = _affix_sets(
                load_affixes(affixes_path) if (affixes_path or affixes_path is None) else {"affix_types": {}}
            )
            affix_set = set(prefixes) | set(suffixes) | set(infixes)

            # produce cand from corrected parts, then safe snap
            corr_parts = [c.split() for c in corrected]
            for i, (tok, parts) in enumerate(zip(tokens, corr_parts)):
                cand = "".join(parts)
                corr_token[i] = _safe_whole_token_snap(tok, cand, ss, lex_set, form2freq)
    except Exception:
        pass  # leave corr_token as original if anything goes wrong

    out_df = per_tok[["sent_id", "token"]].copy()
    out_df["segments"] = base_only
    out_df["corr_segments"] = corrected
    out_df["corr_token"] = corr_token

    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outfile, sep="\t", index=False)
