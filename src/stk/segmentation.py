from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import collections
import pandas as pd
import regex as re

from .normalization import normalize_text, tokenize_keep_clitic
from .utils import load_lexicon, load_affixes, get_affix_sets, _tqdm

# utils
def _heuristic_split(token: str, prefixes: set, suffixes: set, infixes: set) -> List[str]:
    segs = [token]
    for inf in sorted(infixes, key=len, reverse=True):
        if len(segs) == 1 and inf in token[1:-1]:
            L, R = token.split(inf, 1)
            segs = [L, inf, R]
            break
    changed = True
    while changed:
        changed = False
        for pre in sorted(prefixes, key=len, reverse=True):
            if segs and segs[0].startswith(pre) and len(segs[0]) > len(pre) + 1:
                segs = [pre, segs[0][len(pre):]] + segs[1:]
                changed = True
                break
    changed = True
    while changed:
        changed = False
        for suf in sorted(suffixes, key=len, reverse=True):
            if segs and segs[-1].endswith(suf) and len(segs[-1]) > len(suf) + 1:
                segs = segs[:-1] + [segs[-1][:-len(suf)], suf]
                changed = True
                break
    return [s for s in segs if s]

def _format_affixes(parts: List[str], prefixes: set, suffixes: set, infixes: set) -> str:
    """Pretty print: prefix→'pre-', infix→'-infix-', suffix→'-suf'."""
    out: List[str] = []
    for i, s in enumerate(parts):
        if i == 0 and s in prefixes:
            out.append(f"{s}-")
        elif i == len(parts) - 1 and s in suffixes:
            out.append(f"-{s}")
        elif s in infixes:
            out.append(f"-{s}-")
        else:
            out.append(s)
    return " ".join(out)

# ------------------------------ Morfessor ------------------------------
def _train_morfessor_semisupervised(
    tokens: List[str],
    lex_df: Optional[pd.DataFrame],
    prefixes: set, suffixes: set, infixes: set,
    hide_progress: bool = False,
):
    import morfessor
    
    pat = re.compile(r"^[A-Za-z\u00C0-\u024F\u02BC-]+$")

    if isinstance(lex_df, pd.DataFrame) and {"form","freq"} <= set(lex_df.columns):
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

    annotations: Dict[str, List[Tuple[str, ...]]] = {}
    source = (
        lex_df["form"].astype(str).unique()
        if isinstance(lex_df, pd.DataFrame) and "form" in lex_df.columns
        else [w for _, w in sorted(data, key=lambda t: -t[0])[:2000]]
    )
    for w in _tqdm(source, disable=False, desc="Affix annotations"):
        hs = _heuristic_split(w, prefixes, suffixes, infixes)
        if len(hs) > 1:
            annotations[w] = [tuple(hs)]

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

# ------------------------------ SymSpell ------------------------------
def _build_symspell_from_lex_df(lex_df: pd.DataFrame):
    from symspellpy.symspellpy import SymSpell
    import tempfile, os
    ss = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        for _, row in lex_df.iterrows():
            form = str(row.get("form","")).strip()
            if not form: continue
            try:
                f = int(float(row.get("freq","") or 0))
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
    cands.sort(key=lambda x: (-x[1], x[2], x[0]))
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
        if not q: continue
        for s in ss.lookup(q, Verbosity.ALL, max_edit_distance=max_ed):
            if s.term not in lex_set: continue
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

# --- Core API ---
def _format_affixes(parts: List[str], prefixes: set, suffixes: set, infixes: set) -> str:
    """
    Pretty-print morpheme list with affix markup:
      pre-   → prefix
      -suf   → suffix
      -inf-  → infix
    Everything else prints as-is.
    """
    out = []
    for i, p in enumerate(parts):
        raw = str(p)
        if i == 0 and raw in prefixes:
            out.append(f"{raw}-")
        elif i == len(parts) - 1 and raw in suffixes:
            out.append(f"-{raw}")
        elif raw in infixes:
            out.append(f"-{raw}-")
        else:
            out.append(raw)
    # collapse spaces (defensive)
    return " ".join(out).replace("  ", " ").strip()

def _segment_core(
    tokens: List[str],
    morfessor_model_path: Optional[str],
    lexicon_path: Optional[str],
    affixes_path: Optional[str],
    allow_train_if_missing: bool,
    hide_progress: bool,
    save_model_path: Optional[str],
):
    """
    Loads packaged/override resources, trains or loads Morfessor,
    builds lexicon segmentation overrides, and returns base morpheme lists.

    Returns:
      base_parts: List[List[str]]     ← raw morphemes per token (no pretty)
      lex_df: pd.DataFrame | None
      seg_override: Dict[str, str]    ← space-joined override per form
      model: morfessor.BaselineModel
      prefixes, suffixes, infixes: sets
    """
    # -- resources: lexicon --
    lex_any = load_lexicon(lexicon_path) if (lexicon_path or lexicon_path is None) else None
    if isinstance(lex_any, dict):
        lex_df = pd.DataFrame(
            [{"form": k, "freq": v.get("freq", 0), "seg": v.get("seg", "")} for k, v in lex_any.items()]
        )
    else:
        lex_df = lex_any  # may be None

    # -- resources: affixes --
    aff = load_affixes(affixes_path) if (affixes_path or affixes_path is None) else {"affix_types": {}}
    prefixes, suffixes, infixes = get_affix_sets(aff) 

    # -- model: load or train --
    if morfessor_model_path:
        model = _load_morfessor_model(morfessor_model_path)
    else:
        if not allow_train_if_missing:
            raise RuntimeError("Morfessor model required (allow_train_if_missing=False).")
        model = _train_morfessor_semisupervised(tokens, lex_df, prefixes, suffixes, infixes, hide_progress=hide_progress)
        if save_model_path:
            from morfessor import io as mfio
            mfio.MorfessorIO().write_binary_model_file(str(save_model_path), model)

    # -- lexicon segmentation overrides --
    seg_override: Dict[str, str] = {}
    if isinstance(lex_df, pd.DataFrame) and {"form", "seg"} <= set(lex_df.columns):
        for _, row in lex_df.iterrows():
            f = str(row.get("form", ""))
            s = str(row.get("seg", "") or "")
            if f and s.strip():
                seg_override[f] = " ".join(s.replace("+", " ").split())

    # -- base segmentation (no correction) --
    pat_word = re.compile(r"^[A-Za-z\u00C0-\u024F\u02BC-]+$")
    base_parts: List[List[str]] = []
    for tok in _tqdm(tokens, disable=False, desc="Morfessor segment"):
        tok = str(tok)
        if tok in seg_override and seg_override[tok]:
            parts = seg_override[tok].split()
        elif pat_word.match(tok):
            parts, _ = model.viterbi_segment(tok)
        else:
            parts = [tok]
        base_parts.append(parts)

    return base_parts, lex_df, seg_override, model, prefixes, suffixes, infixes


# --- token API
def segment_tokens(
    tokens: List[str],
    morfessor_model_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
    allow_train_if_missing: bool = True,
    with_correction: bool = True,
    hide_progress: bool = False,
    save_model_path: Optional[str] = None,
    pretty_affixes: bool = True,
    return_df: bool = False,     # <- set True to get the exact same columns as segment_file_tsv()
):
    """
    Run the same logic as segment_file_tsv, but in-memory on a list of tokens.

    Returns either:
      - dict with parallel lists: {"segments", "corr_segments", "corr_token"}  (default), or
      - a DataFrame with columns: sent_id, token, segments, corr_segments, corr_token (if return_df=True)
    """
    # Build the same per-token view file API uses
    per_tok = pd.DataFrame({
        "sent_id":  [f"s{1:03d}"] * len(tokens),                # single fabricated sentence by default
        "token_id": list(range(1, len(tokens) + 1)),
        "token":    [str(t) for t in tokens],
    })

    # ---- shared computation (train/load once) ----
    base_parts, lex_df, _seg_override, _model, prefixes, suffixes, infixes = _segment_core(
        tokens=per_tok["token"].astype(str).tolist(),
        morfessor_model_path=morfessor_model_path,
        lexicon_path=lexicon_path,
        affixes_path=affixes_path,
        allow_train_if_missing=allow_train_if_missing,
        hide_progress=hide_progress,
        save_model_path=save_model_path,
    )

    # base string (pretty or plain)
    base_str = [
        _format_affixes(parts, prefixes, suffixes, infixes) if pretty_affixes else " ".join(parts)
        for parts in base_parts
    ]

    # corrected + safe snap (only if we have a lexicon)
    if with_correction and isinstance(lex_df, pd.DataFrame):
        ss = _build_symspell_from_lex_df(lex_df)
        lex_set = set(lex_df["form"].astype(str))
        form2freq = {}
        for _, row in lex_df.iterrows():
            try:
                form2freq[str(row["form"])] = int(float(row.get("freq","") or 0))
            except Exception:
                form2freq[str(row["form"])] = 0

        affix_set = set(prefixes) | set(suffixes) | set(infixes)

        corr_parts: List[List[str]] = []
        for parts in _tqdm(base_parts, disable=hide_progress, desc="Correction"):
            fixed = [_correct_segment(p, ss, lex_set, form2freq, affix_set) for p in parts]
            corr_parts.append(fixed)

        corr_str = [
            _format_affixes(parts, prefixes, suffixes, infixes) if pretty_affixes else " ".join(parts)
            for parts in corr_parts
        ]

        corr_token: List[str] = []
        for tok, parts in zip(per_tok["token"].astype(str).tolist(), corr_parts):
            cand = "".join(parts)
            corr_token.append(_safe_whole_token_snap(tok, cand, ss, lex_set, form2freq))
    else:
        corr_str = base_str[:]
        corr_token = per_tok["token"].astype(str).tolist()

    if return_df:
        out_df = per_tok[["sent_id","token"]].copy()
        out_df["segments"] = base_str
        out_df["corr_segments"] = corr_str
        out_df["corr_token"] = corr_token
        return out_df

    return {
        "segments": base_str,
        "corr_segments": corr_str,
        "corr_token": corr_token,
    }

# ------------------------------ Public: file API ------------------------------
def segment_file_tsv(
    infile: Path,
    outfile: Path,
    morfessor_model_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
    allow_train_if_missing: bool = True,
    with_correction: bool = True,
    hide_progress: bool = False,
    save_model_path: Optional[str] = None,
    pretty_affixes: bool = True,
) -> None:
    """
    Accepts:
      (A) token TSV with 'token'
      (B) normalization TSV with one of: 'corrected_tokens' / 'norm_tokens' / 'tokens'
      (C) raw TXT (one sentence per line)
    Writes: sent_id, token, segments, corr_segments, corr_token
    """
    head = Path(infile).read_text(encoding="utf-8", errors="ignore")[:2048]
    looks_like_tsv = ("\t" in head)

    def _choose_token_col(df):
        for c in ("corrected_tokens", "norm_tokens", "tokens"):
            if c in df.columns:
                return c
        return None

    # ---------- Build per-token view (sent_id, token_id, token) ----------
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
                looks_like_tsv = False
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
        lines = Path(infile).read_text(encoding="utf-8").splitlines()
        rows = []
        for i, line in enumerate(lines, 1):
            norm = normalize_text(line, lowercase=True, keep_diacritics=True)
            toks = tokenize_keep_clitic(norm)
            for j, t in enumerate(toks, 1):
                rows.append((f"s{i:03d}", j, t))
        per_tok = pd.DataFrame(rows, columns=["sent_id", "token_id", "token"])

    tokens = per_tok["token"].astype(str).tolist()

    # ---------- Shared core: load/train model once and produce base parts ----------
    base_parts, lex_df, _seg_override, _model, prefixes, suffixes, infixes = _segment_core(
        tokens=tokens,
        morfessor_model_path=morfessor_model_path,
        lexicon_path=lexicon_path,
        affixes_path=affixes_path,
        allow_train_if_missing=allow_train_if_missing,
        hide_progress=hide_progress,
        save_model_path=save_model_path,
    )

    # Pretty/unpretty helpers (reuse formatting convention across both passes)
    def _join_pretty(parts: List[str]) -> str:
        return _format_affixes(parts, prefixes, suffixes, infixes) if pretty_affixes else " ".join(parts)

    def _unpretty(seg_str: str) -> List[str]:
        # turn "ma- -om- root -an" style back into raw morphemes for snapping
        # (assumes _format_affixes behavior: pre- / -inf- / -suf)
        chunks = []
        for tok in seg_str.split():
            t = tok.strip()
            if not t:
                continue
            # remove surrounding '-' for infixes/suffixes, and trailing '-' for prefixes
            chunks.append(t.strip("-"))
        return [c for c in chunks if c]

    # ---------- Base string segments (no correction) ----------
    base_segments = [_join_pretty(p) for p in base_parts]

    # ---------- Corrected segments + corr_token (SymSpell) ----------
    if with_correction and isinstance(lex_df, pd.DataFrame) and len(lex_df) > 0:
        ss = _build_symspell_from_lex_df(lex_df)
        lex_set = set(lex_df["form"].astype(str))
        form2freq = {
            str(row["form"]): (
                int(float(row.get("freq", "") or 0)) if str(row.get("freq", "")).strip() else 0
            )
            for _, row in lex_df.iterrows()
        }
        affix_set = set(prefixes) | set(suffixes) | set(infixes)

        # Per-segment correction (freq-first), then snap whole token
        corr_parts = []
        corr_token = []
        for tok, parts in zip(tokens, base_parts):
            fixed = [_correct_segment(s, ss, lex_set, form2freq, affix_set) for s in parts]
            corr_parts.append(fixed)
            cand = "".join(fixed)
            snapped = _safe_whole_token_snap(tok, cand, ss, lex_set, form2freq)
            corr_token.append(snapped)

        corr_segments = [_join_pretty(p) for p in corr_parts]
    else:
        corr_segments = base_segments[:]
        corr_token = tokens[:]

    # ---------- Emit file ----------
    out_df = per_tok[["sent_id", "token"]].copy()
    out_df["segments"] = base_segments
    out_df["corr_segments"] = corr_segments
    out_df["corr_token"] = corr_token

    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outfile, sep="\t", index=False)
