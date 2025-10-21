from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import regex as re

from .utils import load_lexicon, load_affixes, get_affix_sets, get_voice_mappings


def _lexicon_to_maps(lex_any: Any) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Accept dict[str, dict] or a DataFrame. Return (lemma_map, gloss_map, pos_map)
    Keys preserve case since S/s are distinct letters in Saysiyat.
    """
    if isinstance(lex_any, dict):
        lemma_map = {k: str(v.get("lemma", "") or k) for k, v in lex_any.items()}
        gloss_map = {k: str(v.get("gloss", "") or "") for k, v in lex_any.items()}
        pos_map   = {k: str(v.get("pos",   "") or "") for k, v in lex_any.items()}
        return lemma_map, gloss_map, pos_map

    # DataFrame branch
    if isinstance(lex_any, pd.DataFrame):
        df = lex_any.copy()
        # normalize column names
        cols = {c.lower(): c for c in df.columns}
        form_col  = cols.get("form")  or cols.get("token_corr") or cols.get("token")
        lemma_col = cols.get("lemma")
        gloss_col = cols.get("gloss") or cols.get("zh")
        pos_col   = cols.get("pos") or cols.get("POS")  # Handle both cases

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
            k = str(f)  # Don't lowercase! S/s are distinct in Saysiyat
            lemma_map[k] = str(l) if str(l).strip() else f
            gloss_map[k] = str(g) if str(g).strip() else ""
            pos_map[k]   = str(p) if str(p).strip() else ""
        return lemma_map, gloss_map, pos_map

    # Unknown type
    return {}, {}, {}


def reconstruct_om_infix_stem(tok: str, lemma_map: Dict[str, str]) -> Tuple[str, bool]:
    """
    Reconstruct the original stem from a word with -om- infix.
    
    In Saysiyat, -om- replaces the first vowel after the first consonant.
    E.g., Sombet -> Sebet (S + e + bet, where -om- replaced the 'e')
    Note: S (uppercase) in Saysiyat is a different letter from s (lowercase)
    
    Returns: (reconstructed_stem, was_reconstructed)
    """
    # Don't lowercase - S is a distinct letter in Saysiyat!
    # Search case-insensitively for 'om' pattern
    om_pattern = re.search(r'om', tok, re.IGNORECASE)
    if not om_pattern:
        return tok, False
    
    om_idx = om_pattern.start()
    
    # -om- should come after at least one consonant
    if om_idx < 1:
        return tok, False
    
    # Get the parts: C + om + rest (preserve original case)
    pre_om = tok[:om_idx]  # consonant(s) before -om-
    post_om = tok[om_idx+2:]  # everything after -om-
    
    # Try to find the original form in lexicon by testing possible vowels
    # Saysiyat vowels: a, ae, e, i, o, oe
    vowels = ['ae', 'oe', 'a', 'e', 'i', 'o']  # Try digraphs first, then single vowels
    
    for vowel in vowels:
        candidate = pre_om + vowel + post_om
        # Try both direct lookup and case variations
        if candidate in lemma_map:
            return candidate, True
    
    # If no match found, return the form without -om- but with 'e' as default
    return pre_om + 'e' + post_om, True


def get_lemma_for_om_verb(tok: str, lemma_map: Dict[str, str]) -> str:
    """
    Get the correct lemma for a verb with -om- infix.
    
    Example: Sombet -> Sebet (the lemma form)
    Note: Preserves case - S is distinct from s in Saysiyat
    """
    # First try direct lookup (no case conversion)
    if tok in lemma_map:
        return lemma_map[tok]
    
    # Try reconstructing from -om- infix
    reconstructed, was_found = reconstruct_om_infix_stem(tok, lemma_map)
    
    if was_found and reconstructed in lemma_map:
        return lemma_map[reconstructed]
    
    # Return the reconstructed form even if not in lexicon
    if was_found:
        return reconstructed
    
    # Default: return the token itself
    return tok


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
                 infixes: set,
                 voice_prefix: dict,
                 voice_infix: dict,
                 voice_suffix: dict) -> Tuple[str, str]:
    """
    Return (LABEL, FEATURE).
    LABEL ∈ {PREF, INFX, SUFF, STEM, CLITIC, PUNC}
    FEATURE may encode VOICE and POL when applicable (e.g., "VOICE=AV;POL=I").
    """
    s = seg.strip().lower()
    if seg == "=":
        return "CLITIC", ""
    if re.fullmatch(r"[.,;:!?()\-]+", seg):
        return "PUNC", ""

    if s in prefixes:
        if s in voice_prefix:
            v, p = voice_prefix[s]
            return "PREF", f"VOICE={v};POL={p}"
        return "PREF", ""

    if s in infixes:
        if s in voice_infix:
            v, p = voice_infix[s]
            return "INFX", f"VOICE={v};POL={p}"
        return "INFX", ""

    if s in suffixes:
        if s in voice_suffix:
            v, p = voice_suffix[s]
            return "SUFF", f"VOICE={v};POL={p}"
        return "SUFF", ""

    return "STEM", ""


def tag_tokens_with_lex(
    corr_tokens: List[str],
    segments: Optional[List[str]] = None,
    lexicon_path: Optional[str] = None,
    affixes_path: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    lex_any = load_lexicon(lexicon_path, return_type="dict")
    lemma_map, gloss_map, pos_map = _lexicon_to_maps(lex_any)

    affixes_data = load_affixes(affixes_path)
    prefixes, suffixes, infixes = get_affix_sets(affixes_data)
    voice_prefix, voice_infix, voice_suffix = get_voice_mappings(affixes_data)

    out: List[Tuple[str, str, str]] = []
    for tok in corr_tokens:
        base = (tok or "").strip()
        key = base 

        lex_pos = (pos_map.get(key, "") or "").strip()

        if lex_pos:
            lemma = (lemma_map.get(key) or base)
            gloss = (gloss_map.get(key) or "")
            out.append((lex_pos, lemma, gloss))
            continue

        segs = _heuristic_segment_for_tagging(base, prefixes, suffixes, infixes)

        # Compose gloss parts & decide coarse POS
        gloss_parts: List[str] = []
        has_om_infix = any(s.lower() == "om" for s in segs)
        
        for s in segs:
            lab, feat = _tag_segment(s, prefixes, suffixes, infixes, voice_prefix, voice_infix, voice_suffix)
            if lab == "STEM":
                # For stems in -om- words, get gloss from reconstructed form
                if has_om_infix:
                    reconstructed_stem, _ = reconstruct_om_infix_stem(key, lemma_map)
                    stem_gloss = gloss_map.get(reconstructed_stem, gloss_map.get(s, ""))
                    # If no gloss found, use the raw segment
                    gloss_parts.append(stem_gloss if stem_gloss else s)
                else:
                    stem_gloss = gloss_map.get(s, "")
                    gloss_parts.append(stem_gloss if stem_gloss else s)
            elif lab == "INFX" and s.lower() == "om":
                if "VOICE=" in feat:
                    m = re.search(r"VOICE=([A-Z]+)", feat)
                    if m:
                        gloss_parts.append(f"{s} (VOICE: {m.group(1)})")
                else:
                    gloss_parts.append(s)
            elif "VOICE=" in feat:
                m = re.search(r"VOICE=([A-Z]+)", feat)
                if m:
                    gloss_parts.append(f"{s} (VOICE: {m.group(1)})")
            elif lab in {"PREF", "INFX", "SUFF"}:
                gloss_parts.append(s)

        if base.startswith("="):
            pos_guess = "CLIT"
        elif any(_tag_segment(s, prefixes, suffixes, infixes, voice_prefix, voice_infix, voice_suffix)[0] in {"PREF", "INFX", "SUFF"} for s in segs):
            pos_guess = "V"
        else:
            pos_guess = "X"

        # Get lemma - use reconstruction for -om- verbs
        if has_om_infix and re.search(r'om', key, re.IGNORECASE):
            lemma = get_lemma_for_om_verb(key, lemma_map)
        else:
            lemma = lemma_map.get(key, base)
            
        gloss = "+".join(gloss_parts)
        out.append((pos_guess, lemma, gloss))

    return out


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

    # Load resources dynamically
    lex_any = load_lexicon(lexicon_path, return_type="dict")
    lemma_map, gloss_map, pos_map = _lexicon_to_maps(lex_any)
    
    affixes_data = load_affixes(affixes_path)
    prefixes, suffixes, infixes = get_affix_sets(affixes_data)
    voice_prefix, voice_infix, voice_suffix = get_voice_mappings(affixes_data)

    rows = []
    for sent_id, sub in df.groupby("sent_id", sort=False):
        for tok in sub["corr_token"].astype(str):
            tok_disp = tok
            if not tok or tok.isspace():
                continue

            key = tok
            lex_pos = (pos_map.get(key, "") or "").strip()

            if lex_pos:
                # Word has POS in lexicon - use it directly
                lemma = lemma_map.get(key, tok)
                gloss = gloss_map.get(key, "")
                pos = lex_pos
                
                # Skip segmentation for nouns 
                if lex_pos.upper() in ["N", "NOUN"]:
                    segs = [tok]
                    seg_tags = [f"{tok}:STEM"]
                    focus_str = ""
                else:
                    # For other POS, keep word intact (no segmentation)
                    segs = [tok]
                    seg_tags = [f"{tok}:STEM"]
                    focus_str = ""
            else:
                segs = _heuristic_segment_for_tagging(tok, prefixes, suffixes, infixes)
                seg_tags: List[str] = []
                focus_vals = set()
                gloss_parts: List[str] = []
                
                # Check if this word has -om- infix first
                has_om_infix = any(s.lower() == "om" for s in segs)

                for s in segs:
                    lab, feat = _tag_segment(s, prefixes, suffixes, infixes, voice_prefix, voice_infix, voice_suffix)
                    seg_tags.append(f"{s}:{lab}{('['+feat+']') if feat else ''}")
                    if feat.startswith("VOICE="):
                        m = re.search(r"VOICE=([A-Z]+)", feat)
                        if m:
                            focus_vals.add(m.group(1))
                    
                    g = ""
                    if lab == "STEM":
                        if has_om_infix:
                            reconstructed_stem, _ = reconstruct_om_infix_stem(key, lemma_map)
                            stem_gloss = gloss_map.get(reconstructed_stem, gloss_map.get(s, ""))
                            # If no gloss found, use the raw segment
                            g = stem_gloss if stem_gloss else s
                        else:
                            stem_gloss = gloss_map.get(s, "")
                            g = stem_gloss if stem_gloss else s
                    elif lab == "INFX" and s.lower() == "om":
                        if "VOICE" in feat:
                            m = re.search(r"VOICE=([A-Z]+)", feat)
                            if m:
                                g = f"{s} (VOICE: {m.group(1)})"
                        else:
                            g = s
                    elif "VOICE" in feat:
                        m = re.search(r"VOICE=([A-Z]+)", feat)
                        if m:
                            g = f"{s} (VOICE: {m.group(1)})"
                    elif lab in {"PREF","INFX","SUFF"}:
                        g = s
                    if g:
                        gloss_parts.append(g)

                gloss = "+".join(gloss_parts)
                focus_str = "|".join(sorted(focus_vals)) if focus_vals else ""
                
                if has_om_infix and re.search(r'om', key, re.IGNORECASE):
                    lemma = get_lemma_for_om_verb(key, lemma_map)
                else:
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