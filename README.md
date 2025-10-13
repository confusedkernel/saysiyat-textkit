# saysiyat-textkit

A small, installable package with both **Python API** and **CLI** for a 3‑stage pipeline:

1. **Normalization & Tokenization** (`normalize`): NFC normalization, clitic‑aware regex tokenization.
2. **Morphological Segmentation** (`segment`): Wraps Morfessor if a model is provided, otherwise uses a tiny fallback heuristic.
3. **Tagging** (`tag`): Minimal demonstrator rule‑based tagger (edit to your spec).
4. **Pipeline**(`pipeline`): Run the entire pipeline in one go.

## Installation

```bash
pip install -e .
```

## CLI usage

```bash
# 1) Normalize+tokenize a raw TXT into TSV with token column
stk normalize --infile raw.txt --outfile tokens.tsv

# 2) Segment tokens (read TSV with 'token' column) into 'segments' column
stk segment --infile tokens.tsv --outfile segments.tsv --morfessor-model path/to/model.bin

# 3) Tag tokens (or segments) into a 'tag' column
stk tag --infile segments.tsv --outfile tagged.tsv
```

## Python API usage

```python
from saysiyat_textkit import normalization, segmentation, tagging

# Normalization+tokenization
tokens = normalization.normalize_and_tokenize("Makat 'sa=miq ta'   kiso.")
print(tokens)

# Segmentation
segmented = segmentation.segment_tokens(tokens, morfessor_model_path=None)  # or a model path
print(segmented)

# Tagging
tags = tagging.tag_tokens(tokens, segmented)
print(list(zip(tokens, segmented, tags)))
```

> The CLI reads/writes TSV files by default. Adjust the code to fit your actual corpus schema.

## Data directory & overrides

The package ships with defaults in `saysiyat_textkit/data/`:

- `lexicon.tsv` (tab-separated: `form lemma freq  seg	pos	gloss`)
- `affixes.json` (e.g., focus markers and separators)

**Override precedence** (highest to lowest):

1. CLI argument (`--lexicon`, `--affixes`) or Python arg (`lexicon_path=...`, `affixes_path=...`)
2. Environment variables: `saysiyat_LEXICON`, `saysiyat_AFFIXES`
3. Built-in package defaults

### Examples

```bash
# Use custom affixes for segmentation + tagging
stk segment --infile tokens.tsv --outfile segments.tsv --affixes /path/affixes.json
stk tag --infile segments.tsv --outfile tagged.tsv --affixes /path/affixes.json

# Or set env vars once (shell)
export saysiyat_LEXICON=/data/lexicon.tsv
export saysiyat_AFFIXES=/data/affixes.json
stk normalize --infile raw.txt --outfile tokens.tsv
```

## Data directory (defaults + overrides)

The package ships with defaults under `saysiyat_textkit/data/`:

- `lexicon.tsv` with columns: `form	lemma	freq	seg	pos	gloss`
- `affixes.json` with keys:
  - `voice_markers` (e.g., `"=en": {"type": "enclitic", "label": "FOC"}`)
  - `affix_types` (e.g., `"ma-": {"type": "prefix", "function": "intransitive"}`)

**Override** at runtime via CLI or Python functions by passing `--lexicon` and/or `--affixes` paths.
If you do nothing, the packaged defaults are used.

### Examples

```bash
# Use custom lexicon & affixes
stk segment --infile tokens.tsv --outfile segments.tsv \
  --lexicon /path/to/lexicon.tsv --affixes /path/to/affixes.json

stk tag --infile segments.tsv --outfile tagged.tsv \
  --lexicon /path/to/lexicon.tsv --affixes /path/to/affixes.json
```

```python
from saysiyat_textkit import segmentation, tagging

segments = segmentation.segment_tokens(tokens,
    morfessor_model_path=None,
    lexicon_path="/path/to/lexicon.tsv",
    affixes_path="/path/to/affixes.json",
)

tags = tagging.tag_tokens(tokens, segments,
    lexicon_path="/path/to/lexicon.tsv",
    affixes_path="/path/to/affixes.json",
)
```
