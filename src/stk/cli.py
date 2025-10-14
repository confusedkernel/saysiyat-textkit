import typer
import pandas as pd
from typing import Optional
from pathlib import Path

from .normalization import normalize_correct_file_to_tsv
from .segmentation import segment_file_tsv
from .tagging import tag_file_tsv

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def normalize(
    infile: Path = typer.Option(..., exists=True, readable=True, help="Raw TXT (one line per record)."),
    outfile: Path = typer.Option(..., help="Output TSV with: orth, norm, tokens, norm_tokens, corrected_tokens, lexicon_seg"),
    lexicon: Optional[Path] = typer.Option(
        None,
        help="Lexicon TSV (form, lemma, freq, seg, gloss). "
             "If omitted, uses the built-in packaged lexicon."
    ),
):
    normalize_correct_file_to_tsv(infile=infile, lexicon_path=lexicon, outfile=outfile)
    typer.echo(f"[normalize] Wrote: {outfile}")


@app.command()
def segment(
    infile: Path = typer.Option(..., exists=True, readable=True, help="Raw TXT (one sentence per line), or TSV from normalize, or per-token TSV."),
    outfile: Path = typer.Option(..., help="Output TSV with: sent_id, token, segments, corr_segments, corr_token"),
    morfessor_model: Optional[Path] = typer.Option(None, help="Optional Morfessor .bin; if omitted we train & cache by fingerprint."),
    lexicon: Optional[Path] = typer.Option(None, help="Lexicon TSV; if omitted, use packaged default."),
    affixes: Optional[Path] = typer.Option(None, help="Affixes JSON; if omitted, use packaged default."),
    save_model: Optional[Path] = typer.Option(None, help="If training, also save model here."),
    no_correction: bool = typer.Option(False, help="Skip SymSpell correction + token snap."),
    no_progress: bool = typer.Option(True, help="Disable progress bars."),
):
    segment_file_tsv(
        infile=infile,
        outfile=outfile,
        morfessor_model_path=str(morfessor_model) if morfessor_model else None,
        lexicon_path=str(lexicon) if lexicon else None,
        affixes_path=str(affixes) if affixes else None,
        allow_train_if_missing=True,
        show_progress=not no_progress,
        with_correction=not no_correction,
        save_model_path=str(save_model) if save_model else None,
    )
    typer.echo(f"[segment] Wrote: {outfile}")


@app.command()
def train_morfessor(
    infile: Path = typer.Option(..., exists=True, help="TSV with a 'token' column (or raw text if --raw-text)."),
    outfile: Path = typer.Option(..., help="Path to write morfessor .bin"),
    lexicon: Optional[Path] = typer.Option(None, help="Optional lexicon TSV (form,lemma,freq,seg)."),
    affixes: Optional[Path] = typer.Option(None, help="Optional affixes JSON."),
    raw_text: bool = typer.Option(False, help="If true, treat infile as raw text, not TSV."),
):
    from .utils import load_lexicon, load_affixes
    from .segmentation import _affix_sets, _train_morfessor_semisupervised
    from regex import re

    aff = load_affixes(str(affixes)) if affixes else {"affix_types": {}}
    prefixes, suffixes, infixes = _affix_sets(aff)
    lex = load_lexicon(str(lexicon), return_type="dataframe") if lexicon else None

    if raw_text:
        text = Path(infile).read_text(encoding="utf-8")
        tokens = re.findall(r"[A-Za-z\u00C0-\u024F\u02BC-]+", text)
    else:
        df = pd.read_csv(infile, sep="\\t", dtype=str, keep_default_na=False)
        tokens = df["token"].tolist()

    model = _train_morfessor_semisupervised(tokens, lex, prefixes, suffixes, infixes)

    from morfessor import io as mfio
    io = mfio.MorfessorIO()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    io.write_binary_model_file(str(outfile), model)
    typer.echo(f"[train_morfessor] wrote: {outfile}")


@app.command()
def tag(
    infile: Path = typer.Option(..., exists=True, readable=True,
                                help="Input TSV with at least a 'token' column."),
    outfile: Path = typer.Option(
        ..., help="Output TSV with 'tag' column added (and lemma/gloss if available)."),
    lexicon: Optional[Path] = typer.Option(
        None, help="Optional override lexicon TSV (form,lemma,freq,seg,pos,gloss)."),
    affixes: Optional[Path] = typer.Option(
        None, help="Optional override affixes JSON."),
):
    """Apply a rule-based/lexicon-aware tagger."""
    tag_file_tsv(
        infile, outfile,
        lexicon_path=str(lexicon) if lexicon else None,
        affixes_path=str(affixes) if affixes else None,
    )
    typer.echo(f"[tag] Wrote: {outfile}")


@app.command()
def pipeline(
    infile: Path = typer.Option(..., exists=True, readable=True,
                                help="Raw TXT file (one sentence per line or free text)."),
    outfile: Path = typer.Option(
        ..., help="Output tagged TSV (sent_id, token_id, token, segments, tag, lemma, gloss)."),
    morfessor_model: Optional[Path] = typer.Option(
        None, help="Optional Morfessor binary model path."),
    lexicon: Optional[Path] = typer.Option(
        None, help="Optional override lexicon TSV."),
    affixes: Optional[Path] = typer.Option(
        None, help="Optional override affixes JSON."),
):
    """Run normalization -> segmentation -> tagging in one go."""
    from .pipeline import run_pipeline_file
    run_pipeline_file(
        infile, outfile,
        morfessor_model_path=str(morfessor_model) if morfessor_model else None,
        lexicon_path=str(lexicon) if lexicon else None,
        affixes_path=str(affixes) if affixes else None,
    )
    typer.echo(f"[pipeline] Wrote: {outfile}")


def main():
    app()
