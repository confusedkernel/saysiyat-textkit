from saysiyat_textkit.normalization import normalize_and_tokenize
from saysiyat_textkit.segmentation import segment_tokens
from saysiyat_textkit.tagging import tag_tokens

def test_roundtrip():
    txt = "hiza korkoring ma'rem"
    toks = normalize_and_tokenize(txt)
    segs = segment_tokens(toks)
    tags = tag_tokens(toks, segs)
    assert len(toks) == len(segs) == len(tags)
