from stk.normalization import normalize_and_tokenize
from stk.segmentation import segment_tokens
from stk.tagging import tag_tokens

def test_roundtrip():
    txt = "hiza korkoring ma'rem"
    toks = normalize_and_tokenize(txt)
    segs = segment_tokens(toks)
    tags = tag_tokens(toks, segs)
    assert len(toks) == len(segs) == len(tags)
