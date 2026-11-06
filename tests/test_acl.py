"""ACL leak test on a two-principal, two-tenant corpus.

The secrets are written to match the probe query better than anything the caller may
see, so a post-filter implementation would either leak or return nothing at small k.
"""

import json
from pathlib import Path

import pytest

from ragtuner.access import Access
from ragtuner.config import RagConfig
from ragtuner.pipeline import build_index
from ragtuner.run import load_corpus

DOCS = {
    "alice_secret.md": "# Payroll\n\nQuarterly payroll bonus pool approved: 410000 dollars for the platform team payroll bonus.",
    "bob_secret.md": "# Payroll\n\nPayroll bonus payroll bonus payroll bonus pool approved: 990000 dollars, the larger payroll bonus.",
    "public.md": "# Handbook\n\nThe payroll calendar runs monthly. Bonus questions go to the people team.",
    "other_tenant.md": "# Payroll\n\nPayroll bonus pool payroll bonus approved for tenant B: 123456 dollars payroll bonus.",
}
ACL = {
    "alice_secret.md": {"tenant": "a", "principals": ["alice"]},
    "bob_secret.md": {"tenant": "a", "principals": ["bob"]},
    "public.md": {"tenant": "a", "principals": []},
    "other_tenant.md": {"tenant": "b", "principals": []},
}
QUERY = "payroll bonus pool approved dollars"


@pytest.fixture
def corpus(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    for name, text in DOCS.items():
        (tmp_path / "docs" / name).write_text(text, encoding="utf-8")
    (tmp_path / "acl.json").write_text(json.dumps(ACL))
    return load_corpus("acl", tmp_path)


def _owner(doc_id: str) -> str:
    return doc_id.rsplit("-", 1)[0]


MODES = [("sparse", None), ("dense", "hash-64"), ("hybrid", "hash-64"), ("better-rag", None)]


@pytest.mark.parametrize("retrieval,embedder", MODES)
def test_no_cross_principal_or_cross_tenant_leak_at_any_k(corpus, tmp_path, retrieval, embedder):
    cfg = RagConfig(parser="plaintext", chunker="structure", chunk_params={"target_words": 50, "max_words": 100},
                    embedder=embedder, retrieval=retrieval, top_k=1, candidates=10)
    index, _ = build_index(corpus, cfg, cache_root=tmp_path / "cache")
    n = len(index.chunks)
    visible_to = {
        Access("a", "alice"): {"alice_secret", "public"},
        Access("a", "bob"): {"bob_secret", "public"},
        Access("a", None): {"public"},
        Access("b", "alice"): {"other_tenant"},
    }
    for access, allowed in visible_to.items():
        for k in range(1, n + 1):
            hits = index.retrieve(QUERY, k=k, candidates=max(k, 2), access=access)
            owners = {_owner(h.doc_id) for h in hits}
            assert owners <= allowed, f"{retrieval} k={k} {access}: leaked {owners - allowed}"
        # pre-filter, not post-filter: at k=1 the caller still gets a result from what they may see
        top = index.retrieve(QUERY, k=1, candidates=2, access=access)
        assert top and _owner(top[0].doc_id) in allowed, f"{retrieval} {access}: top-1 starved"


def test_unrestricted_retrieval_still_sees_everything(corpus, tmp_path):
    cfg = RagConfig(parser="plaintext", chunker="structure", chunk_params={"target_words": 50, "max_words": 100},
                    embedder=None, retrieval="sparse", top_k=10)
    index, _ = build_index(corpus, cfg, cache_root=tmp_path / "cache")
    owners = {_owner(h.doc_id) for h in index.retrieve(QUERY, k=10)}
    assert owners == {"alice_secret", "bob_secret", "public", "other_tenant"}
    # ACL metadata is attached at chunk time, not in the parse cache
    assert all("tenant" in c.metadata for c in index.chunks)


@pytest.mark.parametrize("retrieval,embedder", MODES)
def test_requests_without_tenant_see_no_tenant_owned_chunks(corpus, tmp_path, retrieval, embedder):
    cfg = RagConfig(parser="plaintext", chunker="structure", chunk_params={"target_words": 50, "max_words": 100},
                    embedder=embedder, retrieval=retrieval, top_k=1, candidates=10)
    index, _ = build_index(corpus, cfg, cache_root=tmp_path / "cache")
    for access in (Access(None, None), Access(None, "alice"), Access(None, "bob")):
        assert index.retrieve(QUERY, k=len(index.chunks), access=access) == [], f"{retrieval} {access} crossed tenants"
