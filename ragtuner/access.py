"""Access control: tenant and principal pre-filtering.

Rules (fail closed):
- A chunk carries an optional tenant and a set of principals. Empty principals means
  visible to everyone in its tenant.
- A request carries an Access(tenant, principal). Tenants must match exactly: a chunk
  owned by a tenant is visible only to requests for that tenant (a request with no tenant
  sees none of them), and a tenant-scoped request never sees chunks without a tenant. A
  chunk with principals is visible only to a request whose principal is in that set; an
  anonymous request never sees it. Access=None is the separate unrestricted path used by
  evaluation, never by serving when the corpus has an ACL.
- Filtering happens BEFORE top-k (pre-filter). Post-filtering would starve top-k: when
  the best matches belong to someone else, the caller would get fewer results or none.

ACLs are applied at chunk time, never stored in the parse cache, because the same bytes
can carry different permissions in different deployments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ragtuner.chunk import Chunk


@dataclass(frozen=True)
class Access:
    tenant: str | None = None
    principal: str | None = None


@dataclass(frozen=True)
class DocACL:
    tenant: str | None = None
    principals: frozenset[str] = frozenset()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DocACL":
        return cls(tenant=d.get("tenant"), principals=frozenset(d.get("principals", [])))


def apply_acl(chunks: list[Chunk], acl_by_doc: dict[str, DocACL]) -> list[Chunk]:
    out = []
    for c in chunks:
        acl = acl_by_doc.get(c.doc_id)
        if acl is None:
            out.append(c)
            continue
        meta = dict(c.metadata)
        meta["tenant"] = acl.tenant
        meta["principals"] = sorted(acl.principals)
        out.append(replace(c, metadata=meta))
    return out


def visible(chunk: Chunk, access: Access) -> bool:
    tenant = chunk.metadata.get("tenant")
    principals = chunk.metadata.get("principals") or []
    # Tenants must match exactly: a tenant-owned chunk is hidden from requests that carry
    # no tenant (or another one), and a tenant-scoped request never sees tenantless chunks.
    if tenant != access.tenant:
        return False
    if principals and access.principal not in principals:
        return False
    return True


def allowed_mask(chunks: Sequence[Chunk], access: Access | None) -> np.ndarray | None:
    """Boolean mask of visible chunks, or None when no filtering was requested."""
    if access is None:
        return None
    return np.fromiter((visible(c, access) for c in chunks), dtype=bool, count=len(chunks))


def load_acl(path: str | Path) -> dict[str, DocACL]:
    """acl.json maps a file name (or doc id) to {"tenant": str, "principals": [str]}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: DocACL.from_dict(v) for k, v in raw.items()}
