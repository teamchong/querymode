# specs/

Specifications define **what QueryMode does** before code is written.

- A spec is a contract. Code must conform to the spec, not the other way around.
- New features start here. Write the spec → review → then build.
- Existing specs are updated when behavior intentionally changes.

## Index

| Spec | Status | What it defines |
|------|--------|----------------|
| [invariants.md](invariants.md) | **Living** | Rules that must never break. Every PR checks against this. |
| [api.md](api.md) | **Living** | Public API surface: DataFrame, SQL, HTTP, PG wire, RPC |
| [query-lifecycle.md](query-lifecycle.md) | **Living** | Input → output for every query path (local, edge, fan-out) |
| [data-formats.md](data-formats.md) | **Living** | Byte-level layout: QMCB, Lance fragment, manifest, footer |
| [full-text-search.md](full-text-search.md) | **Draft** | Full-text search: inverted index, BM25, tokenization, typo tolerance |

## Spec vs Other Docs

| | Specs (`specs/`) | Design Docs (`docs/design/`) | Module READMEs (`src/*/README.md`) |
|---|---|---|---|
| **When written** | Before code | Before code | After code |
| **Answers** | What does it do? | Why this approach? | Where is everything? |
| **Audience** | Anyone implementing or testing | Anyone deciding architecture | Anyone navigating the codebase |
| **Changes when** | Behavior intentionally changes | Decision is revisited | Code moves or restructures |

## Template

```markdown
# Spec: [Feature Name]

Status: **Draft** | **Accepted** | **Implemented**

## User-Facing API
What the user sees. Exact method signatures, SQL syntax, HTTP endpoints.

## Behavior
Given X input, expect Y output. Cover happy path + edge cases.

## Acceptance Criteria
How we know it's done. Measurable. Testable.

## Non-Goals
What this feature explicitly does NOT do.

## Dependencies
What must exist before this can be built.
```
