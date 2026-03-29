# Design Documents

Architecture decisions, strategic direction, and system design for QueryMode.

## Documents

| Document | Status | Summary |
|----------|--------|---------|
| [Agentic Infrastructure](agentic-infrastructure.md) | Draft | QueryMode as the query layer for AI agents. Source→Preprocess→Query→Context pipeline |

## How to Use These

- **Before building**: Check if there's a design doc for the area you're working on
- **New features**: Write a design doc BEFORE writing code if the feature touches multiple modules
- **Design docs are NOT code comments** — they explain WHY, not WHAT

## Template

```markdown
# Design: [Feature Name]

Status: **Draft** | **Accepted** | **Implemented** | **Deprecated**

## Problem
What problem does this solve? Why now?

## Approach
How does it work? What are the key decisions?

## Alternatives Considered
What else did you consider and why did you reject it?

## Open Questions
What's unresolved?
```
