# notes/

Per-milestone narrative summaries — one file per significant checkpoint in
a workstream (e.g. "first time tagging finetune converged", "decided to
abandon mv-autoregression", "switched to scene-level holdout").

## Lifecycle

- **Create:** Only when the user explicitly asks (e.g. "save a note about
  this") or confirms a suggestion from Claude. Never create unprompted.
- **Update:** If a note already exists for the current task, Claude may
  update it in-place without re-asking.
- **Name:** `YYYY-MM-DD_short-slug.md` (sortable by date).
- **Audience:** future Claude sessions and the user himself, weeks later.
  Lead with what changed, why, and what to look at next.

Notes differ from [memory/](../memory/) in scope: a memory is a fact /
rule / preference that should influence future decisions; a note is a
narrative record of a decision, an outcome, or a transition point.
