---
name: mirror-memory-locally
description: "When writing to auto-memory for Difix3D, also mirror the same change into .claude/memory/ inside the repo (in-repo backup of auto-memory)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 9840db9a-7944-4e31-8888-4963a4b4f638
---

For this project, the auto-memory directory
(`/home/zinyou/.claude/projects/-local-home-zinyou-projects-Difix3D/memory/`)
must be **mirrored** to an in-repo directory at
`/local/home/zinyou/projects/Difix3D/.claude/memory/`. Every create /
update / delete of an auto-memory file in the same turn must be applied
to the corresponding file in `.claude/memory/`. Both trees stay
byte-identical (same `MEMORY.md` index, same per-memory files, same
frontmatter).

**Why:** the user is worried that `${HOME}/.claude` may be renamed,
moved, or wiped — in which case cross-session memory for this project
would silently vanish. The in-repo mirror is the survival copy: it lives
with the code, is git-trackable, and follows the project if the home
dir changes.

**How to apply:**
- On every memory write/update/delete here, immediately make the
  identical edit under `.claude/memory/` in the same turn — same
  filenames, same MEMORY.md index entries, same frontmatter.
- At session start, if the two trees diverge, treat the in-repo mirror
  as the source of truth and rewrite the auto-memory side to match.
- This is project-specific. Other projects don't need this unless the
  same convention is documented in their CLAUDE.md.

Related: [[handoff-docs]] — the same `.claude/` directory also holds
HANDOFF/COMMANDS/OPEN_TASKS and a sibling [[notes]] tree for milestone
summaries.
