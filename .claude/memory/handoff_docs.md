---
name: handoff-docs
description: "Difix3D has session-handoff docs at .claude/{HANDOFF,COMMANDS,OPEN_TASKS}.md — read them at the start of every chat"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9840db9a-7944-4e31-8888-4963a4b4f638
---

This project maintains its own cross-session handoff under
`/local/home/zinyou/projects/Difix3D/.claude/`:

- `HANDOFF.md` — current goal, active runs, new/changed files this session,
  carry-over context from previous sessions.
- `COMMANDS.md` — copy-pasteable commands for the active workstream
  (tmux attach, dataset regeneration, training relaunch, sanity checks).
- `OPEN_TASKS.md` — checklist of next steps for the active workstream
  plus deferred items from prior workstreams.

**Why:** the user explicitly relies on these to bootstrap new sessions
("update necessary long-term memory in .claude/ folder to help the
following new chat session"). They are updated at the end of significant
turns. The older `codex/` variant of the same docs was deleted; only the
`.claude/` versions are current.

**How to apply:** at the start of any Difix3D session, read all three
before searching the code or asking clarifying questions — the active
training run, tmux session name, output dir, wandb run, and current
task are documented there. At the end of a substantive session, refresh
them (especially when a new dataset/launcher is added or a long-running
job is kicked off).

Related: [[project-venv]] for the env caveat the launchers depend on.
