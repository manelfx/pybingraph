# Git

These working rules apply to review checkpoints, commits and pushes. See
[AGENTS.md](../AGENTS.md) for the instruction entry point and
[Releases](releases.md) for release branches and tags.

## Checkpoints and Git

- Before presenting any code changes for review, create a named stash checkpoint
  and immediately re-apply it so the working tree remains available:

  ```sh
  git stash push -m "codex checkpoint: <description>"
  git stash apply stash@{0}
  ```

- Do not include untracked files in checkpoints unless the user explicitly asks.
- Never use destructive Git commands such as `git reset --hard` or
  `git checkout --` without explicit user approval.
- After a successful push, clear the checkpoint stack with `git stash clear`,
  unless the user asks to preserve a stash entry. The stash stack is shared:
  inspect it first and never remove user-created entries.
- Before applying, dropping, or clearing a stash, verify that the relevant
  entry is a Codex checkpoint. Do not assume `stash@{0}` belongs to Codex.
- Treat unexpected worktree changes as user changes. Do not revert or remove
  them unless explicitly asked.

## Commits

- Use a concise subject line, followed by a blank line and an explanatory body
  when a body is useful.
- Keep every commit-message line at or below 80 columns.
- Use real line breaks in commit messages, never literal `\\n` text.
- Inspect the final commit message with `git log -1 --format=fuller` before
  pushing when the message has been amended or rewritten.
