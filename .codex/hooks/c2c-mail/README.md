# c2c mail hooks for Codex

This is the Codex-native port of `.claude/hooks/c2c-mail`. It watches only
`.claude/code2code/mailbox`, injects filenames (never message bodies), filters
mail by the current Codex thread name, and fails open when identity or mailbox
state cannot be resolved.

The project hook registration is `.codex/hooks.json`. Codex session identity is
resolved from `~/.codex/session_index.jsonl` (`id` to `thread_name`), rather than
Claude Code's `~/.claude/sessions/*.json` registry.

Behavior:

- `SessionStart` and `UserPromptSubmit` report waiting filenames.
- `PostToolUse` reports each filename at most once per Codex session.
- `Stop` returns Codex's JSON block decision while relevant mail remains.
- `PreToolUse` injects the thread name as `instance`/`as` for c2c
  `code2code-send` and `code2code-inbox` calls.

Run the hermetic test suite with:

```bash
bash .codex/hooks/c2c-mail/test.sh
```
