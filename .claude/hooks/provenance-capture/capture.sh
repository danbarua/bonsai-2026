#!/bin/bash
# Shell wrapper for capture.py -- the outer of two independent fail-open
# layers, following the same belt-and-braces reasoning the c2c-mail hooks
# use (the script's own logic, plus a backstop that cannot fail).
#
# capture.py already exits 0 unconditionally. This wrapper guarantees the
# same thing one level up, for the failures capture.py cannot catch because
# it never got to run: no python3 on PATH, an unreadable script, an
# interpreter that dies on import. A forensic hook must never be able to
# block a session, and "the interpreter was missing" is exactly the kind of
# environment difference that would otherwise surface as a broken tool call
# in somebody else's session.
#
# Stdin is passed straight through. Nothing here inspects it -- parsing
# belongs in one place, and that place is capture.py.

PYTHON="$(command -v python3 2>/dev/null)"
if [ -z "$PYTHON" ]; then
  exit 0
fi

"$PYTHON" "$(dirname "$0")/capture.py" 2>/dev/null

# Unconditional. The exit status of the line above is deliberately ignored:
# a non-zero exit from a PostToolUse hook is shown to the model as an error,
# and a capture failure is not the session's problem.
exit 0
