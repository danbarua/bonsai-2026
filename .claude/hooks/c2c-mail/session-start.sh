#!/bin/bash
# SessionStart hook (optional third hook, per the c2c mail-awareness
# brief): same notification as user-prompt-submit.sh, so a resumed
# session opens already knowing its mail backlog instead of waiting for
# the first prompt. SessionStart can't block (there's no turn to block
# yet), so this only ever notifies. See lib/c2c_mail.sh and README.md.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/c2c_mail.sh
source "$SCRIPT_DIR/lib/c2c_mail.sh"

c2c_list_unread | c2c_emit_notification "SessionStart"
exit 0
