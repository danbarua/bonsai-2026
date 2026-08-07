#!/bin/bash
# UserPromptSubmit hook: notifies (never blocks) when c2c/c2gpt mail is
# waiting, on every prompt, so a running session doesn't need a human
# "check your inbox" relay. See lib/c2c_mail.sh for the shared unread
# check and notification format, and README.md for the full design.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/c2c_mail.sh
source "$SCRIPT_DIR/lib/c2c_mail.sh"

c2c_list_unread | c2c_emit_notification "UserPromptSubmit"
exit 0
