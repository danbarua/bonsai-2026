#!/bin/bash
# UserPromptSubmit hook: notifies (never blocks) when c2c/c2gpt mail is
# waiting, on every prompt, so a running session doesn't need a human
# "check your inbox" relay. Only counts mail addressed to this session (by
# /rename'd name) or unaddressed/broadcast -- see lib/c2c_mail.sh's
# c2c_list_unread_for for the addressing filter and its fail-open rule,
# and README.md for the full design.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/c2c_mail.sh
source "$SCRIPT_DIR/lib/c2c_mail.sh"

INPUT="$(cat)"
session_id="$(echo "$INPUT" | c2c_session_id_from_json)"

c2c_list_unread_for "$session_id" | c2c_emit_notification "UserPromptSubmit"
exit 0
