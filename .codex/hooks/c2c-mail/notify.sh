#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/c2c_mail.sh"

event="${1:-UserPromptSubmit}"
input="$(cat)"
session_id="$(printf '%s' "$input" | c2c_session_id_from_json)"
c2c_list_unread_for "$session_id" 2>/dev/null | c2c_emit_notification "$event"
exit 0
