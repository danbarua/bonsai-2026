#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/c2c_mail.sh"

input="$(cat)"
session_id="$(printf '%s' "$input" | c2c_session_id_from_json)"
mail_files="$(c2c_list_unread_for "$session_id" 2>/dev/null)"
[ -n "$mail_files" ] || exit 0

names="$(printf '%s\n' "$mail_files" | while IFS= read -r path; do basename "$path"; done | paste -sd, -)"
python3 -c 'import json,sys; print(json.dumps({"decision":"block", "reason":sys.argv[1]}))' \
  "unread c2c mail: $names — read and handle it before finishing."
exit 0
