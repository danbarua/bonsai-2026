#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/c2c_mail.sh"

input="$(cat)"
[ -n "$(c2c_list_unread 2>/dev/null)" ] || exit 0
session_id="$(printf '%s' "$input" | c2c_session_id_from_json)"
[ -n "$session_id" ] || exit 0

state_dir="${C2C_MAIL_ANNOUNCED_DIR:-${TMPDIR:-/tmp}/codex-c2c-mail-announced}"
mkdir -p "$state_dir" 2>/dev/null || exit 0
safe_id="$(printf '%s' "$session_id" | tr -c 'A-Za-z0-9_.-' '_')"
state_file="$state_dir/$safe_id"
[ -f "$state_file" ] || : > "$state_file"

waiting="$(c2c_list_unread_for "$session_id" 2>/dev/null | while IFS= read -r path; do basename "$path"; done | sort -u)"
[ -n "$waiting" ] || exit 0
new="$(comm -23 <(printf '%s\n' "$waiting") <(sort -u "$state_file" 2>/dev/null))"
[ -n "$new" ] || exit 0
printf '%s\n' "$new" >> "$state_file"
printf '%s\n' "$new" | c2c_emit_notification "PostToolUse"
exit 0
