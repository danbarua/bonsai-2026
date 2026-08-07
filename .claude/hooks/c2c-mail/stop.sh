#!/bin/bash
# Stop hook: blocks the turn from ending while c2c/c2gpt mail is unread,
# so mail doesn't just get silently noticed (via the UserPromptSubmit
# hook) and then ignored. Only blocks on mail addressed to this session
# (by /rename'd name) or unaddressed/broadcast -- see lib/c2c_mail.sh's
# c2c_list_unread_for, and README.md for the full design.
#
# Deliberately not `set -e`: every exit point below is explicit, and
# `set -e` interacting with the grep/test conditions here is easy to get
# subtly wrong (e.g. treating "no match" as a fatal error rather than the
# expected/valid outcome it is for several of these checks).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/c2c_mail.sh
source "$SCRIPT_DIR/lib/c2c_mail.sh"

INPUT="$(cat)"

# REQUIRED loop guard: if this Stop hook already blocked once this cycle,
# always allow the stop now, regardless of mail state -- otherwise mail
# arriving (or still sitting there) while the model is in the middle of
# handling mail could wedge the session in a permanent block loop. Plain
# substring match, not a JSON parse: stop_hook_active is emitted by Claude
# Code itself in a fixed format, not attacker-influenced the way mailbox
# filenames are, so a full parser buys nothing here.
if echo "$INPUT" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then
  exit 0
fi

session_id="$(echo "$INPUT" | c2c_session_id_from_json)"
mail_files="$(c2c_list_unread_for "$session_id")"
if [ -z "$mail_files" ]; then
  exit 0
fi

names=$(echo "$mail_files" | xargs -n1 -I{} basename "{}" | paste -sd, -)
echo "unread c2c mail: $names — read and handle it before finishing." >&2
exit 2
