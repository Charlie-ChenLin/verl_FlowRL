logdir=./logs
mkdir -p "$logdir"
wname="$(tmux display-message -p '#{window_name}')"
safe="$(printf '%s' "$wname" | tr -cs '[:alnum:]._-' '_')"
logfile="$logdir/${safe}.log"
echo "[tmux] tailing to $logfile"

# 对当前窗口的所有 pane 开启 pipe（无 -o，强制覆盖）
for p in $(tmux list-panes -F '#{pane_id}'); do
  tmux pipe-pane -t "$p" "sh -c '
    if command -v stdbuf >/dev/null 2>&1; then S=\"stdbuf -oL -eL\"; else S=cat; fi
    if command -v ts >/dev/null 2>&1; then
      exec $S ts \"[%F %T]\" | tee -a \"$logfile\" >/dev/null
    else
      exec $S tee -a \"$logfile\" >/dev/null
    fi
  '"
done

# tmux new-window -n flowrl-raw-verl070dev0 -c "#{pane_current_path}"
# tmux select-window -t 0:1

