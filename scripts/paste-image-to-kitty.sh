#!/bin/bash

# Function to find the parent process recursively
find_parent_process() {
    local pid=$1
    while [ "$pid" != 1 ]; do
        parent_pid=$(ps -o ppid= -p "$pid" | xargs)
        process_name=$(ps -o comm= -p "$parent_pid" | xargs)
        if [[ "$process_name" == "kitty" || "$process_name" == "tmux" ]]; then
            echo "$parent_pid"
            return
        fi
        pid=$parent_pid
    done
    echo ""
}

# Get the PID of the process running "python -m llmvm.client"
client_pid=$(pgrep -f "python -m llmvm.client")

if [ -z "$client_pid" ]; then
    echo "The process 'python -m llmvm.client' is not running. Exiting script."
    exit 1
fi

# Find the parent process (either Kitty or tmux)
parent_pid=$(find_parent_process "$client_pid")

if [ -z "$parent_pid" ]; then
    echo "Could not find a Kitty or tmux process running the client. Exiting script."
    exit 1
fi

parent_process_name=$(ps -o comm= -p "$parent_pid" | xargs)

if [[ "$parent_process_name" == "tmux" ]]; then
    # Find the tmux session and window
    tmux_session=$(tmux list-sessions -F "#{session_id}" | head -n 1)
    tmux_window=$(tmux list-windows -t "$tmux_session" -F "#{window_id}" | head -n 1)

    /usr/sbin/screencapture -ci

    # Using 'osascript' to activate the specific Kitty window (assuming single Kitty instance)
    /usr/bin/osascript -e "tell application \"kitty\" to activate"

    # Switch to the tmux window running the process
    tmux select-window -t "$tmux_window"

    # Wait a bit to ensure the window has focus
    sleep 0.1

    /usr/bin/osascript \
      -e 'tell application "System Events"' \
      -e 'key down control ' \
      -e 'keystroke "y"' \
      -e 'key up control ' \
      -e 'keystroke "p"' \
      -e 'end tell'

    # Check if no arguments were passed to the script
    if [ $# -eq 0 ]; then
        # If no arguments, also send Enter
        /usr/bin/osascript -e 'tell application "System Events" to keystroke return'
    fi
elif [[ "$parent_process_name" == "kitty" ]]; then
    /usr/sbin/screencapture -ci

    # Using 'osascript' to activate the specific Kitty window by process ID
    /usr/bin/osascript -e "tell application \"System Events\" to set frontmost of (every process whose unix id is $parent_pid) to true"

    # Wait a bit to ensure the window has focus
    sleep 0.1

    /usr/bin/osascript \
      -e 'tell application "System Events"' \
      -e 'key down control ' \
      -e 'keystroke "y"' \
      -e 'key up control ' \
      -e 'keystroke "p"' \
      -e 'end tell'

    # Check if no arguments were passed to the script
    if [ $# -eq 0 ]; then
        # If no arguments, also send Enter
        /usr/bin/osascript -e 'tell application "System Events" to keystroke return'
    fi
else
    echo "Unexpected parent process: $parent_process_name. Exiting script."
    exit 1
fi


