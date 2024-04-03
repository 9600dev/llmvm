#!/bin/bash

# Focus Kitty using yabai or another method
# If using yabai and you know the space or window ID, you could switch to it directly
# Example: yabai -m window --focus <window_id>


/usr/sbin/screencapture -c -i -W 

# Using 'osascript' to activate Kitty by name. This brings the window to the foreground
/usr/bin/osascript -e 'tell application "kitty" to activate'

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
