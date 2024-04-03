#!/bin/bash

# Set the desired output filename
TEMP_PDF=$(mktemp)
FILE="${TEMP_PDF}.pdf"

echo $FILE

# AppleScript to save the active page in Chrome as PDF
osascript <<EOF

tell application "Google Chrome"
    activate
    tell application "System Events"
        keystroke "p" using {command down}
        delay 1
        keystroke tab
        delay 0.1
        keystroke tab
        delay 0.1
        keystroke tab
        delay 0.1
        keystroke tab
        delay 0.1
        keystroke tab
        delay 0.1
        keystroke tab
        delay 0.1
        keystroke tab
        delay 0.1
        keystroke tab
        delay 0.1
        keystroke tab
        delay 0.1
        keystroke return
        delay 0.3 
        keystroke "g" using {shift down, command down}
        delay 0.2 
        keystroke "$FILE"
        delay 0.8 
        keystroke return
        delay 0.2
        keystroke return
    end tell
end tell
EOF


# Your text string to send to Kitty
TEXT="[PdfContent($FILE)]"
echo $TEXT

osascript <<EOF
-- Activate Kitty and send the specified text string
tell application "kitty"
    activate
    delay 0.1
    tell application "System Events"
        keystroke "$TEXT" -- Simulate typing the text string
    end tell
end tell
EOF
