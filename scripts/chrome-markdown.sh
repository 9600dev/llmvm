#!/bin/bash

conda init "$(basename "${SHELL}")"

cd ~/

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/homebrew/Caskroom/miniforge/base/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]; then
        . "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
    else
        export PATH="/opt/homebrew/Caskroom/miniforge/base/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

eval "$(/opt/homebrew/bin/brew shellenv)"

conda activate llmvm

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
        delay 1.5
        keystroke return
        delay 0.3 
        keystroke "g" using {shift down, command down}
        delay 0.3 
        keystroke "$FILE"
        delay 1.0 
        keystroke return
        delay 1.0 
        keystroke return
    end tell
end tell
EOF

cd ~/dev/llmvm/scripts
python generate_markdown.py -m claude-3-haiku-20240307 -o ~/work/docs -t $FILE 

