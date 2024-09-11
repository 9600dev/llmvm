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

# Run the AppleScript
osascript <<EOF

tell application "Google Chrome"
    set currentURL to URL of active tab of front window
    set currentTitle to title of active tab of front window
    
    if (currentURL starts with "https://docs.google.com/document/" or currentURL starts with "https://drive.google.com/document/") and (currentTitle does not end with "- Google Drive") then
        -- A Google Doc is visible, so let's perform the key presses
        tell application "System Events"
            -- Activate Chrome to ensure it receives the key presses
            set frontmost of process "Google Chrome" to true
            
            -- Open File menu
            key code 3 using {control down, option down} -- 3 is the key code for "F"
            delay 0.5 -- Wait for menu to open
            
            -- Select Download
            key code 2 using {shift down} -- 2 is the key code for "D"
            delay 0.5 -- Wait for submenu to open
            
            -- Select Markdown
            key code 46 using {shift down} -- 46 is the key code for "M"
        end tell
        return "File -> Download -> Markdown command executed on the Google Doc"
    else
        return "No Google Doc is currently visible"
    end if
end
EOF

sleep 2 

DOWNLOADED_FILE=$(ls -t ~/Downloads/*.md 2>/dev/null | head -n 1)
if [ -n "$DOWNLOADED_FILE" ]; then
    echo "Latest downloaded Markdown file: $DOWNLOADED_FILE"
    
    # Check if the file is less than 10 seconds old
    FILE_AGE=$(($(date +%s) - $(date -r "$DOWNLOADED_FILE" +%s)))
    if [ $FILE_AGE -lt 10 ]; then
        echo "File is less than 10 seconds old. Processing..."
        # Add your processing code here
        # For example:
        # cd ~/dev/llmvm/scripts
        # python generate_markdown.py -m claude-3-haiku-20240307 -o ~/work/docs -t $FILE 
    else
        echo "Not processed: File is older than 10 seconds"
    fi
else
    echo "No Markdown files found in Downloads directory"
fi

# cd ~/dev/llmvm/scripts
# python generate_markdown.py -m claude-3-haiku-20240307 -o ~/work/docs -t $FILE 

