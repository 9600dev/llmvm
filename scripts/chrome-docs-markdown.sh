#!/bin/bash

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

sleep 8

DOWNLOADED_FILE=$(/bin/ls -t ~/Downloads/*.md 2>/dev/null | head -n 1)
if [ -n "$DOWNLOADED_FILE" ]; then
    echo "Latest downloaded Markdown file: $DOWNLOADED_FILE"

    # Check if the file is less than 10 seconds old
    FILE_AGE=$(($(date +%s) - $(date -r "$DOWNLOADED_FILE" +%s)))
    echo "File age: $FILE_AGE seconds"
    if [ $FILE_AGE -lt 20 ]; then
        echo "File is less than 20 seconds old. Processing..."

        TITLE=$(basename "$DOWNLOADED_FILE" .md)
        echo "File: $TITLE"

        # Convert title to lowercase and replace spaces with underscores
        DIR_NAME=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

        # Create the directory
        DEST_DIR="$HOME/work/docs/$DIR_NAME"
        mkdir -p "$DEST_DIR"
        echo "Created directory: $DEST_DIR"

        # Move the file to the new directory
        mv "$DOWNLOADED_FILE" "$DEST_DIR/index.md"
        echo "Moved file to: $DEST_DIR/index.md"
    else
        echo "Not processed: File is older than 10 seconds"
    fi
else
    echo "No Markdown files found in Downloads directory"
fi