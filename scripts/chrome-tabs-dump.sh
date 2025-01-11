#!/bin/bash

osascript <<EOF
tell application "Google Chrome"
    set tabList to {}
    set windowList to every window
    repeat with theWindow in windowList
        set tabList to tabList & (every tab of theWindow whose URL is not "")
    end repeat

    set output to ""
    repeat with theTab in tabList
        set theURL to URL of theTab
        set theTitle to title of theTab
        set output to output & theURL & "," & theTitle & linefeed
    end repeat

    return output
end tell
EOF
