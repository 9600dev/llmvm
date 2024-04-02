#!/bin/bash

osascript <<EOF 
-- Store the name of the currently active application
tell application "System Events"
	set activeAppName to name of first application process whose frontmost is true
end tell

-- Activate Kitty, paste from clipboard and press enter
tell application "kitty"
	activate
	tell application "System Events"
		key code 9 using {command down} -- Cmd+V to paste
		delay 0.1 -- Wait a moment for the paste and enter to register
		key code 36 -- Press Enter
		delay 0.1 -- Optional: Wait a moment before switching back
	end tell
end tell

-- Switch back to the originally active application
tell application activeAppName
	activate
end tell
