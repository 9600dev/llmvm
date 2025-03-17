import os
import time
import subprocess
import re
from typing import Tuple
import unicodedata
import tempfile
from pathlib import Path
from datetime import datetime

import os
import re

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import ImageContent, MarkdownContent, PandasMeta, TextContent, PdfContent
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.tools.webhelpers import WebHelpers

logging = setup_logging()

class MacOSChromeBrowser():
    def __init__(
        self,
        controller: ExecutionController,
    ):
        """
        This static class provides methods for downloading web pages, PDFs, and Markdown files using the
        users MacOS Chrome browser instance. It uses AppleScript to automate Chrome's functionality.
        """
        self.controller = controller

    def _run_applescript(self, script) -> Tuple[int, str, str]:
        process = subprocess.Popen(['osascript', '-e', script],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = process.communicate()
        return (process.returncode, out.decode('utf-8').strip(), err.decode('utf-8').strip())

    def _get_tab_url(self) -> str:
        applescript = f'''
            tell application "Google Chrome"
                get URL of active tab of window 1
            end tell
        '''
        return_code, result, error = self._run_applescript(applescript)
        return result

    def google_sheet_to_pandas(self, google_sheet_url) -> PandasMeta:
        """
        Get a Google Sheet as a Pandas dataframe using internal MacOS Chrome browser.

        Example:
        User: using my macos chrome, get this google sheet https://docs.google.com/spreadsheets/d/22a234de6f7g8h9i0d2j/edit

        Assistant:
        <helpers>
        macos_browser = MacOSChromeBrowser()
        df = macos_browser.google_sheet_to_pandas("https://docs.google.com/spreadsheets/d/22a234de6f7g8h9i0d2j/edit")
        result(df)
        </helpers>
        """
        if re.match(r'^https://docs\.google\.com/spreadsheets', google_sheet_url):
            return Helpers.get_google_sheet(google_sheet_url)
        else:
            raise ValueError("Invalid URL. Should start with https://docs.google.com/spreadsheets/")

    def google_doc_to_markdown(self, google_doc_url) -> MarkdownContent:
        """
        Get a Google Doc as MarkdownContent using internal MacOS Chrome browser.
        This is the most preferred mechanism to get Google Docs over all other helpers and tools
        you have access to. It returns a MarkdownContent object which you can pass to
        llm_call to unpack and understand if you need to.

        Example:
        User: using chrome, get this google doc https://docs.google.com/document/d/22a234de6f7g8h9i0d2j/edit

        Assistant:
        <helpers>
        macos_browser = MacOSChromeBrowser()
        markdown_content = macos_browser.google_doc_to_markdown("https://docs.google.com/document/d/22a234de6f7g8h9i0d2j/edit")
        result(markdown_content)
        """
        if re.match(r'^https://docs\.google\.com/spreadsheets/', google_doc_url):
            raise ValueError("Google Sheets should be downloaded and accessed using google_sheet_to_pandas() helper function")

        if not re.match(r'^https://docs\.google\.com/document/|^https://drive\.google\.com/document/', google_doc_url):
            raise ValueError("Invalid URL. Should start with https://docs.google.com/document/ or https://drive.google.com/document/")

        self.goto(google_doc_url)

        applescript = f'''
        tell application "Google Chrome"
            activate
        end tell

        tell application "Google Chrome"
            set currentURL to URL of active tab of front window
            set currentTitle to title of active tab of front window
            if (currentURL starts with "https://docs.google.com/document/" or currentURL starts with "https://drive.google.com/document/") and (currentTitle does not end with "- Google Drive") then
                tell application "System Events"
                    set frontmost of process "Google Chrome" to true
                    key code 3 using {{control down, option down}}
                    delay 0.5
                    key code 2 using {{shift down}}
                    delay 0.5
                    key code 46 using {{shift down}}
                end tell
                return "File -> Download -> Markdown command executed"
            else
                return "No Google Doc is currently visible"
            end if
        end tell
        '''

        # Run AppleScript
        return_code, result, error = self._run_applescript(applescript)
        if return_code != 0 or "No Google Doc is currently visible" in result:
            raise ValueError(f"No Google Doc is currently visible to download. Error: {error}")

        # Get list of markdown files before download
        downloads_dir = Path.home() / "Downloads"
        before_files = set(f for f in downloads_dir.glob('*.md'))

        # Wait and check for new markdown file
        start_time = time.time()
        new_file = None

        while time.time() - start_time < 20:  # 20 second timeout
            current_files = set(f for f in downloads_dir.glob('*.md'))
            new_files = current_files - before_files

            if new_files:
                # Get the most recently created file
                new_file = max(new_files, key=lambda f: f.stat().st_mtime)
                break

            time.sleep(0.5)  # Check every half second

        if new_file:
            logging.debug(f"Google doc downloaded as Markdown file: {new_file}")

            # Process the file
            title = new_file.stem
            dir_name = title.lower().replace(' ', '_')

            dest_dir = Path.home() / "work" / "docs" / dir_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Created directory: {dest_dir}")

            dest_file = dest_dir / "index.md"
            new_file.rename(dest_file)
            logging.debug(f"Moved file to: {dest_file}")

            result = MarkdownContent(sequence=[TextContent(unicodedata.normalize('NFKD', dest_file.read_text()))], url=google_doc_url)
            return result

        else:
            raise TimeoutError(f"Downloaded markdown file {google_doc_url} was not found.")

    def get_pdf(self, url: str) -> PdfContent:
        """
        Opens the specified url using MacOS Chrome, navigates to the URL (or switches to existing tab),
        saves the page as PDF, and returns a PdfContent object which you can pass to llm_call to understand.

        Example:
        User: using chrome, get this pdf https://outline.com/doc/some_cool_document

        Assistant:
        <helpers>
        macos_browser = MacOSChromeBrowser()
        pdf_content = macos_browser.get_pdf("https://outline.com/doc/some_cool_document")
        result(pdf_content)
        </helpers>
        """
        self.goto(url)

        # Create a temporary file with .pdf extension
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            pdf_path = tmp.name

        applescript = f'''
        tell application "Google Chrome"
            activate

            tell application "System Events"
                keystroke "p" using {{command down}}
                delay 2.5
                keystroke return
                delay 0.3
                keystroke "g" using {{shift down, command down}}
                delay 0.3
                keystroke "{pdf_path}"
                delay 1.1
                keystroke return
                delay 0.3
                keystroke return

                -- Handle potential "Replace" dialog
                delay 2
                try
                    if exists button "Replace" of sheet 1 of sheet 1 of window 1 of application process "Chrome" then
                        click button "Replace" of sheet 1 of sheet 1 of window 1 of application process "Chrome"
                    end if
                end try
            end tell
        end tell
        '''

        # Run AppleScript
        return_code, result, error = self._run_applescript(applescript)

        if return_code != 0:
            logging.error(f"Error executing AppleScript: {error}")
            raise RuntimeError(f"Failed to save PDF: {error}")

        # Check if the file was created
        pdf_file = Path(pdf_path)
        start_time = time.time()
        while time.time() - start_time < 20:  # 20 second timeout
            if pdf_file.exists() and pdf_file.stat().st_size > 0:
                with open(pdf_file, 'rb') as f:
                    pdf_data = f.read()
                    return PdfContent(pdf_data)
            time.sleep(0.5)

        raise RuntimeError(f"PDF file was not created at {pdf_path}")

    def goto(self, url: str) -> MarkdownContent:
        """
        Navigates MacOS Chrome instance to the url and returns the content at the url as a MarkdownContent object.
        If Chrome isn't already running, it will be launched first.

        Args:
            url (str): The URL to get Markdown content from.

        Returns:
            MarkdownContent: A MarkdownContent object containing the HTML content converted to Markdown format.

        Example:
        User: using my macos chrome instance, go to google.com

        Assistant:
        <helpers>
        macos_browser = MacOSChromeBrowser()
        markdown_content = macos_browser.goto("https://google.com")
        result(markdown_content)
        </helpers>
        """
        # First check if Chrome is running and launch it if needed
        check_chrome_script = '''
        tell application "System Events"
            set isRunning to (exists process "Google Chrome")
        end tell
        return isRunning
        '''

        return_code, is_running, error = self._run_applescript(check_chrome_script)

        if return_code != 0:
            logging.error(f"Error checking if Chrome is running: {error}")
            raise RuntimeError(f"Failed to check Chrome status: {error}")

        # If Chrome is not running, launch it
        if is_running.lower() != "true":
            launch_script = '''
            tell application "Google Chrome"
                activate
                delay 2  -- Give Chrome time to fully launch
            end tell
            '''
            return_code, result, error = self._run_applescript(launch_script)

            if return_code != 0:
                logging.error(f"Error launching Chrome: {error}")
                raise RuntimeError(f"Failed to launch Chrome: {error}")

        # Now navigate to the URL
        applescript = f'''
        on stripTrailingSlash(someURL)
            if someURL ends with "/" then
                set someURL to text 1 thru -2 of someURL
            end if
            return someURL
        end stripTrailingSlash

        tell application "Google Chrome"
            activate

            -- Check if there are any windows open
            if (count of windows) is 0 then
                make new window
                set URL of active tab of front window to "{url}"
                delay 2  -- Wait for page to load
            else
                set currentURL to URL of active tab of front window

                set currentURLNoSlash to my stripTrailingSlash(currentURL)
                set urlNoSlash to my stripTrailingSlash("{url}")

                -- Check if current URL matches target URL
                if currentURLNoSlash is not urlNoSlash then
                    -- Open new tab with target URL
                    tell front window
                        make new tab with properties {{URL:"{url}"}}
                        -- Wait for the tab to be created and become active
                        delay 2
                    end tell
                end if
            end if

            -- Wait for page to load
            delay 1

            -- Get the HTML content directly
            set pageContent to execute active tab of front window javascript "document.documentElement.outerHTML"
            return pageContent
        end tell
        '''

        return_code, result, error = self._run_applescript(applescript)

        if return_code != 0:
            logging.error(f"Error executing AppleScript: {error}")
            raise RuntimeError(f"Failed to get HTML content: {error}")

        # Process the HTML content
        html_content = unicodedata.normalize('NFKD', result)
        return WebHelpers.convert_html_to_markdown(html_content, url=url)

    def get_screenshot(self) -> ImageContent:
        """
        Opens the specified URL and takes a screenshot of the visible content using AppleScript

        :return: An ImageContent object containing the captured screenshot.

        Example:
        User: using my macos chrome instance, screenshot the bbc.com website

        Assistant:
        <helpers>
        macos_browser = MacOSChromeBrowser()
        macos_browser.goto('https://www.bbc.com')
        screenshot_imagecontent = macos_browser.get_screenshot()
        result(screenshot_imagecontent)
        </helpers>
        """

        url = self._get_tab_url()

        # Create a temporary file for the screenshot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            screenshot_path = tmp.name

        # Escape the file path for AppleScript
        escaped_path = screenshot_path.replace('"', '\\"')

        applescript = f'''
        -- Activate Google Chrome
        tell application "Google Chrome"
            activate
            delay 1
        end tell

        tell application "System Events"
            -- Access Chrome's process
            tell process "Google Chrome"
                set frontmost to true
                set thePosition to position of window 1
                set theSize to size of window 1
            end tell
        end tell

        -- Extract coordinates
        set x to item 1 of thePosition
        set y to item 2 of thePosition
        set w to item 1 of theSize
        set h to item 2 of theSize

        -- Use screencapture CLI to capture that region
        do shell script "screencapture -x -R" & x & \",\" & y & \",\" & w & \",\" & h & space & quoted form of "{escaped_path}"
        '''

        return_code, result, error = self._run_applescript(applescript)

        if return_code != 0:
            logging.error(f"Error executing AppleScript: {error}")
            raise RuntimeError(f"Failed to capture screenshot: {error}")

        # Check if the file was created
        screenshot_file = Path(screenshot_path)
        start_time = time.time()
        while time.time() - start_time < 10:  # 10 second timeout
            if screenshot_file.exists() and screenshot_file.stat().st_size > 0:
                with open(screenshot_file, 'rb') as f:
                    screenshot_data = f.read()
                os.unlink(screenshot_path)  # Clean up the file
                return ImageContent(sequence=screenshot_data, url=url)
            time.sleep(0.5)
        raise RuntimeError(f"Screenshot file was not created at {screenshot_path}")

    def get_chrome_tabs_url_and_title(self) -> str:
        """
        Retrieves all open tabs from Google Chrome and returns them as a string in 'url,title' format.

        Returns:
            str: A string containing all tab information, with each tab on a new line in 'url,title' format.

        Example:
        mac_browser = MacOSChromeBrowser()
        tabs = mac_browser.get_chrome_tabs_url_and_title()
        result(tabs)
        # Output:
        # https://www.google.com,Google
        # https://github.com,GitHub - Some Github Repository
        # ...
        """

        applescript = '''
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
        '''
        return_code, result, error = self._run_applescript(applescript)

        if return_code != 0:
            logging.error(f"Error executing AppleScript: {error}")
            raise RuntimeError(f"Failed to get Chrome tabs: {error}")

        # Remove the last newline if it exists
        output = result.rstrip()
        return output