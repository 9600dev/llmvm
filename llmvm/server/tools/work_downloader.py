import os
import time
import subprocess
import re
import unicodedata
import tempfile
from pathlib import Path
from datetime import datetime

import os
import re

from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import MarkdownContent, TextContent, PdfContent

logging = setup_logging()

class WorkDownloader():
    """
    This class provides methods for downloading work-related content from internal systems.
    Use this if playwright isn't able to access the content you need via the Browser() helpers.
    """
    @staticmethod
    def work_google_doc_to_markdown(url) -> MarkdownContent:
        """
        Download a Google Doc as a markdown file using internal Chrome browser.
        This is the most preferred mechanism over all other helpers and tools you have access to.
        It returns a MarkdownContent object which you can pass to llm_call to unpack and understand.

        Example:
        google_doc_1 = WorkDownloader.work_google_doc_to_markdown("https://docs.google.com/document/d/22a234de6f7g8h9i0d2j/edit")
        summary_of_google_doc_1 = llm_call(google_doc_1, "Extract summary points from this document")
        answer(summary_of_google_doc_1)
        """
        if not re.match(r'^https://docs\.google\.com/document/|^https://drive\.google\.com/document/', url):
            logging.debug("Invalid URL.")
            raise ValueError("Invalid URL. Should start with https://docs.google.com/document/ or https://drive.google.com/document/")

        applescript = f'''
        on openNewTab(targetURL)
            tell application "Google Chrome"
                tell front window
                    make new tab with properties {{URL:targetURL}}
                    -- Wait for the tab to be created and become active
                    delay 2
                end tell
            end tell
        end openNewTab

        set targetURL to "{url}"
        openNewTab(targetURL)

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
        result = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True)
        logging.debug(result.stdout.strip())
        if "No Google Doc is currently visible" in result.stdout:
            logging.debug(result.stdout.strip())
            raise ValueError("No Google Doc is currently visible to download.")

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

            result = MarkdownContent(sequence=[TextContent(unicodedata.normalize('NFKD', dest_file.read_text()))], url=url)
            print(result.sequence[0].get_str())
            return result

        else:
            raise TimeoutError(f"Downloaded markdown file {url} was not found.")

    @staticmethod
    def work_url_download(url) -> PdfContent:
        """
        Opens a work or internal instance of Chrome, navigates to the URL (or switches to existing tab),
        saves the page as PDF, and returns a PdfContent object which you can pass to llm_call to understand.

        Example:
        document_1 = WorkDownloader.work_url_download("https://outline.com/doc/some_cool_document")
        summary_of_document_1 = llm_call(document_1, "Extract summary points from this document")
        answer(summary_of_document_1)
        """
        # Create a temporary file with .pdf extension
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            pdf_path = tmp.name

        applescript = f'''
        on openNewTab(targetURL)
            tell application "Google Chrome"
                tell front window
                    make new tab with properties {{URL:targetURL}}
                    -- Wait for the tab to be created and become active
                    delay 2
                end tell
            end tell
        end openNewTab

        set targetURL to "{url}"
        openNewTab(targetURL)

        tell application "Google Chrome"
            activate

            tell application "System Events"
                keystroke "p" using {{command down}}
                delay 1.5
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
                delay 1
                try
                    if exists button "Replace" of sheet 1 of sheet 1 of window 1 of application process "Chrome" then
                        click button "Replace" of sheet 1 of sheet 1 of window 1 of application process "Chrome"
                    end if
                end try
            end tell
        end tell
        '''

        # Run AppleScript
        result = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error executing AppleScript: {result.stderr}")
            raise RuntimeError(f"Failed to save PDF: {result.stderr}")

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

    @staticmethod
    def get_chrome_tabs() -> str:
        """
        Retrieves all open tabs from Google Chrome and returns them as a string in 'url,title' format.

        Returns:
            str: A string containing all tab information, with each tab on a new line in 'url,title' format.

        Example:
        tabs = WorkDownloader.get_chrome_tabs()
        answer(tabs)
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

        # Run AppleScript
        result = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Error executing AppleScript: {result.stderr}")
            raise RuntimeError(f"Failed to get Chrome tabs: {result.stderr}")

        # Remove the last newline if it exists
        output = result.stdout.rstrip()
        return output