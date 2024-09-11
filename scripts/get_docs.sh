#!/bin/bash

# Path to the file containing the list of Google Doc URLs
URL_LIST_FILE=$(realpath ~/work/docs.md)

# Path to your chrome-docs-markdown.sh script
CHROME_DOCS_SCRIPT="chrome-docs-markdown.sh"

# Check if the URL list file exists
if [ ! -f "$URL_LIST_FILE" ]; then
    echo "Error: URL list file not found at $URL_LIST_FILE"
    exit 1
fi

# Check if the chrome-docs-markdown.sh script exists
if [ ! -f "$CHROME_DOCS_SCRIPT" ]; then
    echo "Error: chrome-docs-markdown.sh script not found at $CHROME_DOCS_SCRIPT"
    exit 1
fi

# Read the URL list file line by line
while IFS= read -r url
do
    # Skip empty lines and lines starting with #
    [[ -z "$url" || "$url" == \#* ]] && continue

    echo "Processing: $url"
    
    # Open the URL in Chrome
    open -a "Google Chrome" "$url"
    
    # Wait for the page to load (adjust the sleep time if needed)
    sleep 5
    
    # Run the chrome-docs-markdown.sh script
    bash "$CHROME_DOCS_SCRIPT"
    
    # Wait between processing each URL (adjust if needed)
    sleep 10
    
    echo "Finished processing: $url"
    echo "------------------------"
done < "$URL_LIST_FILE"

echo "All URLs processed."
