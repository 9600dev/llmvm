# https://chromewebstore.google.com/detail/j2team-cookies/okpidcojinmlaakglciglbpcpajaibco?hl=en
# says it's featured, but who knows...

# export to json file
# cat cookies.json | python extract_chrome_json.py >> ~/.local/share/llmvm/cookies.txt

import json
import sys
from datetime import datetime, timedelta

def chrome_to_netscape(chrome_cookie):
    domain = chrome_cookie.get('domain', '')
    domain_flag = "TRUE" if chrome_cookie.get('hostOnly', False) else "FALSE"
    
    path = chrome_cookie.get('path', '/')
    secure = "TRUE" if chrome_cookie.get('secure', False) else "FALSE"
    expires = int(chrome_cookie.get('expirationDate', 0))
    name = chrome_cookie.get('name', '')
    value = chrome_cookie.get('value', '')
    
    return f"{domain}\t{domain_flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}"

def main():
    try:
        data = json.load(sys.stdin)
        cookies = data.get('cookies', [])
        
        print("# Netscape HTTP Cookie File")
        for cookie in cookies:
            print(chrome_to_netscape(cookie))
    except json.JSONDecodeError:
        print("Error: Invalid JSON input", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing cookies: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
