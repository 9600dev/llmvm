
#!/bin/bash -e
# extract_chrome_cookies.sh:

# Convert from Chrome's Cookies format to Netscape cookies,
# which can then be used by wget and curl.

# USAGE:
#
# $ extract_chrome_cookies.sh > /tmp/cookies.txt
# or
# $ extract_chrome_cookies.sh ~/.config/google-chrome/Default/Cookies > /tmp/cookies.txt

# USING WITH WGET:
# $ wget --load-cookies=/tmp/cookies.txt http://mysite.com

# USING WITH CURL:
# $ curl --cookie /tmp/cookies.txt http://mysite.com

# Note: If you do not specify a SQLite filename, this script will
# find the default Chrome Cookies file.

cleanup() {
    rm -f $TMPFILE
    exit 0
}
trap cleanup EXIT INT QUIT TERM

if [ "$#" -ge 1 ]; then
    SQLFILE="$1"
else
    SQLFILE=$(ls -t ~/.config/google-chrome/Default/Cookies | head -1)
fi

if [ "$SQLFILE" != "-" -a ! -r "$SQLFILE" ]; then
    echo "Error. File $SQLFILE is not readable." >&2
    exit 1
fi

# Copy the Cookies file because Chrome keeps a lock on it
TMPFILE=$(mktemp /tmp/chrome_cookies.XXXXXXXXXX)
cat "$SQLFILE" > $TMPFILE

echo "# Netscape HTTP Cookie File"
sqlite3 -separator $'\t' $TMPFILE << EOF
.mode tabs
.header off
select host_key,
case substr(host_key,1,1)='.' when 0 then 'FALSE' else 'TRUE' end,
path,
case is_secure when 0 then 'FALSE' else 'TRUE' end,
expires_utc/1000000-11644473600,
name,
value
from cookies;
EOF

cleanup
