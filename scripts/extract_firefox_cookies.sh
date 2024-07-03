
#!/bin/bash -e
# extract_cookies.sh:
#
# Convert from Firefox's cookies.sqlite format to Netscape cookies,
# which can then be used by wget and curl.

# USAGE:
# $ ./extract_cookies.sh > /tmp/cookies.txt
# or
# $ ./extract_cookies.sh /path/to/cookies.sqlite > /tmp/cookies.txt

cleanup() {
    [ -f "$TMPFILE" ] && rm -f "$TMPFILE"
    exit 0
}
trap cleanup EXIT INT QUIT TERM

find_firefox_profile() {
    local os=$(uname)
    local profile_path

    if [ "$os" = "Darwin" ]; then
        profile_path="$HOME/Library/Application Support/Firefox/Profiles"
    elif [ "$os" = "Linux" ]; then
        profile_path="$HOME/.mozilla/firefox"
    else
        echo "Unsupported operating system" >&2
        exit 1
    fi

    find "$profile_path" -name "cookies.sqlite" -print0 | xargs -0 ls -t | head -1
}

if [ "$#" -ge 1 ]; then
    SQLFILE="$1"
else
    if [ -t 0 ]; then
        SQLFILE=$(find_firefox_profile)
    else
        SQLFILE="-"
    fi
fi

if [ "$SQLFILE" != "-" ] && [ ! -r "$SQLFILE" ]; then
    echo "Error. File $SQLFILE is not readable." >&2
    exit 1
fi

TMPFILE=$(mktemp /tmp/cookies.sqlite.XXXXXXXXXX)
cat "$SQLFILE" > "$TMPFILE"

echo "# Netscape HTTP Cookie File"
sqlite3 -separator $'\t' "$TMPFILE" <<EOF
.mode tabs
.header off
select host,
case substr(host,1,1)='.' when 0 then 'FALSE' else 'TRUE' end,
path,
case isSecure when 0 then 'FALSE' else 'TRUE' end,
expiry,
name,
value
from moz_cookies;
EOF

cleanup

