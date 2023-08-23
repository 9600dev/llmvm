#!/bin/bash -e
# extract_cookies.sh:
#
# Convert from Firefox's cookies.sqlite format to Netscape cookies,
# which can then be used by wget and curl. (Why don't wget and curl
# just use libsqlite if it's installed?)

# USAGE:
#
# $ extract_cookies.sh > /tmp/cookies.txt
# or
# $ extract_cookies.sh ~/.mozilla/firefox/*default*/cookies.sqlite > /tmp/cookies.txt

# USING WITH WGET:
# $ wget --load-cookies=/tmp/cookies.txt http://mysite.com

# USING WITH CURL:
# $ curl --cookie /tmp/cookies.txt http://mysite.com

# Note: If you do not specify an SQLite filename, this script will
# intelligently find it for you.
#
# A) Usually it will check all profiles under ~/.mozilla/firefox/ and
# use the cookies.sqlite that was updated most recently.
#
# B) If you've redirected stdin (with < or |) , then that will be used.


# HISTORY: I believe this is circa 2010 from:
# http://slacy.com/blog/2010/02/using-cookies-sqlite-in-wget-or-curl/
# However, that site is down now.

# Cleaned up by Hackerb9 (2017) to be more robust and require less typing.


cleanup() {
    rm -f $TMPFILE
    exit 0
}
trap cleanup  EXIT INT QUIT TERM


if [ "$#" -ge 1 ]; then
    SQLFILE="$1"
else
    if tty -s; then
	SQLFILE=$(ls -t ~/.mozilla/firefox/*/cookies.sqlite | head -1)
    else
	SQLFILE="-"		# Will use 'cat' below to read stdin
    fi
fi
    
if [ "$SQLFILE" != "-" -a ! -r "$SQLFILE" ]; then
    echo "Error. File $SQLFILE is not readable." >&2
    exit 1
fi

# We have to copy cookies.sqlite, because FireFox has a lock on it
TMPFILE=`mktemp /tmp/cookies.sqlite.XXXXXXXXXX`
cat "$SQLFILE" >> $TMPFILE


# This is the format of the sqlite database:
# CREATE TABLE moz_cookies (id INTEGER PRIMARY KEY, name TEXT, value TEXT, host TEXT, path TEXT,expiry INTEGER, lastAccessed INTEGER, isSecure INTEGER, isHttpOnly INTEGER);

echo "# Netscape HTTP Cookie File"
sqlite3 -separator $'\t' $TMPFILE <<- EOF
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
