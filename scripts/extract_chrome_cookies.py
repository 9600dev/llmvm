# Description: This script extracts the cookies from the Chrome browser and writes them to a file in the Netscape HTTP Cookie File format.
# code to decrypt the cookies is courtesy of https://github.com/n8henrie/pycookiecheat/

import sqlite3
import os.path
import urllib.parse
import keyring
import sys
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.ciphers.modes import CBC
from cryptography.hazmat.primitives.hashes import SHA1
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def clean(decrypted: bytes) -> str:
    last = decrypted[-1]
    if isinstance(last, int):
        return decrypted[:-last].decode("utf8")
    return decrypted[: -ord(last)].decode("utf8")

def chrome_decrypt(
    encrypted_value: bytes, key: bytes, init_vector: bytes
) -> str:
    # Encrypted cookies should be prefixed with 'v10' or 'v11' according to the
    # Chromium code. Strip it off.
    encrypted_value = encrypted_value[3:]

    cipher = Cipher(
        algorithm=AES(key),
        mode=CBC(init_vector),
    )
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(encrypted_value) + decryptor.finalize()

    return clean(decrypted)

def chrome_cookies():

    salt = b'saltysalt'
    iv = b' ' * 16
    length = 16

    # If running Chrome on OSX
    if sys.platform == 'darwin':
        my_pass = keyring.get_password('Chrome Safe Storage', 'Chrome')
        my_pass = my_pass.encode('utf8')
        iterations = 1003
        cookie_file = os.path.expanduser(
            '~/Library/Application Support/Google/Chrome/Default/Cookies'
        )

    # If running Chromium on Linux
    elif sys.platform == 'linux':
        my_pass = 'peanuts'.encode('utf8')
        iterations = 1
        cookie_file = os.path.expanduser(
            '~/.config/chromium/Default/Cookies'
        )
    else:
        raise Exception("This script only works on OSX or Linux.")

    kdf = PBKDF2HMAC(
        algorithm = SHA1(),
        iterations = iterations,
        length = length,
        salt = salt,
    )
    enc_key = kdf.derive(my_pass)

    conn = sqlite3.connect(cookie_file)
    sql = 'SELECT host_key, name, path, is_secure, ((expires_utc/1000000)-11644473600), encrypted_value, value FROM cookies'

    cookies_list = []

    with conn:
        for host_key, name, path, is_secure, expires_utc, encrypted_value, value in conn.execute(sql):

            # if there is a not encrypted value or if the encrypted value
            # doesn't start with the 'v10' prefix, return v
            if not value and (encrypted_value[:3] == b'v10'):
                value = chrome_decrypt(encrypted_value, key=enc_key, init_vector=iv)

            cookies_list.append({
                'domain': host_key,
                'path': path,
                'secure': bool(is_secure),
                'expires': max(expires_utc, 0),
                'name': name,
                'value': value,
            })

    return cookies_list

if __name__ == '__main__':
    output_path = os.path.expanduser('~/.config/llmvm/cookies.txt')
    cookies = chrome_cookies()

    with open(output_path, 'w') as f:
        f.write("# Netscape HTTP Cookie File\n")

        # for host_key, path, is_secure, name, _value, encrypted_value, _exptime in conn.execute(sql):
        for cookie in cookies:
            # Formatting the cookie attributes
            is_secure = 'TRUE' if cookie['secure'] else 'FALSE'
            
            # Writing the formatted cookie to the file
            f.write(f"{cookie['domain']}\t{'TRUE' if cookie['domain'].startswith('.') else 'FALSE'}\t{cookie['path']}\t{is_secure}\t{cookie['expires']}\t{cookie['name']}\t{cookie['value']}\n")
