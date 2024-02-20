import datetime as dt
import os
from typing import List, Optional, Tuple

import datetime

from llmvm.common.logging_helpers import setup_logging
from llmvm.server.tools.webhelpers import WebHelpers

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from email.header import decode_header
import email.utils
from base64 import urlsafe_b64decode

logging = setup_logging()

class GmailSearcher():
    """
    This module fetches emails from the gmail API
    For now it only reads newsletters, implemented as a search for emails from a bunch of newsy domains.
    It could do other things too (sending, downloading other stuff) but I'm less convinced of the value.

    To get this to work, follow the instructions here: https://developers.google.com/gmail/api/quickstart/python
    Then move the 'credentials.json' file you downloaded to ~/.config/llmvm/credentials.json
    The first time you run the code, it will open a browser window to authenticate with your gmail account.
    That will create a new file called 'token.json' in the root llmvm directory.
    """

    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

    NEWS_DOMAINS = [
        'substack',
        'theinformation',
        'washingtonpost',
        'nytimes',
        'sfchronicle',
        'thedispatch',
        'baltimoresun',
        'chims.net', # marginal revolution
        'smdailyjournal.com',
        'thediff',
        'parentdata',
        'bloomberg',
        'sfstandard',
        'theskimm.com',
        'theverge.com',
        'stratechery',
        'bensbites.co',
        'payloadspace.com',
        'spotrac.com',
        'amediaoperator',
        'gzeromedia',
        'oliverburkeman',
        'a16z',
        'platformer.news',
        'economist',
        'axios',
    ]

    @staticmethod
    def get_gmail_service():
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.

        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', GmailSearcher.SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    '~/.config/llmvm/credentials.json', GmailSearcher.SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        return build('gmail', 'v1', credentials=creds)

    @staticmethod
    def list_messages(service, user_id):
        since_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y/%m/%d")
        query = f'({" OR ".join("from:*" + domain for domain in GmailSearcher.NEWS_DOMAINS)}) after:{since_date} is:unread'
        response = service.users().messages().list(userId=user_id, q=query).execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])
        
        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id, q=query, 
                                                    pageToken=page_token).execute()
            messages.extend(response['messages'])

        return messages

    @staticmethod
    def decode_mime_header(s):
        decoded_parts = decode_header(s)
        # Concatenate decoded parts that may be in different encodings
        return ''.join(
            str(part[0], part[1] or 'utf-8') if isinstance(part[0], bytes) else part[0]
            for part in decoded_parts
        )

    @staticmethod
    def get_mime_message(service, user_id, message_id):
        message = service.users().messages().get(userId=user_id, id=message_id, format='raw').execute()
        msg_str = urlsafe_b64decode(message['raw'].encode('ASCII'))
        mime_msg = email.message_from_bytes(msg_str)
        return mime_msg

    @staticmethod
    def get_plain_body_from_mime(mime_msg):
        if mime_msg.is_multipart():
            for part in mime_msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if content_type == 'text/plain' and 'attachment' not in content_disposition:
                    try:
                        # Attempt UTF-8 decoding first
                        return part.get_payload(decode=True).decode('utf-8')
                    except UnicodeDecodeError:
                        # If UTF-8 decoding fails, use 'ISO-8859-1' or another suitable encoding
                        # Or use 'ignore' to ignore undecodable bytes
                        return part.get_payload(decode=True).decode('ISO-8859-1', errors='ignore')
        else:
            try:
                return mime_msg.get_payload(decode=True).decode('utf-8')
            except UnicodeDecodeError:
                return mime_msg.get_payload(decode=True).decode('ISO-8859-1', errors='ignore')

    @staticmethod
    def get_message_details(service, user_id, message_id):
        mime_msg = GmailSearcher.get_mime_message(service, user_id, message_id)
        if not mime_msg:
            return {}

        # Decode subject and from headers using the new decode_mime_header function
        subject = GmailSearcher.decode_mime_header(mime_msg['subject'])
        from_email = GmailSearcher.decode_mime_header(mime_msg['from'])

        parsed_date = email.utils.parsedate_to_datetime(mime_msg['Date'])
        date_time = parsed_date.strftime('%Y-%m-%d %H:%M:%S')

        # Assuming get_plain_body_from_mime remains the same, including its call
        body = GmailSearcher.get_plain_body_from_mime(mime_msg)

        return {'subject': subject, 'from': from_email, 'body': body, 'id': message_id, 'date_time': date_time}

    @staticmethod
    def search_newsletters(query):
        # ignore "query" for now, its behavior is a bit unclear in this case

        service = GmailSearcher.get_gmail_service()
        messages = GmailSearcher.list_messages(service, 'me')

        emails = []
        for message in messages:
            details = GmailSearcher.get_message_details(service, 'me', message['id'])
            emails.append(details)

        return emails
