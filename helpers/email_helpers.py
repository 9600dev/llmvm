import datetime
import datetime as dt
import os
import smtplib
import tempfile
from email.encoders import encode_base64
from email.mime.base import MIMEBase
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List

import icalevents.icalparser
import pytz
from icalendar import Calendar, Event, vCalAddress, vDatetime, vText

from helpers.container import Container


class EmailHelpers():
    @staticmethod
    def parse_events(ics_content: str) -> List[Dict[str, str]]:
        """Parse an ics file and return the events

        Args:
            ics_content (str): ics file content

        Returns:
            List[Dict[str, str]]: list of events
        """
        events = []
        for event in icalevents.icalparser.parse_events(ics_content):
            events.append({
                'summary': event.summary,
                'description': event.description,
                'location': event.location,
                'start': event.start,
                'end': event.end,
                'all_day': event.all_day,
            })

        return events

    @staticmethod
    def __send_email_helper(
        sender: str,
        receiver: str,
        mime_text: MIMEMultipart | MIMEText,
    ):
        """Send an email from sender to receiver with the specified subject and body text"""
        container = Container()
        smtp_server = container.get('smtp_server')
        smtp_port = int(container.get('smtp_port'))
        smtp_username = container.get('smtp_username')
        smtp_password = container.get('smtp_password')

        mailserver = smtplib.SMTP(smtp_server, smtp_port)
        mailserver.login(smtp_username, smtp_password)
        mailserver.sendmail(
            sender,
            receiver,
            mime_text.as_string())
        mailserver.quit()

    @staticmethod
    def send_email(
        sender_email: str,
        receiver_email: str,
        subject: str,
        body: str,
    ):
        """Send an email from sender to receiver with the specified subject and body text"""
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        EmailHelpers.__send_email_helper(
            sender_email,
            receiver_email,
            msg,
        )

    @staticmethod
    def send_calendar_invite(
        from_name: str,
        from_email: str,
        attendee_emails: List[str],
        subject: str,
        body: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ):
        """Send a calendar invite to the attendee

        Args:
            from_name (str): name of the sender
            from_email (str): email of the sender
            attendee_email (str): email of the attendee
        """
        cal = Calendar()
        cal.add('prodid', '-//9600//CalendarApp//EN')
        cal.add('version', '2.0')
        cal.add('method', 'REQUEST')

        local_timezone = dt.datetime.now().astimezone().tzinfo
        utc_timezone = pytz.timezone('UTC')

        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=local_timezone)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=local_timezone)

        now = dt.datetime.now().replace(tzinfo=local_timezone)

        # Add subcomponents
        event = Event()
        event.add('name', subject)
        event.add('summary', body)

        event['dtstart'] = start_date.astimezone(utc_timezone).strftime('%Y%m%dT%H%M%SZ')
        event['dtend'] = end_date.astimezone(utc_timezone).strftime('%Y%m%dT%H%M%SZ')
        event['dtstamp'] = now.astimezone(utc_timezone).strftime('%Y%m%dT%H%M%SZ')

        # Add the organizer
        organizer = vCalAddress('MAILTO:{}'.format(from_email))

        # Add parameters of the event
        organizer.params['name'] = vText(from_name)
        # organizer.params['role'] = vText('CEO')
        event['organizer'] = organizer
        event['location'] = vText('Not supplied')

        import uuid

        event['uid'] = str(uuid.uuid4())
        event.add('priority', 5)

        for attendee_email in attendee_emails:
            attendee = vCalAddress('MAILTO:{}'.format(attendee_email))
            # attendee.params['name'] = vText('Richard Roe')
            # attendee.params['role'] = vText('REQ-PARTICIPANT')
            event.add('attendee', attendee, encode=0)

            # Add the event to the calendar
            cal.add_component(event)
            with tempfile.NamedTemporaryFile(suffix='.ics', mode='wb', delete=False) as f:
                f.write(cal.to_ical())
                f.close()

                msg = MIMEMultipart("alternative")

                msg["Subject"] = subject
                msg["From"] = from_email
                msg["To"] = attendee_email
                msg["Content-Class"] = "urn:content-classes:calendarmessage"
                msg["Content-Type"] = "text/calendar; method=REQUEST"

                msg.attach(MIMEText(body))

                filename = f.name
                part = MIMEBase('text', "calendar", method="REQUEST", name=filename)
                part.set_payload(cal.to_ical())
                encode_base64(part)
                part.add_header('Content-Description', filename)
                part.add_header("Content-class", "urn:content-classes:calendarmessage")
                part.add_header('Content-Transfer-Encoding', '8bit')
                part.add_header("Filename", filename)
                part.add_header("Path", filename)
                msg.attach(part)

                for attendee_email in attendee_emails:
                    EmailHelpers.__send_email_helper(
                        from_email,
                        attendee_email,
                        msg,
                    )
