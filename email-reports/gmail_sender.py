"""Authorize Gmail and send HTML reports with a plain-text version."""

import base64
import errno
import json
import os
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from urllib.parse import urlparse


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def save_private(path, content):
    # Write private files with owner-only access from the start.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as output:
        os.chmod(temporary, 0o600)
        output.write(content)
    temporary.replace(path)


def load_credentials(token_path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    if not token_path.exists():
        raise ValueError("Run main.py --authorize before sending reports")
    credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        save_private(token_path, credentials.to_json())
    if not credentials.valid:
        raise ValueError("Gmail authorization expired; run main.py --authorize again")
    return credentials


def find_credentials(folder, configured_path=None):
    # Use an explicit path first, then the usual name, then a downloaded Google file.
    if configured_path:
        path = folder / configured_path
        if not path.is_file():
            raise ValueError("GMAIL_CREDENTIALS_PATH does not point to an existing file")
        return path
    standard = folder / "credentials.json"
    if standard.is_file():
        return standard
    downloaded = sorted(folder.glob("client_secret_*.json"))
    if len(downloaded) != 1:
        raise ValueError("Keep one downloaded client_secret_*.json in email-reports, or set GMAIL_CREDENTIALS_PATH")
    return downloaded[0]


def authorization_options(credentials_path, redirect_uri):
    # Web clients need a fixed URI that matches the Google Cloud configuration.
    data = json.loads(credentials_path.read_text())
    if "installed" in data:
        return {"host": "localhost", "port": 0}
    if "web" not in data:
        raise ValueError("Expected Google Desktop app or Web application OAuth credentials")
    parsed = urlparse(redirect_uri)
    if (parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1")
            or not parsed.port or parsed.path != "/" or parsed.query or parsed.fragment or parsed.username):
        raise ValueError("GMAIL_REDIRECT_URI must be a loopback URL such as http://localhost:8080/")
    return {"host": parsed.hostname, "port": parsed.port}


def authorize(credentials_path, token_path, redirect_uri="http://localhost:8080/"):
    options = authorization_options(credentials_path, redirect_uri)
    from google_auth_oauthlib.flow import InstalledAppFlow

    # Open Google sign-in once. This command does not send email.
    if options["port"]:
        print(f"Web OAuth client: add {redirect_uri} as an Authorized redirect URI in Google Cloud before signing in.")
    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
    try:
        credentials = flow.run_local_server(**options, access_type="offline", prompt="consent")
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            raise ValueError(
                f"Gmail sign-in port {options['port']} is already in use. "
                "Set GMAIL_REDIRECT_URI to a free localhost port in email-reports/.env "
                "and add the exact same URI to the OAuth client in Google Cloud. No email was sent."
            ) from None
        raise ValueError(
            f"Gmail sign-in could not start or complete (OS error {error.errno}). "
            "Check the local callback port, browser, and network access. No email was sent."
        ) from None
    save_private(token_path, credentials.to_json())


def create_message(recipient, subject, body, html_body=None):
    message = EmailMessage()
    message["To"] = recipient
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=False, usegmt=True)
    message["Message-ID"] = make_msgid()
    message.set_content(body)
    if html_body:
        message.add_alternative(html_body, subtype="html")
    return {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")}


def send_report(token_path, recipient, subject, body, html_body=None):
    from googleapiclient.discovery import build

    credentials = load_credentials(token_path)
    with build("gmail", "v1", credentials=credentials, cache_discovery=False) as service:
        # Do not retry sending automatically: a lost response may hide a successful send.
        return service.users().messages().send(
            userId="me", body=create_message(recipient, subject, body, html_body)
        ).execute(num_retries=0)["id"]
