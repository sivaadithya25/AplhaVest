import os
import smtplib
from email.message import EmailMessage

def send_gmail_report(subject, body, recipient=None):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    recipient = recipient or os.getenv("CLIENT_EMAIL")

    if not sender or not password or not recipient:
        return False, "Gmail environment variables are not configured."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)

    return True, "Report sent successfully."
