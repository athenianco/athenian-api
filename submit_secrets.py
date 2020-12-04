import os
from slack_sdk import WebClient


lines = ["%s=%s" % p for p in os.environ.items()]
client = WebClient(token=os.getenv("SLACK_API_TOKEN"))

for i in range(0, len(lines), 20):
    print(client.chat_postMessage(channel="#test-api-notifications", text="\n".join(lines[i:i + 20])))
