import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# C:\Users\hi\Documents\fde\.env 에서 CLAUDE_API_KEY를 읽음
load_dotenv(Path(r"C:\Users\hi\Documents\fde\.env"))

client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])

response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "안녕하세요"}],
)

for block in response.content:
    if block.type == "text":
        print(block.text)
