import os
import json
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LOG_FILE = "logs/app.log"
RUNBOOK_FILE = "runbook_map.json"


def read_logs(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def extract_errors(logs):
    # very simple extraction
    return [l for l in logs if "ERROR" in l or "WARN" in l]


def summarize_with_llm(error_lines):
    joined_logs = "\n".join(error_lines[:20])  # limit context for small demo
    prompt = (
        "You are a production support assistant. "
        "Summarize the likely root cause and impacted components from these logs:\n\n"
        f"{joined_logs}\n\n"
        "Return 2–3 concise sentences."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def suggest_runbooks(error_lines):
    with open(RUNBOOK_FILE, "r") as f:
        mapping = json.load(f)
    suggestions = set()
    for line in error_lines:
        for key, link in mapping.items():
            if key in line:
                suggestions.add(f"{key} → {link}")
    return suggestions


def main():
    logs = read_logs(LOG_FILE)
    errors = extract_errors(logs)

    print("=== Top Error Patterns ===")
    counts = Counter([e.split(":")[1].strip().split()[0] for e in errors])
    for err, count in counts.most_common():
        print(f"{err}: {count}")

    print("\n=== LLM Root Cause Summary ===")
    summary = summarize_with_llm(errors)
    print(summary)

    print("\n=== Suggested Runbooks ===")
    for s in suggest_runbooks(errors):
        print(s)


if __name__ == "__main__":
    main()
