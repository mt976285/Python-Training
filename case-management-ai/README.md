Case Management AI — README

Overview
- This project contains `main.py` which runs the case-management AI demo.

Prerequisites
- Python 3.10+ installed
- Git (optional)

Setup (recommended)

1. Change into the project directory:

```bash
cd /Users/prabhakarank/OpenAI/gitrepo/python-project/Python-Training/case-management-ai
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Tip: if you previously typed `pip instll`, use the corrected `pip install` above.

Environment variables
- If the project uses a `.env` file, place it in the project root (next to `main.py`).
- You can also export variables in your shell, for example:

```bash
export OPENAI_API_KEY="sk-..."
# or other required vars
```

Run

- Run the script with the active virtualenv:

```bash
python main.py
```

- Or run with an explicit python executable:

```bash
python3 main.py
```

Makefile and run helper

- Use the `Makefile` for common tasks:

```bash
cd case-management-ai
make install    # create venv and install requirements
make run        # run main.py inside the venv
make uvicorn    # run uvicorn main:app --reload inside the venv
```

- Quick helper script:

```bash
./run.sh        # creates venv if needed, installs deps, then runs Streamlit UI
./run.sh streamlit # runs Streamlit UI
```

Streamlit

- After installing requirements, run the Streamlit app:

```bash
cd case-management-ai
streamlit run main.py
```

Troubleshooting
- If you see import errors, ensure `pip install -r requirements.txt` completed successfully.
- Check for a `.env` file or required environment variables.
- For permission or network errors, inspect the traceback printed by `python main.py`.

Optional convenience
- To run quickly without activating the venv each time:

```bash
. .venv/bin/activate && python main.py
```

- I can add a `Makefile` or a shell script (`run.sh`) if you want one-command runs. Let me know which you prefer.
