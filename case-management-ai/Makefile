.PHONY: venv install run uvicorn clean

VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
ACTIVATE=. $(VENV)/bin/activate


venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

run:
	$(ACTIVATE) && streamlit run main.py

streamlit:
	$(ACTIVATE) && streamlit run main.py

clean:
	rm -rf $(VENV)
