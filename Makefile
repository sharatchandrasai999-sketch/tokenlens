.PHONY: install test count analyze app
install:
	pip install -e ".[dev,app]"
test:
	pytest -q
count:
	tokenlens count --text "Summarize this contract in plain English." --output-tokens 200
analyze:
	tokenlens analyze data/sample_log.jsonl
app:
	streamlit run src/tokenlens/app.py
