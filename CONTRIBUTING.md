# Contributing

Thanks for helping. This is a small, personal project, so keep changes small and easy to review.

## Before you send a change

- Run the tests: `pytest -q`
- Keep the no-key path working: `tokenlens count` and `tokenlens analyze data/sample_log.jsonl` must work with no API key and no download.
- Do not add a new model price without a source, and do not change a price silently — prices are the point of the tool.

## Reporting a problem

Open an issue and include:

- The exact command you ran
- What happened, and what you expected instead
- Your Python version and how you installed it (`pip install -e .` or similar)

Do not include API keys, private logs, or personal details.
