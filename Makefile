.PHONY: test eval serve

test:
	PYTHONPATH=. pytest -q

eval:
	PYTHONPATH=. python -m app.eval_cli

serve:
	PYTHONPATH=. uvicorn app.main:app --reload --port 8000
