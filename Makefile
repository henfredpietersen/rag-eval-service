.PHONY: test eval compare serve up

test:
	PYTHONPATH=. pytest -q

eval:
	PYTHONPATH=. python -m app.eval_cli

compare:
	PYTHONPATH=. python -m app.eval_cli --compare

serve:
	PYTHONPATH=. uvicorn app.main:app --reload --port 8000

up:
	docker compose up --build
