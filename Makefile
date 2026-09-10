.PHONY: install run test lint docker

install:
	python -m pip install -r requirements.txt

run:
	python app.py

lint:
	python -m ruff check app.py tests

test: lint
	python -m compileall -q app.py tests
	python -m pytest -q

docker:
	docker build -t sleep-intelligence .
