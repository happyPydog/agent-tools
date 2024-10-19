test:
	poetry run pytest -vv tests \
	--cov=. \
	--cov-report=term \
	--cov-report=xml:coverage.xml

e2e-test:
	poetry run pytest -vv e2e

install:
	poetry install

format:
	poetry run isort src/ragtor
	poetry run black src/ragtor

lint:
	poetry run isort --check src/ragtor
	poetry run black --check src/ragtor
	poetry run flake8 src/ragtor --max-line-length=119
	poetry run mypy src/ragtor --ignore-missing-imports --check-untyped-defs 
