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
	poetry run isort src/agent_tools
	poetry run black src/agent_tools

lint:
	poetry run black --check src/agent_tools
	poetry run flake8 src/agent_tools --max-line-length=120
	poetry run mypy src/agent_tools --ignore-missing-imports --check-untyped-defs 
