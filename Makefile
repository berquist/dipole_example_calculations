test:
	pytest -v

test-cov:
	pytest -v --cov=. --cov-report=html
