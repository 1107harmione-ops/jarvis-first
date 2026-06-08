.PHONY: install run test test-verbose lint clean

## Install dependencies
install:
	pip install -r requirements.txt

## Start development server
run:
	./start.sh

## Run tests
test:
	python -m pytest $(file) -v --asyncio-mode=auto

## Run tests with verbose output
test-verbose:
	python -m pytest -v --asyncio-mode=auto --tb=long

## Lint with ruff
lint:
	ruff check app/ tests/

## Clean cache and test artifacts
clean:
	rm -rf __pycache__ .pytest_cache *.egg-info
	rm -rf app/**/__pycache__ tests/__pycache__
	rm -f *.db test_*.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

## Build and start all services
docker-up:
	docker compose up --build -d

## Stop all services
docker-down:
	docker compose down

## View logs
docker-logs:
	docker compose logs -f

## Run RQ worker directly (without Docker)
worker:
	python -m app.reminders.worker

## Show help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Development:"
	@echo "  install       Install Python dependencies"
	@echo "  run           Start development server"
	@echo "  test          Run tests (file=tests/test_*.py to filter)"
	@echo "  test-verbose  Run all tests with detailed output"
	@echo "  lint          Check code style"
	@echo "  clean         Remove caches and test DB"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up     Build and start all services"
	@echo "  docker-down   Stop all services"
	@echo "  docker-logs   View container logs"
	@echo ""
	@echo "Background:"
	@echo "  worker        Run RQ reminder worker"
