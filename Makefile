PHONY: all setup deploy clean

all: setup deploy

setup:
	echo "Setting up the environment..."

	uv venv && source .venv/bin/activate && uv sync

	echo "Setup complete."

deploy:
	echo "Running DVC stages..."

	source .venv/bin/activate && python -m dvc repro

	echo "Building Docker image..."

	docker compose up -d --build

	echo "Deployment complete."

clean:
	echo "Cleaning up..."

	docker compose down

	echo "Cleanup complete."
