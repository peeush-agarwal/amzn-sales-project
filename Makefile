PHONY: all deploy clean

all: deploy

deploy:
	echo "Building Docker image..."

	docker compose build --no-cache && docker compose up -d

	echo "Deployment complete."

clean:
	echo "Cleaning up..."

	docker compose down

	echo "Cleanup complete."
