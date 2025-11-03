"""Simple script to start the Amazon Discount Predictor API."""

import sys
from pathlib import Path

src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))


if __name__ == "__main__":
    import uvicorn
    from api.app import _config

    base_url = f"http://{_config['api']['host']}:{_config['api']['port']}"

    print("=" * 80)
    print(f"🚀 Starting {_config['api']['name']} v{_config['api']['version']}")
    print("=" * 80)
    print(f"📍 URL: {base_url}")
    print(f"📚 Docs: {base_url}/docs")
    print(f"❤️  Health: {base_url}/health")
    print("=" * 80)

    uvicorn.run(
        "api.app:app",
        host=_config["api"]["host"],
        port=_config["api"]["port"],
        workers=_config["api"].get("workers", 1),
        log_level=_config["logs"]["level"].lower(),
        reload=False,
    )
