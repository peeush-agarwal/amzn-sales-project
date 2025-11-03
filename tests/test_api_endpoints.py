from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.app as api_app


class FakePredictor:
    """A minimal fake predictor used for testing API endpoints.

    It exposes the small subset of the real predictor API used by the
    endpoints: is_ready, get_model_info, predict_from_raw and simple metrics.
    """

    def __init__(self):
        self.prediction_count = 42
        self.total_prediction_time = 1.23

    def is_ready(self):
        return True

    def get_model_info(self):
        return {
            "model_name": "fake_model",
            "metrics": {"rmse": 0.5},
            "avg_prediction_time_ms": (
                self.total_prediction_time / self.prediction_count * 1000
            ),
        }

    def predict_from_raw(self, payload):
        # Return a deterministic fake prediction
        return 0.15


# Inject fake predictor into the existing api module (avoids running startup)
api_app._predictor = FakePredictor()

# Build a small Test app that re-uses the route handlers from the real app but
# doesn't execute the real lifespan/startup logic (which would try to load
# models). This keeps tests fast and hermetic.
test_app = FastAPI()
test_app.add_api_route("/", api_app.root, methods=["GET"])
test_app.add_api_route("/health", api_app.health_check, methods=["GET"])
test_app.add_api_route("/model/info", api_app.model_info, methods=["GET"])
test_app.add_api_route("/metrics", api_app.metrics, methods=["GET"])
test_app.add_api_route("/predict_discount", api_app.predict_discount, methods=["POST"])

client = TestClient(test_app)


def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    payload = r.json()
    assert "Welcome to" in payload.get("message", "")


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["model_loaded"] is True
    assert j["status"] == "healthy"


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    j = r.json()
    assert j["api_version"] == api_app.APP_VERSION
    assert j["predictions_served"] == api_app._predictor.prediction_count


def test_predict_discount_endpoint():
    payload = {
        "product_name": "Wayona Nylon Braided USB Cable",
        "category": "Computers&Accessories",
        "actual_price": "1099",
        "rating": "4.2",
        "rating_count": "24269",
        "about_product": "High Compatibility : Compatible With iPhone...",
        "user_name": "Manav",
        "review_title": "Satisfied",
        "review_content": "Looks durable Charging is fine too",
    }

    r = client.post("/predict_discount", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "prediction_decimal" in j and "prediction_percent" in j
    assert abs(j["prediction_decimal"] - 0.15) < 1e-8
