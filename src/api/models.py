"""Pydantic models for API request/response schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    """Request schema for discount prediction with raw product data."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_name": "Wayona Nylon Braided USB Cable",
                "category": "Computers&Accessories",
                "actual_price": "₹1,099",
                "rating": 4.2,
                "rating_count": "24,269",
                "about_product": "High Compatibility : Compatible With iPhone...",
                "user_name": "Manav",
                "review_title": "Satisfied",
                "review_content": "Looks durable Charging is fine too",
            }
        }
    )

    product_id: Optional[str] = Field(
        None,
        description="Product ID (optional)",
        json_schema_extra={"example": "B07JW9H4J1"},
    )

    product_name: str = Field(
        ...,
        description="Name of the product",
        json_schema_extra={"example": "Wayona Nylon Braided USB Cable"},
    )

    category: str = Field(
        ...,
        description="Product category",
        json_schema_extra={"example": "Computers&Accessories"},
    )

    actual_price: str = Field(
        ...,
        description="Actual price of the product (can include currency symbols)",
        json_schema_extra={"example": "₹1,099"},
    )

    rating: float = Field(
        ...,
        description="Product rating",
        json_schema_extra={"example": 4.2},
    )

    rating_count: str = Field(
        ...,
        description="Number of ratings",
        json_schema_extra={"example": "24,269"},
    )

    about_product: str = Field(
        ...,
        description="Product description",
        json_schema_extra={
            "example": "High Compatibility : Compatible With iPhone 12, 11..."
        },
    )

    user_id: Optional[str] = Field(
        None,
        description="User ID (optional)",
        json_schema_extra={"example": "AG3D6O4STAQKAY2UVGEUV46KN35Q"},
    )

    user_name: str = Field(
        ...,
        description="Reviewer user name",
        json_schema_extra={"example": "Manav"},
    )

    review_id: Optional[str] = Field(
        None,
        description="Review ID (optional)",
        json_schema_extra={"example": "R1EXAMPLE12345"},
    )

    review_title: str = Field(
        ...,
        description="Review title",
        json_schema_extra={"example": "Satisfied"},
    )

    review_content: str = Field(
        ...,
        description="Review content/text",
        json_schema_extra={"example": "Looks durable Charging is fine too"},
    )

    img_link: Optional[str] = Field(
        None,
        description="Image link of the product (optional)",
        json_schema_extra={"example": "https://example.com/image.jpg"},
    )

    product_link: Optional[str] = Field(
        None,
        description="Product link/URL (optional)",
        json_schema_extra={"example": "https://example.com/product/12345"},
    )


class PredictionResponse(BaseModel):
    """Response schema for discount prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "discount_percentage_numeric": 15.5,
                "discount_percentage": "15.5%",
                "model_name": "random_forest",
                "model_version": "abc123def456",
            }
        }
    )

    prediction_decimal: float = Field(
        ...,
        description="Predicted discount in decimal",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.155},
    )

    prediction_percent: float = Field(
        ...,
        description="Predicted discount percentage",
        ge=0.0,
        le=100.0,
        json_schema_extra={"example": 15.5},
    )

    model_name: Optional[str] = Field(
        None,
        description="Name of the model used for prediction",
        json_schema_extra={"example": "random_forest"},
    )

    model_version: Optional[str] = Field(
        None,
        description="Version or run ID of the model",
        json_schema_extra={"example": "abc123def456"},
    )

    confidence_interval: Optional[Dict[str, float]] = Field(
        None,
        description="Confidence interval for prediction (if available)",
        json_schema_extra={"example": {"lower": 12.0, "upper": 18.0}},
    )


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(
        ..., description="Health status", json_schema_extra={"example": "healthy"}
    )

    model_loaded: bool = Field(
        ..., description="Whether model is loaded", json_schema_extra={"example": True}
    )

    rag_ready: Optional[bool] = Field(
        None,
        description="Whether the RAG assistant is initialised",
        json_schema_extra={"example": True},
    )

    model_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Model metadata",
        json_schema_extra={
            "example": {
                "model_name": "random_forest",
                "run_id": "abc123",
                "metrics": {"val_rmse": 0.05},
            }
        },
    )

    version: str = Field(
        ..., description="API version", json_schema_extra={"example": "1.0.0"}
    )


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    detail: str = Field(
        ...,
        description="Error message",
        json_schema_extra={"example": "Invalid input: features contain NaN values"},
    )

    error_type: Optional[str] = Field(
        None,
        description="Type of error",
        json_schema_extra={"example": "ValidationError"},
    )


class RAGContext(BaseModel):
    """Context snippet returned alongside RAG answers."""

    sequence_id: int = Field(
        ...,
        description="Sequence ID of the context snippet",
        json_schema_extra={"example": 1},
    )
    product_id: str = Field(
        ...,
        description="Product ID",
        json_schema_extra={"example": "B01LZ5X5Z8"},
    )
    category: str = Field(
        ...,
        description="Product category",
        json_schema_extra={
            "example": "Electronics|Mobiles&Accessories|MobileAccessories|Chargers|WallChargers"
        },
    )
    rating: float = Field(
        ...,
        description="Product rating",
        json_schema_extra={"example": 4.5},
    )
    content: str = Field(
        ...,
        description="Retrieved text snippet",
        json_schema_extra={"example": "Product: Wireless Mouse ..."},
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the source",
        json_schema_extra={"example": {"product_name": "Wireless Mouse"}},
    )


class QuestionAnswerRequest(BaseModel):
    """Request payload for the marketing assistant."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Provide details about 'MI Usb Type-C Cable Smartphone (Black)'?",
                "top_k": 2,
            }
        }
    )

    question: str = Field(
        ..., description="User question for the assistant", min_length=3
    )
    top_k: Optional[int] = Field(
        None,
        description="How many context passages to retrieve",
        ge=1,
        le=5,
    )


class QuestionAnswerResponse(BaseModel):
    """Response payload for RAG based answers."""

    answer: str = Field(
        ...,
        description="Model generated answer",
        json_schema_extra={"example": "The discount is 32%."},
    )
    contexts: List[RAGContext] = Field(
        default_factory=list,
        description="Supporting contexts that grounded the answer",
    )
