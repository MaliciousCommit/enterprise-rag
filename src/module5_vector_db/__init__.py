# src/module5_vector_db/__init__.py
from src.module5_vector_db.collection_manager import (
    create_production_collection,
    create_payload_indexes,
    get_collection_stats,
)

__all__ = [
    "create_production_collection",
    "create_payload_indexes",
    "get_collection_stats",
]
