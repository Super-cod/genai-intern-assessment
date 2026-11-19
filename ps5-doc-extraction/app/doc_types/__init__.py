"""Registry of supported document types. Add a new type by registering another DocumentType."""
from .base import DocumentType, Issue, RawField
from .invoice import INVOICE

REGISTRY: dict[str, DocumentType] = {INVOICE.name: INVOICE}

__all__ = ["DocumentType", "Issue", "RawField", "REGISTRY"]
