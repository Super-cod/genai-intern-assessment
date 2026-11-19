"""Domain exceptions mapped to HTTP status codes by the API layer."""


class RAGError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class UnsupportedFileError(RAGError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(RAGError):
    status_code = 413
    code = "file_too_large"


class EmptyDocumentError(RAGError):
    status_code = 422
    code = "no_extractable_text"


class CorruptDocumentError(RAGError):
    status_code = 422
    code = "corrupt_document"


class EmptyIndexError(RAGError):
    status_code = 409
    code = "no_documents_indexed"


class DocumentNotFoundError(RAGError):
    status_code = 404
    code = "document_not_found"


class ServiceUnavailableError(RAGError):
    status_code = 503
    code = "model_service_unavailable"
