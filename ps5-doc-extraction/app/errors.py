"""Domain exceptions; the API layer maps them to HTTP status codes."""


class ExtractionError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class EmptyFileError(ExtractionError):
    status_code = 400
    code = "empty_file"


class FileTooLargeError(ExtractionError):
    status_code = 413
    code = "file_too_large"


class UnsupportedFormatError(ExtractionError):
    status_code = 415
    code = "unsupported_format"


class UnknownDocumentTypeError(ExtractionError):
    status_code = 400
    code = "unknown_document_type"


class CorruptFileError(ExtractionError):
    status_code = 422
    code = "corrupt_file"


class EncryptedFileError(ExtractionError):
    status_code = 422
    code = "encrypted_file"


class TooManyPagesError(ExtractionError):
    status_code = 422
    code = "too_many_pages"


class NoTextError(ExtractionError):
    status_code = 422
    code = "no_extractable_text"


class ModelUnavailableError(ExtractionError):
    status_code = 503
    code = "model_unavailable"


class ModelOutputError(ExtractionError):
    status_code = 502
    code = "model_output_invalid"
