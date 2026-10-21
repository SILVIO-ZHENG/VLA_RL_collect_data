"""Stable application errors shared by HTTP and internal task use cases."""
from http import HTTPStatus


class ValidationError(ValueError):
    def __init__(self, message: str, *, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = int(status)


class TaskError(ValidationError):
    def __init__(self, reason_code: str, message: str, status: int = HTTPStatus.CONFLICT):
        super().__init__(message, status=status)
        self.reason_code = reason_code
