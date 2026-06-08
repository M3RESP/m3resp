"""M3Resp-specific exceptions."""


class M3RespError(Exception):
    """Base exception for M3Resp errors."""


class OptionalDependencyError(M3RespError, ImportError):
    """Raised when an optional modality package is not installed."""


class MissingModalityDataError(M3RespError):
    """Raised when a workflow step needs missing modality data."""


class UnsupportedWorkflowError(M3RespError):
    """Raised when an adapter cannot infer the requested workflow operation."""
