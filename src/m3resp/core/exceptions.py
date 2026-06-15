"""M3Resp-specific exceptions."""


class M3RespError(Exception):
    """Base exception for M3Resp errors."""


class OptionalDependencyError(M3RespError, ImportError):
    """Raised when an optional modality package is not installed."""


class MissingModalityDataError(M3RespError):
    """Raised when a workflow step needs missing modality data."""


class UnsupportedWorkflowError(M3RespError):
    """Raised when an adapter cannot infer the requested workflow operation."""


class PipelineError(M3RespError):
    """Base exception for declarative pipeline errors."""


class UnknownStepError(PipelineError):
    """Raised when a spec references a step name that is not registered."""


class PipelineSpecError(PipelineError):
    """Raised when a pipeline spec is malformed or fails static validation."""
