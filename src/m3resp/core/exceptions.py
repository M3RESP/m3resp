"""M3Resp-specific exceptions."""


class M3RespError(Exception):
    """Base exception for M3Resp errors."""


class OptionalDependencyError(M3RespError, ImportError):
    """Raised when an optional modality package is not installed."""


class MissingModalityDataError(M3RespError):
    """Raised when a workflow step needs missing modality data."""


class VariantAlreadyExistsError(M3RespError):
    """Raised when a preprocessing variant would silently overwrite an existing one."""


class UnsupportedWorkflowError(M3RespError):
    """Raised when an adapter cannot infer the requested workflow operation."""


class UnresolvedChannelError(M3RespError, LookupError):
    """Raised when a requested channel is not present in a recording, or when
    it cannot be worked out which channel is meant.

    Distinct from `MissingModalityDataError`: the modality was loaded, but the
    recording does not carry that particular channel - a ventilator export
    without an esophageal pressure, or one whose vendor naming is not yet
    registered.
    """


class PipelineError(M3RespError):
    """Base exception for declarative pipeline errors."""


class UnknownStepError(PipelineError):
    """Raised when a spec references a step name that is not registered."""


class PipelineSpecError(PipelineError):
    """Raised when a pipeline spec is malformed or fails static validation."""


class StepMetadataError(PipelineError):
    """Raised when a step's registered GUI/discovery metadata is inconsistent."""


class UnknownPipelineError(PipelineError):
    """Raised when ``M3Session.run_pipeline`` references an unregistered name."""


class DataModelValidationError(M3RespError):
    """Raised by ``export_store`` when the data model store fails its checks,
    so nothing is written. ``problems`` lists every problem found, in the
    words of ``validate_store``."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        count = len(self.problems)
        noun = "problem" if count == 1 else "problems"
        listed = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(
            f"The data model store has {count} {noun}, so nothing was written:\n"
            f"{listed}\n"
            "Fix them, or pass validate=False to export_store to save anyway."
        )


class UnsynchronizedDataWarning(UserWarning):
    """Warned when a step compares recordings that were never placed on a
    shared clock.

    Filter it with ``warnings.simplefilter("ignore", UnsynchronizedDataWarning)``
    - or better, call ``M3Session.synchronize_raw_modalities`` or
    ``M3Session.skip_synchronization``.
    """
