"""AgentFlow domain exceptions."""


class AgentFlowError(Exception):
    """Base exception for all AgentFlow errors."""


class ConfigurationError(AgentFlowError):
    """Raised when configuration loading or validation fails."""


class ProjectNotFoundError(AgentFlowError):
    """Raised when a specified or discovered project repository cannot be found."""


class ProcessExecutionError(AgentFlowError):
    """Raised when safe subprocess execution fails unexpectedly."""


class PersistenceError(AgentFlowError):
    """Raised when database initialization, migration, or query execution fails."""


class AdapterError(AgentFlowError):
    """Base exception for all agent adapter errors."""


class UnsupportedCapabilityError(AdapterError):
    """Raised when an adapter is requested to perform an operation it does not support."""


class AdapterNotFoundError(AdapterError):
    """Raised when an adapter for a specified provider is not found in the registry."""


class CLIExecutionError(AdapterError):
    """Raised when provider CLI execution fails."""


class ResponseParsingError(AdapterError):
    """Raised when provider CLI output cannot be parsed."""


class StructuredParsingError(ResponseParsingError):
    """Raised when structured output (JSON / schema) parsing or extraction fails."""
