"""AgentFlow domain exceptions."""


class AgentFlowError(Exception):
    """Base exception for all AgentFlow errors."""


class ConfigurationError(AgentFlowError):
    """Raised when configuration loading or validation fails."""


class ProjectNotFoundError(AgentFlowError):
    """Raised when a specified or discovered project repository cannot be found."""


class InitError(AgentFlowError):
    """Raised when `agentflow init` cannot safely generate a starter project profile."""


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


class WorkflowError(AgentFlowError):
    """Base exception for all workflow orchestration errors."""


class InvalidStateTransitionError(WorkflowError):
    """Raised when a workflow attempts an illegal state transition."""


class PlanningBlockedError(WorkflowError):
    """Raised when interactive planning cannot proceed and the run must be marked BLOCKED."""


class ImplementationBlockedError(WorkflowError):
    """Raised when implementation cannot proceed and the run must be marked BLOCKED."""


class ReviewBlockedError(WorkflowError):
    """Raised when review cannot proceed or mandatory findings cannot be resolved."""


class DocumentationBlockedError(WorkflowError):
    """Raised when the documentation workflow cannot proceed."""


class ResumeError(WorkflowError):
    """Raised when a run cannot be safely resumed (terminal state, missing artifacts, etc.)."""


class RunLockedError(WorkflowError):
    """Raised when a run is already controlled by another live process."""


class WorktreeError(AgentFlowError):
    """Base exception for all Git worktree management errors."""


class WorktreeLockedError(WorktreeError):
    """Raised when a worktree is already locked by another writer."""
