"""Routing configuration schema: model alias bindings and per-stage rule definitions."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentflow.agents.base import Provider, validate_model_allowed
from agentflow.config.retirement import apply_retirements
from agentflow.task.profile import Stage


class ModelRef(BaseModel):
    """A concrete provider+model binding referenced by a role alias.

    For example, the alias 'implementation.lightweight'.
    """

    model_config = ConfigDict(extra="ignore")

    provider: str
    model: str

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        Provider.from_string(v)
        return v

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        validate_model_allowed(v)
        return v

    @property
    def provider_enum(self) -> Provider:
        """Resolve the string provider field to its Provider enum member."""
        return Provider.from_string(self.provider)


class RoleOverride(BaseModel):
    """User-specified provider/model overrides, scoped per pipeline stage.

    Unlike a single global override, each stage's `route()` call only sees an override when
    this stage is present in `overrides` -- a stage not listed here is routed normally.
    """

    model_config = ConfigDict(extra="ignore")

    overrides: dict[Stage, ModelRef] = Field(default_factory=dict)

    def for_stage(self, stage: Stage) -> ModelRef | None:
        """Return the override for `stage`, or None if this stage isn't overridden."""
        return self.overrides.get(stage)


class PlannerModels(BaseModel):
    """Model bindings for the planning role."""

    model_config = ConfigDict(extra="ignore")
    default: ModelRef
    architecture: ModelRef


class ImplementationModels(BaseModel):
    """Model bindings for the implementation role."""

    model_config = ConfigDict(extra="ignore")
    lightweight: ModelRef
    standard: ModelRef
    escalation: ModelRef


class ReviewModels(BaseModel):
    """Model bindings for the review role."""

    model_config = ConfigDict(extra="ignore")
    default: ModelRef
    deep: ModelRef
    architecture: ModelRef


class DocumentationModels(BaseModel):
    """Model bindings for the documentation role."""

    model_config = ConfigDict(extra="ignore")
    default: ModelRef


class ModelsConfig(BaseModel):
    """The routing.yaml `models:` section: role -> alias -> concrete provider/model binding."""

    model_config = ConfigDict(extra="ignore")

    planner: PlannerModels
    implementation: ImplementationModels
    review: ReviewModels
    documentation: DocumentationModels

    def resolve(self, alias: str, retired: Mapping[str, str] | None = None) -> ModelRef:
        """Resolve a dotted role alias such as 'implementation.lightweight' to a ModelRef.

        When `retired` (the global `models.retired` map, see config/retirement.py) is given,
        the resolved model is substituted through it -- e.g. a project still configured for a
        retired model transparently gets its replacement -- and the substituted model is
        re-validated against the V1 model pool before being returned.
        """
        category, sep, key = alias.partition(".")
        section = getattr(self, category, None) if sep else None
        ref = getattr(section, key, None) if section is not None else None
        if not isinstance(ref, ModelRef):
            raise ValueError(f"Undefined model alias: '{alias}'.")
        if retired:
            substituted = apply_retirements(ref.model, retired)
            if substituted != ref.model:
                validate_model_allowed(substituted)
                return ModelRef(provider=ref.provider, model=substituted)
        return ref


DEFAULT_MODELS_CONFIG = ModelsConfig(
    planner=PlannerModels(
        default=ModelRef(provider="anthropic", model="Claude Sonnet 5"),
        architecture=ModelRef(provider="anthropic", model="Claude Opus 5.5"),
    ),
    implementation=ImplementationModels(
        lightweight=ModelRef(provider="openai", model="GPT-6 Luna"),
        standard=ModelRef(provider="openai", model="GPT-6 Sol"),
        escalation=ModelRef(provider="anthropic", model="Claude Sonnet 5"),
    ),
    review=ReviewModels(
        default=ModelRef(provider="google", model="Gemini 3.8 Flash"),
        deep=ModelRef(provider="anthropic", model="Claude Sonnet 5"),
        architecture=ModelRef(provider="anthropic", model="Claude Opus 5.5"),
    ),
    documentation=DocumentationModels(
        default=ModelRef(provider="google", model="Gemini 3.8 Flash"),
    ),
)


class ComplexityRuleWhen(BaseModel):
    """Match condition for a project-specific implementation routing rule."""

    model_config = ConfigDict(extra="ignore")
    complexity: str

    @field_validator("complexity")
    @classmethod
    def _validate_complexity(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        cleaned = v.strip().lower()
        if cleaned not in allowed:
            raise ValueError(
                f"Invalid complexity condition: '{v}'. Expected one of {sorted(allowed)}."
            )
        return cleaned


class ComplexityRule(BaseModel):
    """A single first-match-wins project rule mapping a complexity level to a model alias."""

    model_config = ConfigDict(extra="ignore")
    id: str
    when: ComplexityRuleWhen
    use: str


class PlanningRouting(BaseModel):
    """Planning-stage routing rules (routing.yaml `routing.planning:` section)."""

    model_config = ConfigDict(extra="ignore")
    architecture_if_any: list[str] = Field(default_factory=list)
    default: str = "planner.default"
    architecture: str = "planner.architecture"


class ImplementationRouting(BaseModel):
    """Implementation-stage routing rules (routing.yaml `routing.implementation:` section)."""

    model_config = ConfigDict(extra="ignore")
    force_standard_if_any: list[str] = Field(default_factory=list)
    rules: list[ComplexityRule] = Field(default_factory=list)


class ReviewRouting(BaseModel):
    """Review-stage routing rules (routing.yaml `routing.review:` section)."""

    model_config = ConfigDict(extra="ignore")
    deep_if_any: list[str] = Field(default_factory=list)
    architecture_if: dict[str, bool] = Field(default_factory=dict)
    default: str = "review.default"
    deep: str = "review.deep"
    architecture: str = "review.architecture"
    medium_is_mandatory: bool = False


class DocumentationRouting(BaseModel):
    """Documentation-stage routing rules (routing.yaml `routing.documentation:` section)."""

    model_config = ConfigDict(extra="ignore")
    default: str = "documentation.default"


class RoutingRulesConfig(BaseModel):
    """The routing.yaml `routing:` section: per-stage rule sets."""

    model_config = ConfigDict(extra="ignore")

    planning: PlanningRouting = Field(default_factory=PlanningRouting)
    implementation: ImplementationRouting = Field(default_factory=ImplementationRouting)
    review: ReviewRouting = Field(default_factory=ReviewRouting)
    documentation: DocumentationRouting = Field(default_factory=DocumentationRouting)


DEFAULT_ROUTING_RULES = RoutingRulesConfig(
    planning=PlanningRouting(
        architecture_if_any=[
            "architecture_change",
            "new_service",
            "new_datastore",
            "security_boundary_change",
            "payment",
            "cross_service_ownership_change",
        ],
    ),
    implementation=ImplementationRouting(
        force_standard_if_any=[
            "authentication",
            "authorization",
            "data_ownership",
            "concurrency",
            "idempotency",
            "transaction_logic",
            "destructive_schema_change",
            "payment",
            "security_boundary_change",
            "ai_or_rag",
        ],
        rules=[
            ComplexityRule(
                id="low-complexity",
                when=ComplexityRuleWhen(complexity="low"),
                use="implementation.lightweight",
            ),
            ComplexityRule(
                id="medium-complexity",
                when=ComplexityRuleWhen(complexity="medium"),
                use="implementation.standard",
            ),
            ComplexityRule(
                id="high-complexity",
                when=ComplexityRuleWhen(complexity="high"),
                use="implementation.standard",
            ),
        ],
    ),
    review=ReviewRouting(
        deep_if_any=[
            "authentication",
            "authorization",
            "payment",
            "concurrency",
            "transaction_logic",
            "data_ownership",
            "destructive_schema_change",
        ],
        architecture_if={"architecture_change": True},
    ),
)


def all_referenced_aliases(routing_rules: RoutingRulesConfig) -> list[str]:
    """Collect every model alias referenced by a RoutingRulesConfig, for validation."""
    aliases = [
        routing_rules.planning.default,
        routing_rules.planning.architecture,
        routing_rules.review.default,
        routing_rules.review.deep,
        routing_rules.review.architecture,
        routing_rules.documentation.default,
    ]
    aliases.extend(rule.use for rule in routing_rules.implementation.rules)
    return aliases
