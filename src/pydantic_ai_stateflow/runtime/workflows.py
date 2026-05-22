"""``@sf.workflow`` decorator + workflow class registry.

Decorator stores ``(name, input_type, output_type)`` as ClassVars on
the decorated class and applies ``@Durable.dbos_class()`` so the
class behaves as a DBOSConfiguredInstance. Registration in
``_workflow_registry`` lets ``sf.create_app(workflows=[instance, ...])``
look up the metadata via ``type(instance)``.

See spec §B.2 and §C for full design.
"""
from __future__ import annotations

import inspect
from typing import Any, ClassVar, TypeVar, get_type_hints, overload

from pydantic import BaseModel

from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.runtime._kebab import kebab_case as _kebab_case

C = TypeVar("C", bound=type)

# Sentinel — apps that need a different name from kebab-derived can
# set ``_sf_workflow_name`` ClassVar before applying the decorator.
_NAME_ATTR = "_sf_workflow_name"
_INPUT_ATTR = "_sf_workflow_input"
_OUTPUT_ATTR = "_sf_workflow_output"
_BLOCKING_ATTR = "_sf_workflow_blocking"

# Process-wide registry: kebab-name → class.
_workflow_registry: dict[str, type] = {}


@overload
def workflow(cls: C) -> C: ...
@overload
def workflow(
    *,
    name: str | None = ...,
    input: type[BaseModel],
    output: type[BaseModel],
    blocking: bool = False,
) -> Any: ...


def workflow(
    cls: Any = None,
    /,
    *,
    name: str | None = None,
    input: type[BaseModel] | None = None,
    output: type[BaseModel] | None = None,
    blocking: bool = False,
) -> Any:
    """Register a class as a stateflow workflow.

    Required kwargs: ``input`` (pydantic BaseModel for ``run`` arg),
    ``output`` (pydantic BaseModel for ``run`` return). ``name``
    defaults to kebab-case of the class name (e.g. ``BrainstormFlow``
    → ``brainstorm-flow``).

    Effects on the decorated class:
    1. Applies ``@Durable.dbos_class()``.
    2. Validates the class has ``async def run(self, input: <input>)
       -> <output>`` (signature checked at decoration time).
    3. Stores ``(name, input, output, blocking)`` as ClassVars.
    4. Registers in the process-wide workflow registry under ``name``.

    ``blocking=True`` makes the auto-generated HTTP endpoint await
    the workflow and return the output model with HTTP 200 (default
    is fire-and-forget — returns ``{workflow_id, started_at}``).
    """

    def _apply(target: type) -> type:
        if input is None or output is None:
            raise TypeError(
                f"@sf.workflow on {target.__name__}: input= and output= are required keyword args",
            )
        resolved_name = name or _kebab_case(target.__name__)
        # Validate ``run`` exists. Read from the class __dict__ to avoid
        # method binding artefacts and to see the raw function DBOS
        # decorated.
        run_method = target.__dict__.get("run") or getattr(target, "run", None)
        if run_method is None or not callable(run_method):
            raise TypeError(
                f"@sf.workflow on {target.__name__}: class must define ``async def run(self, input)``",
            )
        # Validate ``run`` is wrapped by ``@Durable.workflow()`` (which
        # stacks ``@DBOS.workflow()``). DBOS marks decorated functions
        # with a ``dbos_func_decorator_info`` attribute that survives
        # ``functools.wraps`` chains. Missing → user forgot the
        # decorator; this would fail silently at runtime since DBOS
        # would never see the workflow.
        if not hasattr(run_method, "dbos_func_decorator_info"):
            raise TypeError(
                f"@sf.workflow on {target.__name__}: ``run`` must be "
                f"decorated with ``@Durable.workflow()`` "
                f"(missing dbos_func_decorator_info marker)",
            )
        # Validate run signature via type hints.
        try:
            hints = get_type_hints(run_method)
        except Exception:
            hints = {}
        # hints excludes 'self'; check the arg-named param matches ``input``.
        # We don't enforce parameter name; only that the return type matches.
        if hints.get("return") is not None and hints["return"] is not output:
            raise TypeError(
                f"@sf.workflow on {target.__name__}: ``run`` return type "
                f"{hints['return']} does not match output={output}",
            )
        # Validate the first non-self parameter annotation matches ``input``.
        # Use ``inspect.signature`` (not type hints) because ``from
        # __future__ import annotations`` can leave hints as strings
        # that ``get_type_hints`` would resolve via the function's
        # globals — but we want to be lenient: if the resolved hint
        # exists AND differs from ``input``, raise; if it's missing or
        # unresolvable, skip the check.
        try:
            sig = inspect.signature(run_method)
            params = [p for p in sig.parameters.values() if p.name != "self"]
        except (TypeError, ValueError):
            params = []
        if params:
            first = params[0]
            # Resolve string annotations through get_type_hints (already
            # tried above). Fall back to the raw annotation if not in
            # hints (e.g. forward refs that fail to resolve).
            annot = hints.get(first.name, first.annotation)
            if (
                annot is not inspect.Parameter.empty
                and not isinstance(annot, str)
                and annot is not input
            ):
                raise TypeError(
                    f"@sf.workflow on {target.__name__}: ``run`` first "
                    f"argument type {annot} does not match input={input}",
                )
        # Apply DBOS class decoration. Idempotent: if already applied,
        # this is a no-op (DBOS records the wrapping in a class attr).
        if not getattr(target, "_dbos_class_decorated", False):
            target = Durable.dbos_class()(target)
            try:
                target._dbos_class_decorated = True  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                pass
        # Store metadata as ClassVars (set on the class object directly).
        setattr(target, _NAME_ATTR, resolved_name)
        setattr(target, _INPUT_ATTR, input)
        setattr(target, _OUTPUT_ATTR, output)
        setattr(target, _BLOCKING_ATTR, blocking)
        # Register.
        existing = _workflow_registry.get(resolved_name)
        if existing is not None and existing is not target:
            raise ValueError(
                f"Duplicate @sf.workflow name {resolved_name!r}: "
                f"{existing.__module__}.{existing.__qualname__} and "
                f"{target.__module__}.{target.__qualname__}",
            )
        _workflow_registry[resolved_name] = target
        return target

    if cls is not None:
        # Bare @workflow — only allowed if class declares input/output via ClassVars.
        # (Not the intended usage; supported for symmetry with @stateflow_agent.)
        raise TypeError(
            "@sf.workflow must be called with input= and output= kwargs: "
            "@sf.workflow(input=MyIn, output=MyOut)",
        )
    return _apply


def get_workflow_class(name: str) -> type:
    """Look up a workflow class by its kebab-name."""
    try:
        return _workflow_registry[name]
    except KeyError as exc:
        raise KeyError(
            f"No workflow registered under {name!r}. "
            f"Did you forget @sf.workflow on the class?",
        ) from exc


def list_workflow_classes() -> dict[str, type]:
    """Snapshot of the workflow registry."""
    return dict(_workflow_registry)


def clear_workflow_registry() -> None:
    """For tests — drops all registrations."""
    _workflow_registry.clear()


def workflow_metadata(instance_or_cls: Any) -> tuple[str, type[BaseModel], type[BaseModel], bool]:
    """Read ``(name, input, output, blocking)`` from a @sf.workflow class or instance.

    Raises if the class is not @sf.workflow-decorated.
    """
    cls = instance_or_cls if isinstance(instance_or_cls, type) else type(instance_or_cls)
    if not hasattr(cls, _NAME_ATTR):
        raise TypeError(
            f"{cls.__name__} is not @sf.workflow-decorated — "
            f"cannot extract workflow metadata",
        )
    return (
        getattr(cls, _NAME_ATTR),
        getattr(cls, _INPUT_ATTR),
        getattr(cls, _OUTPUT_ATTR),
        getattr(cls, _BLOCKING_ATTR),
    )


__all__ = [
    "clear_workflow_registry",
    "get_workflow_class",
    "list_workflow_classes",
    "workflow",
    "workflow_metadata",
]
