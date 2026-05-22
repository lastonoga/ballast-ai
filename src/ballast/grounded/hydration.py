from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ballast.grounded._spec import FieldRole, OutputSpec
from ballast.grounded.ref import Ref


class HydrationMap:
    """Walks an output value and replaces Ref instances with entities."""

    def __init__(self, spec: OutputSpec) -> None:
        self._spec = spec

    async def hydrate(self, value: BaseModel, *, repos: dict[type, Any]) -> dict[str, Any]:
        """Return a dict-shaped hydrated view of `value`.

        Returns a dict (not a typed BaseModel) so consumers don't need a
        separate hydrated-output type per pattern. Repos must be indexed
        by entity TYPE (per spec 4A.0.4).
        """
        return await _hydrate_model(value, self._spec, repos)


async def _hydrate_model(
    obj: BaseModel, spec: OutputSpec, repos: dict[type, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fspec in spec.fields.items():
        value = getattr(obj, name)
        match fspec.role:
            case FieldRole.REF:
                if not isinstance(value, Ref):
                    out[name] = value
                else:
                    target = fspec.target_type
                    if target is None:  # defensive; scan_output always sets this for REF
                        out[name] = value
                    else:
                        repo = repos.get(target)
                        if repo is None:
                            raise KeyError(
                                f"hydrate: missing repo for {target.__name__}"
                            )
                        out[name] = await value.hydrate(repo)

            case FieldRole.LIST_REF:
                target = fspec.target_type
                if target is None:
                    out[name] = list(value)
                else:
                    repo = repos.get(target)
                    if repo is None:
                        raise KeyError(
                            f"hydrate: missing repo for {target.__name__}"
                        )
                    out[name] = [
                        await r.hydrate(repo) if isinstance(r, Ref) else r for r in value
                    ]

            case FieldRole.OPTIONAL_REF:
                if value is None:
                    out[name] = None
                elif isinstance(value, Ref):
                    target = fspec.target_type
                    if target is None:
                        out[name] = value
                    else:
                        repo = repos.get(target)
                        if repo is None:
                            raise KeyError(
                                f"hydrate: missing repo for {target.__name__}"
                            )
                        out[name] = await value.hydrate(repo)
                else:
                    out[name] = value

            case FieldRole.NESTED:
                assert fspec.nested_spec is not None
                out[name] = await _hydrate_model(value, fspec.nested_spec, repos)

            case FieldRole.LIST_NESTED:
                assert fspec.nested_spec is not None
                out[name] = [
                    await _hydrate_model(v, fspec.nested_spec, repos) for v in value
                ]

            case _:
                out[name] = value
    return out
