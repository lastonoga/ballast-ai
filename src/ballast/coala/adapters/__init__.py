"""Runtime adapters for CoALAUnit — workflow, tool, capability."""
from ballast.coala.adapters.tool import as_tool
from ballast.coala.adapters.workflow import as_workflow

__all__ = ["as_tool", "as_workflow"]
