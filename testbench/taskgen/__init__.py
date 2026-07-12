"""taskgen — declarative kernel-task generation.

A `TaskSpec` (spec.py) describes one (op, phase) kernel; `write_task` materializes it.
Families live in families/*.py grouped by kernel type; registry.all_specs() collects them.
gen_tasks.py is a thin runner over this package.
"""
from .spec import TaskSpec, write_task
from .registry import all_specs

__all__ = ["TaskSpec", "write_task", "all_specs"]
