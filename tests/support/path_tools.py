# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Pi-shaped path tools bound to one rooted environment.

Production composes these tools through the answer tool registry; tests that
exercise the tools themselves assemble them here.
"""

from dlightrag.engine.agent.environment import AccessScheduler, SearchToolchain
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.tools.contracts import AgentTool
from dlightrag.engine.agent.tools.files import (
    bash_tool,
    edit_tool,
    find_tool,
    grep_tool,
    ls_tool,
    read_tool,
    view_tool,
    write_tool,
)


def path_tools(
    environment: ExecutionEnvironment,
    *,
    scheduler: AccessScheduler,
    search_toolchain: SearchToolchain | None = None,
) -> list[AgentTool]:
    toolchain = search_toolchain or SearchToolchain(fd="fd", ripgrep="rg")
    return [
        read_tool(environment, scheduler),
        view_tool(environment, scheduler),
        bash_tool(environment, scheduler),
        edit_tool(environment, scheduler),
        write_tool(environment, scheduler),
        grep_tool(environment, scheduler, search_toolchain=toolchain),
        find_tool(environment, scheduler, search_toolchain=toolchain),
        ls_tool(environment, scheduler),
    ]
