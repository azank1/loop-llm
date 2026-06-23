#!/usr/bin/env python3
"""CDV agent-loop demo via MCP tool handlers (no live MCP server required).

Simulates the Cursor Agent CDV script (see examples/demo_cdv_mcp.py).
  loop_start → loop_step (failing tests) → loop_step (passing) → loop_end

Run: python examples/demo_cdv_mcp.py
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from loopllm.mcp_server import _tool_loop_end, _tool_loop_start, _tool_loop_step


def _mock_ctx(channel_b_score: float, feedback: str = "") -> MagicMock:
    """MCP Context mock returning a fixed Channel B critic score."""
    mock_ctx = MagicMock()
    mock_response = MagicMock()
    mock_response.content.text = json.dumps({
        "score": channel_b_score,
        "passed": channel_b_score >= 0.7,
        "deficiencies": [] if channel_b_score >= 0.7 else ["not done yet"],
        "feedback": feedback or f"critic score {channel_b_score}",
    })
    mock_ctx.sample = AsyncMock(return_value=mock_response)
    return mock_ctx


async def main() -> None:
    print("=== loopllm_loop_start ===")
    start = json.loads(
        _tool_loop_start(
            goal="make the failing test pass",
            task_type="bugfix",
            required_patterns=["passed"],
        )
    )
    session_id = start["session_id"]
    print(json.dumps(start, indent=2))

    print("\n=== loopllm_loop_step (step 1) ===")
    step1 = json.loads(
        await _tool_loop_step(
            session_id=session_id,
            step_output="pytest: 3 FAILED, 12 passed",
            ctx=_mock_ctx(0.55, "tests still failing"),
        )
    )
    print(json.dumps(step1, indent=2))

    print("\n=== loopllm_loop_step (step 2) ===")
    step2 = json.loads(
        await _tool_loop_step(
            session_id=session_id,
            step_output="pytest: 42 passed, 0 failed",
            ctx=_mock_ctx(0.92, "all tests green"),
        )
    )
    print(json.dumps(step2, indent=2))

    print("\n=== loopllm_loop_end ===")
    end = json.loads(_tool_loop_end(session_id))
    print(json.dumps(end, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
