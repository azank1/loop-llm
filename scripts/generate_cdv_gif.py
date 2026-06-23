#!/usr/bin/env python3
"""Generate img/agent_loop.gif from CDV MCP demo JSON (Cursor-style panels).

Requires Pillow: pip install pillow
Run from repo root: python scripts/generate_cdv_gif.py
"""
from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from PIL import Image, ImageDraw, ImageFont

from loopllm.mcp_server import _tool_loop_end, _tool_loop_start, _tool_loop_step

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "img" / "agent_loop.gif"

BG = (30, 30, 30)
PANEL = (37, 37, 38)
BORDER = (60, 60, 60)
TITLE = (200, 200, 200)
TOOL = (78, 201, 176)
KEY = (156, 220, 254)
STR = (206, 145, 120)
NUM = (181, 206, 168)
TEXT = (212, 212, 212)
MUTED = (140, 140, 140)


def _mock_ctx(score: float) -> MagicMock:
    mock_ctx = MagicMock()
    mock_response = MagicMock()
    mock_response.content.text = json.dumps({
        "score": score,
        "passed": score >= 0.7,
        "deficiencies": [],
        "feedback": "",
    })
    mock_ctx.sample = AsyncMock(return_value=mock_response)
    return mock_ctx


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "consola.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


async def _collect_frames() -> list[tuple[str, dict]]:
    start = json.loads(
        _tool_loop_start(
            goal="make the failing test pass",
            task_type="bugfix",
            required_patterns=["passed"],
        )
    )
    sid = start["session_id"]
    step1 = json.loads(
        await _tool_loop_step(
            session_id=sid,
            step_output="pytest: 3 FAILED, 12 passed",
            ctx=_mock_ctx(0.55),
        )
    )
    step2 = json.loads(
        await _tool_loop_step(
            session_id=sid,
            step_output="pytest: 42 passed, 0 failed",
            ctx=_mock_ctx(0.92),
        )
    )
    return [
        ("loopllm_loop_start", start),
        ("loopllm_loop_step", step1),
        ("loopllm_loop_step", step2),
    ]


def _highlight_json(data: dict) -> list[tuple[str, tuple[int, int, int]]]:
    """Render JSON lines with simple syntax coloring."""
    raw = json.dumps(data, indent=2)
    lines: list[tuple[str, tuple[int, int, int]]] = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('"channel_a_score"') or stripped.startswith('"channel_b_score"'):
            lines.append((line, TOOL))
        elif stripped.startswith('"score_source"') or stripped.startswith('"decision"'):
            lines.append((line, KEY))
        elif stripped.startswith('"score"'):
            lines.append((line, NUM))
        elif '"' in stripped and ":" in stripped:
            lines.append((line, STR if stripped.endswith(',') else TEXT))
        else:
            lines.append((line, TEXT))
    return lines


def _render_frame(
    tool_name: str,
    payload: dict,
    width: int = 800,
    height: int = 520,
) -> Image.Image:
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    font = _font(13)
    title_font = _font(14)

    draw.rectangle((12, 12, width - 12, height - 12), fill=PANEL, outline=BORDER, width=1)
    draw.text((24, 20), "Cursor Agent — MCP tool call", fill=MUTED, font=title_font)
    draw.text((24, 44), tool_name, fill=TOOL, font=title_font)

    y = 76
    for line, color in _highlight_json(payload):
        wrapped = textwrap.wrap(line, width=92) or [""]
        for wline in wrapped:
            draw.text((28, y), wline, fill=color, font=font)
            y += 17
            if y > height - 24:
                break

    draw.text((24, height - 28), "Conservative Dual-Verify · loopllm 0.7.0", fill=MUTED, font=font)
    return img


async def main() -> None:
    frames_data = await _collect_frames()
    images = [_render_frame(name, data) for name, data in frames_data]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        OUT,
        save_all=True,
        append_images=images[1:],
        duration=3500,
        loop=0,
        optimize=True,
    )
    size_kb = OUT.stat().st_size // 1024
    print(f"Wrote {OUT} ({size_kb} KB, {len(images)} frames)")


if __name__ == "__main__":
    asyncio.run(main())
