#!/usr/bin/env python3
"""CDV loop audit script — exercises the full v0.8 MCP tool chain without a live server.

Covers:
  1a. loopllm_intercept: prompt quality scoring + recall_available flag
  1b. loopllm_loop_start: suggested budget, similar_episodes injection
  1c. loopllm_loop_step: CDV scoring (channel_a_only without MCP sampling)
  1d. loopllm_loop_end: episode persistence + Bayesian learning
  1e. Recovery: snapshot → drop controller → hydrate → loop_resume → continue
  1f. loopllm_recall: episode returned after loop_end

Run from repo root:
  .venv/bin/python scripts/audit_cdv_loop.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

os.environ["LOOPLLM_PROVIDER"] = "mock"

import loopllm.mcp_server as srv  # noqa: E402

DIVIDER = "─" * 60


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def show(label: str, raw: str) -> dict:  # type: ignore[return]
    try:
        d = json.loads(raw)
        print(f"\n[{label}]")
        print(json.dumps(d, indent=2))
        return d
    except Exception:
        print(f"\n[{label}] (raw): {raw[:200]}")
        return {}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = __import__("pathlib").Path(tmp) / "audit.db"
        os.environ["LOOPLLM_DB"] = str(db_path)

        # Reset module state between runs
        srv._store = None  # type: ignore[attr-defined]
        srv._priors = None  # type: ignore[attr-defined]
        srv._agent_loop = None  # type: ignore[attr-defined]
        srv._episodic = None  # type: ignore[attr-defined]
        srv._status_path = None  # type: ignore[attr-defined]
        srv._history_path = None  # type: ignore[attr-defined]
        srv._episodes_feed_path = None  # type: ignore[attr-defined]

        # ── 1a: intercept cold (no episodes) ─────────────────────────────
        section("1a. loopllm_intercept — cold start")
        r = show("intercept", srv._tool_intercept(
            "add retry logic to the download function"
        ))
        print(f"  ↳ route={r.get('route')} score={r.get('quality', {}).get('quality_score')} "
              f"recall_available={r.get('recall_available')}")

        # ── 1b: loop_start (no history) ───────────────────────────────────
        section("1b. loopllm_loop_start — cold (no episodes)")
        start_r = show("loop_start", srv._tool_loop_start(
            goal="add retry with exponential backoff to download(); raise after 3 tries",
            task_type="bugfix",
            evaluator_type="composite",
            quality_criteria=["retry", "backoff", "raise"],
            required_patterns=["retry"],
        ))
        sid = start_r.get("session_id", "")
        print(f"  ↳ session_id={sid}")
        print(f"  ↳ suggested_budget={start_r.get('suggested_budget')} "
              f"threshold={start_r.get('quality_threshold')} "
              f"confidence={start_r.get('confidence')}")
        print(f"  ↳ similar_episodes={start_r.get('similar_episodes')} "
              f"memory_hint={start_r.get('memory_hint')!r}")

        # ── 1c: loop_step — failing artifact ──────────────────────────────
        import asyncio  # noqa: PLC0415

        section("1c. loopllm_loop_step — step 1 (no 'retry' in artifact)")
        step1_r = show("loop_step_1", asyncio.run(srv._tool_loop_step(
            session_id=sid,
            step_output="def download(url): return requests.get(url).text",
            ctx=None,
        )))
        print(f"  ↳ decision={step1_r.get('decision')} score={step1_r.get('score')}")
        print(f"  ↳ cdv_mode={step1_r.get('cdv_mode')}")
        print(f"  ↳ channel_a_score={step1_r.get('channel_a_score')}")
        print(f"  ↳ deficiencies={step1_r.get('deficiencies')}")

        section("1c. loopllm_loop_step — step 2 (artifact passes criteria)")
        step2_r = show("loop_step_2", asyncio.run(srv._tool_loop_step(
            session_id=sid,
            step_output=(
                "def download(url, retries=3):\n"
                "    for attempt in range(retries):\n"
                "        try:\n"
                "            return requests.get(url).text\n"
                "        except Exception:\n"
                "            time.sleep(2 ** attempt)\n"
                "    raise RuntimeError('download failed after retry')"
            ),
            ctx=None,
        )))
        print(f"  ↳ decision={step2_r.get('decision')} score={step2_r.get('score')}")
        print(f"  ↳ cdv_mode={step2_r.get('cdv_mode')}")
        print(f"  ↳ channel_a_score={step2_r.get('channel_a_score')}")

        # ── 1d: loop_end ──────────────────────────────────────────────────
        section("1d. loopllm_loop_end — finalise and learn")
        end_r = show("loop_end", srv._tool_loop_end(session_id=sid))
        print(f"  ↳ final_score={end_r.get('final_score')} "
              f"steps_run={end_r.get('steps_run')} "
              f"stop_reason={end_r.get('stop_reason')!r}")

        # ── 1e: Recovery contract ─────────────────────────────────────────
        section("1e. Recovery contract: snapshot → hydrate → resume → step")

        # Start a new loop, capture snapshot
        start2_r = show("loop_start_2", srv._tool_loop_start(
            goal="fix flaky auth tests",
            task_type="bugfix",
        ))
        sid2 = start2_r.get("session_id", "")
        asyncio.run(srv._tool_loop_step(
            session_id=sid2,
            step_output="partial fix: reduced test flakiness by 50%",
            ctx=None,
        ))
        print(f"\n  ↳ Started loop sid2={sid2}, did 1 step, now simulating restart...")

        # Simulate restart: wipe in-memory controller, keep DB
        srv._agent_loop = None  # type: ignore[attr-defined]

        run_status_r = show("run_status_after_restart", srv._tool_run_status())
        active = run_status_r.get('active_runs', [])
        count_ar = active.get('count', len(active)) if isinstance(active, dict) else len(active)
        print(f"  ↳ active_runs count={count_ar}")

        resume_r = show("loop_resume", srv._tool_loop_resume(session_id=sid2))
        print(f"  ↳ source={resume_r.get('source')} "
              f"session_id={resume_r.get('session_id')}")

        step_after_r = show("loop_step_after_resume", asyncio.run(srv._tool_loop_step(
            session_id=sid2,
            step_output="all auth tests now pass, test suite green",
            ctx=None,
        )))
        print(f"  ↳ decision={step_after_r.get('decision')} "
              f"score={step_after_r.get('score')}")

        # ── 1f: recall after episode ──────────────────────────────────────
        section("1f. loopllm_recall — episode from step 1d should surface")
        recall_r = show("recall", srv._tool_recall(
            query="retry backoff download",
            task_type="bugfix",
        ))
        count = recall_r.get("count", 0)
        print(f"  ↳ recall returned {count} episode(s)")
        for ep in recall_r.get("episodes", [])[:3]:
            print(f"     goal={ep.get('goal')!r} score={ep.get('score_final')}")

        # ── 1g: intercept again — recall_available now true ───────────────
        section("1g. loopllm_intercept — recall_available after episodes recorded")
        intercept2_r = show("intercept_2", srv._tool_intercept(
            "add retry logic to the upload function"
        ))
        print(f"  ↳ recall_available={intercept2_r.get('recall_available')}")

        section("AUDIT COMPLETE")
        print("  All phases ran without exception.")
        print(f"  Temp DB: {db_path}")


if __name__ == "__main__":
    main()
