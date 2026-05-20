"""
agents/code_executor.py -- Sandboxed Python code execution for agents

Agents write Python code as part of their analysis. This module:
  1. Receives the code string from the agent
  2. Executes it in a restricted namespace with market data available
  3. Captures stdout, return value, and any errors
  4. Returns structured result + saves code to storage/agent_code/
  5. Feeds result back to the agent for further reasoning

Available in agent code namespace:
  df         -- OHLCV DataFrame for the current ticker
  indicators -- dict of computed indicators
  profile    -- TickerProfile dict
  np, pd     -- numpy and pandas
  math       -- math module
"""
import asyncio
import io
import json
import math
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import SYS

CODE_DIR = SYS.STORAGE_DIR / "agent_code"
CODE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_repr(obj: Any, max_len: int = 800) -> str:
    try:
        if isinstance(obj, pd.DataFrame):
            return f"DataFrame({obj.shape[0]}x{obj.shape[1]})\n{obj.to_string(max_rows=8)}"
        if isinstance(obj, pd.Series):
            return f"Series(len={len(obj)})\n{obj.to_string(max_rows=8)}"
        if isinstance(obj, np.ndarray):
            return f"ndarray{obj.shape}: {np.array2string(obj, max_line_width=80, precision=4)}"
        s = repr(obj)
        return s[:max_len] + "..." if len(s) > max_len else s
    except Exception:
        return str(type(obj).__name__)


async def execute_agent_code(
    code: str,
    df: Optional[pd.DataFrame],
    indicators: Dict,
    profile: Dict,
    symbol: str,
    agent_name: str = "agent",
) -> Dict:
    """
    Execute agent-written code and return structured result.

    Returns:
        {
            code: str,
            output: str,       # captured stdout
            result: str,       # repr of last expression value
            error: str | None,
            elapsed_ms: int,
            saved_path: str,
            success: bool,
        }
    """
    t0 = time.time()

    # Build namespace
    namespace = {
        "df": df if df is not None else pd.DataFrame(),
        "indicators": indicators,
        "profile": profile,
        "np": np,
        "pd": pd,
        "math": math,
        "print": print,
        "__builtins__": {
            "len": len, "list": list, "dict": dict, "tuple": tuple,
            "set": set, "str": str, "int": int, "float": float,
            "bool": bool, "round": round, "abs": abs, "min": min,
            "max": max, "sum": sum, "sorted": sorted, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "range": range,
            "isinstance": isinstance, "type": type, "print": print,
            "repr": repr, "__import__": __import__,
        }
    }

    captured = io.StringIO()
    old_stdout = sys.stdout
    result_val = None
    error = None

    try:
        sys.stdout = captured
        # Try eval for expressions, exec for statements
        try:
            compiled = compile(code, "<agent_code>", "eval")
            result_val = eval(compiled, namespace)
        except SyntaxError:
            compiled = compile(code, "<agent_code>", "exec")
            exec(compiled, namespace)
            result_val = namespace.get("result", None)
    except Exception as e:
        tb = traceback.format_exc()
        error = f"{type(e).__name__}: {e}\n{tb.strip().split(chr(10))[-1]}"
    finally:
        sys.stdout = old_stdout

    stdout_out = captured.getvalue().strip()
    result_str = _safe_repr(result_val) if result_val is not None else ""
    elapsed_ms = int((time.time() - t0) * 1000)

    # Save code to disk
    code_id = str(uuid.uuid4())[:8]
    fname = f"{symbol}_{agent_name}_{code_id}.py"
    saved_path = str(CODE_DIR / fname)
    try:
        header = f"# Agent: {agent_name} | Symbol: {symbol} | {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        (CODE_DIR / fname).write_text(header + code)
    except Exception:
        saved_path = ""

    output_parts = []
    if stdout_out:
        output_parts.append(stdout_out)
    if result_str and result_str != "None":
        output_parts.append(result_str)

    return {
        "code": code,
        "output": "\n".join(output_parts) if output_parts else "(no output)",
        "result": result_str,
        "error": error,
        "elapsed_ms": elapsed_ms,
        "saved_path": saved_path,
        "success": error is None,
        "code_id": code_id,
    }


async def run_agent_code_with_retry(
    code: str,
    df: Optional[pd.DataFrame],
    indicators: Dict,
    profile: Dict,
    symbol: str,
    agent_name: str = "agent",
) -> Dict:
    """Execute with one retry on syntax error (LLM sometimes adds markdown fences)."""
    # Strip markdown code fences if present
    cleaned = code.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    result = await execute_agent_code(
        cleaned, df, indicators, profile, symbol, agent_name
    )
    return result
