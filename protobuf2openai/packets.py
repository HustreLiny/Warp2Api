from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
import json

from .state import STATE, ensure_tool_ids
from .helpers import normalize_content_to_list, segments_to_text, segments_to_warp_results
from .models import ChatMessage


def packet_template() -> Dict[str, Any]:
    return {
        "task_context": {"active_task_id": ""},
        "input": {"context": {}, "user_inputs": {"inputs": []}},
        "settings": {
            "model_config": {
                "base": "claude-4.1-opus",
                "planning": "gpt-5 (high reasoning)",
                "coding": "auto",
            },
            "rules_enabled": False,
            "web_context_retrieval_enabled": False,
            "supports_parallel_tool_calls": False,
            "planning_enabled": False,
            "warp_drive_context_enabled": False,
            "supports_create_files": False,
            "use_anthropic_text_editor_tools": False,
            "supports_long_running_commands": False,
            "should_preserve_file_content_in_history": False,
            "supports_todos_ui": False,
            "supports_linked_code_blocks": False,
            "supported_tools": [9],
        },
        "metadata": {"logging": {"is_autodetected_user_query": True, "entrypoint": "USER_INITIATED"}},
    }


def map_history_to_warp_messages(history: List[ChatMessage], task_id: str, system_prompt_for_last_user: Optional[str] = None, attach_to_history_last_user: bool = False) -> List[Dict[str, Any]]:
    ensure_tool_ids()
    msgs: List[Dict[str, Any]] = []
    # Insert server tool_call preamble as first message
    msgs.append({
        "id": (STATE.tool_message_id or str(uuid.uuid4())),
        "task_id": task_id,
        "tool_call": {
            "tool_call_id": (STATE.tool_call_id or str(uuid.uuid4())),
            "server": {"payload": "IgIQAQ=="},
        },
    })

    # Determine the last input message index (either last 'user' or last 'tool' with tool_call_id)
    last_input_index: Optional[int] = None
    for idx in range(len(history) - 1, -1, -1):
        _m = history[idx]
        if _m.role == "user":
            last_input_index = idx
            break
        if _m.role == "tool" and _m.tool_call_id:
            last_input_index = idx
            break

    for i, m in enumerate(history):
        mid = str(uuid.uuid4())
        # Skip the final input message; it will be placed into input.user_inputs
        if (last_input_index is not None) and (i == last_input_index):
            continue
        if m.role == "user":
            user_query_obj: Dict[str, Any] = {"query": segments_to_text(normalize_content_to_list(m.content))}
            msgs.append({"id": mid, "task_id": task_id, "user_query": user_query_obj})
        elif m.role == "assistant":
            _assistant_text = segments_to_text(normalize_content_to_list(m.content))
            if _assistant_text:
                msgs.append({"id": mid, "task_id": task_id, "agent_output": {"text": _assistant_text}})
            for tc in (m.tool_calls or []):
                msgs.append({
                    "id": str(uuid.uuid4()),
                    "task_id": task_id,
                    "tool_call": {
                        "tool_call_id": tc.get("id") or str(uuid.uuid4()),
                        "call_mcp_tool": {
                            "name": (tc.get("function", {}) or {}).get("name", ""),
                            "args": (json.loads((tc.get("function", {}) or {}).get("arguments", "{}")) if isinstance((tc.get("function", {}) or {}).get("arguments"), str) else (tc.get("function", {}) or {}).get("arguments", {})) or {},
                        },
                    },
                })
        elif m.role == "tool":
            # Preserve tool_result adjacency by placing it directly in task_context
            if m.tool_call_id:
                msgs.append({
                    "id": str(uuid.uuid4()),
                    "task_id": task_id,
                    "tool_call_result": {
                        "tool_call_id": m.tool_call_id,
                        "call_mcp_tool": {
                            "success": {
                                "results": segments_to_warp_results(normalize_content_to_list(m.content))
                            }
                        },
                    },
                })
    return msgs


def attach_user_and_tools_to_inputs(packet: Dict[str, Any], history: List[ChatMessage], system_prompt_text: Optional[str]) -> None:
    # Use the final post-reorder message as input (user or tool result)
    if not history:
        assert False, "post-reorder 必须至少包含一条消息"
    last = history[-1]
    if last.role == "user":
        user_query_payload: Dict[str, Any] = {"query": segments_to_text(normalize_content_to_list(last.content))}
        # Note: router.py is responsible for injecting SYSTEM_PROMPT from .env.
        # Keep this function focused on attaching the user's input or tool results to the packet. 
        packet["input"]["user_inputs"]["inputs"].append({"user_query": user_query_payload})
        return
    if last.role == "tool" and last.tool_call_id:
        packet["input"]["user_inputs"]["inputs"].append({
            "tool_call_result": {
                "tool_call_id": last.tool_call_id,
                "call_mcp_tool": {
                    "success": {"results": segments_to_warp_results(normalize_content_to_list(last.content))}
                },
            }
        })
        return
    # If neither, assert to catch protocol violations
    assert False, "post-reorder 最后一条必须是 user 或 tool 结果" 