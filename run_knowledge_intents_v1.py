#!/usr/bin/env python3
"""Build structured Knowledge Intent V1 records from approved primary questions.

This job intentionally reads only the V2.1 question export.  It never reads
support-case messages, engineer answers, production RAG tables, or topic
embedding artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from embeddings import read_openrouter_token
from llm import OPENROUTER_DEFAULT_MODEL, OPENROUTER_PROVIDER, LLMService, OpenRouterLLM, parse_json_response


LLM_MODEL = OPENROUTER_DEFAULT_MODEL
PROVIDER = OPENROUTER_PROVIDER
OPENROUTER_DEFAULT_RPM = 20
MAX_RETRIES = 1

FAMILIES = {
    "firmware",
    "password_access",
    "compatibility",
    "feature_capability",
    "capacity_limit",
    "configuration",
    "troubleshooting",
    "integration",
    "recording_storage",
    "network",
    "mobile_cloud",
    "documentation_certificate",
    "software",
    "accessory",
    "pricing_commercial",
    "other",
}

ACTIONS = {
    "find_or_get",
    "check_latest",
    "update_install",
    "rollback_recover",
    "reset",
    "configure",
    "connect",
    "add",
    "remove",
    "bind",
    "unbind",
    "check_support",
    "check_compatibility",
    "check_limit",
    "check_specification",
    "diagnose",
    "fix",
    "select",
    "replace",
    "download",
    "view",
    "export_import",
    "other",
}

OBJECT_TYPES = {
    "camera",
    "nvr",
    "dvr",
    "intercom_monitor",
    "door_station",
    "access_terminal",
    "access_controller",
    "alarm_hub",
    "sensor",
    "turnstile",
    "barrier",
    "software",
    "mobile_app",
    "accessory",
    "generic_device",
    "other",
}

PROBLEMS = {
    "offline",
    "cannot_add",
    "no_video",
    "no_audio",
    "no_notification",
    "connection_failure",
    "authentication_failure",
    "firmware_failure",
    "compatibility_failure",
}

FEATURE_ALIASES = {
    "hikconnect": "hik_connect",
    "hik_connect": "hik_connect",
    "hik-connect": "hik_connect",
    "hik connect": "hik_connect",
    "rtsp": "rtsp",
    "onvif": "onvif",
    "sip": "sip",
    "sharpsense": "sharpsense",
    "anpr": "anpr",
    "lpr": "anpr",
    "autotracking": "autotracking",
    "auto_tracking": "autotracking",
    "fingerprint": "fingerprint",
    "exit_button": "exit_button",
    "audio": "audio",
    "microphone": "audio",
    "motion_detection": "motion_detection",
    "recording_schedule": "recording_schedule",
    "bitrate": "bitrate",
    "alarm_output": "alarm_output",
    "alarm_input": "alarm_input",
    "face_recognition": "face_recognition",
    "face": "face_recognition",
    "card": "card",
    "rfid": "card",
    "poe": "poe",
    "wifi": "wifi",
    "wi-fi": "wifi",
    "4g": "4g",
    "h264": "h264",
    "h.264": "h264",
    "h265": "h265",
    "h.265": "h265",
    "hevc": "h265",
    "rack_ears": "rack_ears",
    "rack ears": "rack_ears",
    "firmware": "firmware",
    "прошивка": "firmware",
}

PROBLEM_ALIASES = {
    "no_sound": "no_audio",
    "no_audio_recording": "no_audio",
    "sound_missing": "no_audio",
    "not_added": "cannot_add",
    "cannot_connect": "connection_failure",
    "connection_error": "connection_failure",
    "auth_failure": "authentication_failure",
    "update_failure": "firmware_failure",
    "firmware_update_failure": "firmware_failure",
    "no_time_sync": "time_sync_failure",
}

RELATION_ALIASES = {
    "door_station_monitor": "door_station_to_monitor",
    "door_station_to_monitor": "door_station_to_monitor",
    "door_station_to_indoor_monitor": "door_station_to_monitor",
    "camera_nvr": "camera_to_nvr",
    "camera_to_nvr": "camera_to_nvr",
    "device_hikconnect": "device_to_hikconnect",
    "device_to_hikconnect": "device_to_hikconnect",
    "sensor_hub": "sensor_to_hub",
    "sensor_to_hub": "sensor_to_hub",
    "device_app": "device_to_mobile_app",
    "device_to_mobile_app": "device_to_mobile_app",
    "camera_camera": "camera_to_camera",
    "camera_to_camera": "camera_to_camera",
    "monitor_panel": "door_station_to_monitor",
    "panel_monitor": "door_station_to_monitor",
}

SYSTEM_PROMPT = """Ты строишь структурированный knowledge intent для технического вопроса поддержки.

НЕ отвечай на вопрос и не объясняй решение. Не используй engineer answer: его нет
и он запрещён для этой задачи. Верни только строгий JSON без markdown и reasoning.

Верни ровно эти шесть полей:
{
  "family": "...",
  "action": "...",
  "feature": "... or null",
  "problem": "... or null",
  "relation": "... or null",
  "object_type": "..."
}

family выбирай только из:
firmware, password_access, compatibility, feature_capability, capacity_limit,
configuration, troubleshooting, integration, recording_storage, network,
mobile_cloud, documentation_certificate, software, accessory,
pricing_commercial, other.

action выбирай только из:
find_or_get, check_latest, update_install, rollback_recover, reset, configure,
connect, add, remove, bind, unbind, check_support, check_compatibility,
check_limit, check_specification, diagnose, fix, select, replace, download,
view, export_import, other.

object_type выбирай только из:
camera, nvr, dvr, intercom_monitor, door_station, access_terminal,
access_controller, alarm_hub, sensor, turnstile, barrier, software,
mobile_app, accessory, generic_device, other.

feature — только короткий стандартизированный lowercase snake_case технический
концепт или null. Предпочитай такие labels, когда они подходят: rtsp, onvif,
sip, hik_connect, sharpsense, anpr, autotracking, fingerprint, exit_button,
audio, motion_detection, recording_schedule, bitrate, alarm_output,
alarm_input, face_recognition, card, poe, wifi, 4g, h264, h265, rack_ears.
Не помещай в feature производителя, модель, версию, IP или серийный номер.

problem используй только для реально проблемного состояния, обычно family
troubleshooting; выбирай только из: offline, cannot_add, no_video, no_audio,
no_notification, connection_failure, authentication_failure,
firmware_failure, compatibility_failure. Иначе null.

relation — короткий lowercase snake_case тип связи/отношения или null. Для
количественных и соединительных вопросов используй, например:
door_station_to_monitor, camera_to_nvr, device_to_hikconnect, sensor_to_hub,
device_to_mobile_app, camera_to_camera.

Выбирай action по настоящей цели пользователя, а не по одному слову в вопросе.
Обязательные различия:
- где найти/получить/скачать firmware -> family firmware, action find_or_get;
- какая актуальная/последняя версия firmware -> firmware.check_latest;
- как установить/обновить firmware -> firmware.update_install;
- проблема после обновления -> family troubleshooting, action diagnose или fix,
  problem firmware_failure; не firmware.find_or_get и не firmware.update_install;
- «устройство совместимо?» -> compatibility.check_compatibility;
- «прилагаются ли rack ears?» -> accessory.check_specification с feature rack_ears;
- NVR offline -> troubleshooting.diagnose с problem offline;
- нет audio recording -> troubleshooting.diagnose с problem no_audio;
- количество устройств в конкретной связи -> capacity_limit.check_limit с
  relation, чтобы разные relations не объединялись.

Не добавляй сведения, которых нет в canonical_question. Поля должны отражать
запрошенное знание, а не конкретный ответ."""

REPAIR_PROMPT = """Предыдущий JSON не прошёл схему. Повтори только строгий JSON ровно
с шестью полями family, action, feature, problem, relation, object_type.
family, action и object_type должны быть только из перечисленных enum; null
допустим только для feature, problem и relation. Не отвечай на вопрос. Не
используй модели, версии, IP или серийные номера в feature/relation/problem."""

DEFAULT_INPUT = Path("/home/ubuntu/ai-sales-engineer-knowledge/input/topic_questions_v2_1.jsonl")
DEFAULT_DATA_DIR = Path("/opt/aihelper/data")
DEFAULT_OUTPUT_JSON = Path("/opt/aihelper/data/knowledge_intents_v1.json")
DEFAULT_OUTPUT_MARKDOWN = Path("/opt/aihelper/KNOWLEDGE_INTENT_REVIEW.md")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    file_values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        # Match normal dotenv behavior when a file contains duplicate keys:
        # the last assignment in the file wins, while an explicit process
        # environment value still takes precedence over the file.
        file_values[name] = value
    for name, value in file_values.items():
        os.environ.setdefault(name, value)


def load_input(path: Path) -> list[dict]:
    rows = []
    seen = set()
    required = (
        "support_case_id", "analysis_id", "source_content_hash",
        "canonical_question", "domain", "knowledge_type", "models",
    )
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        missing = [field for field in required if field not in raw]
        if missing:
            raise ValueError(f"input line {line_number} missing: {', '.join(missing)}")
        case_id = int(raw["support_case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate support_case_id in input: {case_id}")
        seen.add(case_id)
        question = str(raw["canonical_question"]).strip()
        if not question:
            raise ValueError(f"input line {line_number} has an empty canonical_question")
        models = raw["models"] if isinstance(raw["models"], list) else []
        rows.append({
            "support_case_id": case_id,
            "analysis_id": int(raw["analysis_id"]),
            "source_content_hash": str(raw["source_content_hash"]),
            "canonical_question": question,
            "domain": str(raw["domain"]),
            "knowledge_type": str(raw["knowledge_type"]),
            # This is upstream deterministic metadata.  LLM output never
            # replaces or augments it.
            "models": list(dict.fromkeys(str(model) for model in models if str(model))),
        })
    return rows


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def clean_token(value: str) -> str:
    value = value.strip().casefold().replace("ё", "е")
    value = re.sub(r"[\s\-/]+", "_", value)
    value = re.sub(r"[^a-z0-9_]+", "", value)
    return re.sub(r"_+", "_", value).strip("_")


def normalize_optional(value: object, aliases: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional fields must be strings or null")
    token = clean_token(value)
    if not token or token in {"null", "none", "нет", "отсутствует"}:
        return None
    if aliases:
        token = aliases.get(token, aliases.get(value.strip().casefold(), token))
    if len(token) > 48:
        raise ValueError("optional label is too long")
    return token


def model_or_ip_in_text(text: str, models: list[str]) -> bool:
    for model in models:
        if model and re.search(re.escape(model), text, re.IGNORECASE):
            return True
    return bool(re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text))


def parse_intent(content: str, row: dict) -> dict:
    data = parse_json_response(content)
    expected = {"family", "action", "feature", "problem", "relation", "object_type"}
    if not isinstance(data, dict) or not expected.issubset(data) or set(data) - expected - {"knowledge_key"}:
        raise ValueError("response must contain the six intent fields and no unsupported fields")
    family = data["family"] if isinstance(data["family"], str) else ""
    action = data["action"] if isinstance(data["action"], str) else ""
    object_type = data["object_type"] if isinstance(data["object_type"], str) else ""
    family = clean_token(family)
    action = clean_token(action)
    object_type = clean_token(object_type)
    if family not in FAMILIES:
        raise ValueError(f"illegal family: {family}")
    if action not in ACTIONS:
        raise ValueError(f"illegal action: {action}")
    if object_type not in OBJECT_TYPES:
        raise ValueError(f"illegal object_type: {object_type}")
    feature = normalize_optional(data["feature"], FEATURE_ALIASES)
    problem = normalize_optional(data["problem"], PROBLEM_ALIASES)
    relation = normalize_optional(data["relation"], RELATION_ALIASES)
    if feature == "firmware" and family == "firmware":
        feature = None
    for value in (feature, problem, relation):
        if value and model_or_ip_in_text(value, row["models"]):
            raise ValueError("intent field retains a model or IP")
    return {
        "family": family,
        "action": action,
        "feature": feature,
        "problem": problem,
        "relation": relation,
        "object_type": object_type,
    }


def build_knowledge_key(intent: dict) -> str:
    parts = [intent["family"], intent["action"]]
    for field in ("feature", "problem", "relation"):
        value = intent.get(field)
        if value and value not in parts:
            parts.append(value)
    return ".".join(parts)


def safe_error_message(error: Exception) -> str:
    message = str(error)
    for env_name in ("OPENROUTER_API_KEY",):
        secret = os.environ.get(env_name, "")
        if secret:
            message = message.replace(secret, "<redacted>")
    return message[:1000]


def recent_429_cooldown(
    failures_path: Path,
    rate_limit_marker_path: Path,
    window_seconds: float = 300.0,
) -> float:
    paths = [failures_path, rate_limit_marker_path]
    latest = None
    for path in paths:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8").splitlines() if path == failures_path else [path.read_text(encoding="utf-8")]
        for line in source:
            try:
                item = json.loads(line)
                if path == failures_path and "429" not in str(item.get("error_message", "")):
                    continue
                timestamp = datetime.fromisoformat(str(item.get("created_at") or item["last_429_at"]))
                latest = max(latest or timestamp, timestamp)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    if latest is None:
        return 0.0
    elapsed = (datetime.now(timezone.utc) - latest).total_seconds()
    return max(0.0, window_seconds - elapsed)


class RequestRateLimiter:
    """Schedule OpenRouter request starts below the configured rate limit."""

    def __init__(self, requests_per_minute: int = 30, initial_delay: float = 0.0):
        self.interval = 60.0 / requests_per_minute
        self.next_start = time.monotonic() + initial_delay
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.interval
        if wait:
            time.sleep(wait)


def classify_one(
    llm: LLMService,
    row: dict,
    limiter: RequestRateLimiter | None = None,
) -> tuple[dict, int]:
    payload = json.dumps({
        "canonical_question": row["canonical_question"],
        "domain": row["domain"],
        "knowledge_type": row["knowledge_type"],
        "models": row["models"],
    }, ensure_ascii=False)
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": payload},
    ]
    messages = base_messages
    last_error = None
    for retry in range(MAX_RETRIES + 1):
        try:
            if limiter is not None:
                limiter.acquire()
            content = llm.extract(messages, max_tokens=600)
            return parse_intent(content, row), retry + 1
        except (ValueError, json.JSONDecodeError) as error:
            last_error = error
            if retry >= MAX_RETRIES:
                break
            messages = base_messages + [{"role": "user", "content": REPAIR_PROMPT}]
        except Exception as error:
            # OpenRouterLLM owns network/timeout retries.  A batch parser error
            # may receive one repair request, but transport failures do not
            # trigger another unbounded outer retry.
            last_error = error
            break
    raise RuntimeError(f"intent classification failed after {MAX_RETRIES + 1} attempts: {safe_error_message(last_error)}")


def classify_cases(
    rows: list[dict],
    checkpoint: Path,
    failures_path: Path,
    llm: LLMService,
    requests_per_minute: int,
) -> tuple[list[dict], int]:
    input_by_key = {
        (row["support_case_id"], row["source_content_hash"], row["canonical_question"]): row
        for row in rows
    }
    existing = load_jsonl(checkpoint)
    completed = {}
    for item in existing:
        key = (int(item.get("support_case_id", -1)), str(item.get("source_content_hash", "")), str(item.get("canonical_question", "")))
        if key in input_by_key and item.get("knowledge_key") and item.get("provider") == PROVIDER and item.get("model") == LLM_MODEL and all(field in item for field in ("family", "action", "feature", "problem", "relation", "object_type", "scope_models")):
            completed[key] = item

    pending = []
    for row in rows:
        key = (row["support_case_id"], row["source_content_hash"], row["canonical_question"])
        if key in completed:
            continue
        pending.append(row)

    def classify_pending(row: dict) -> tuple[dict, dict | None, Exception | None]:
        key = (row["support_case_id"], row["source_content_hash"], row["canonical_question"])
        try:
            intent, attempts = classify_one(llm, row, limiter)
            item = {
                "support_case_id": row["support_case_id"],
                "analysis_id": row["analysis_id"],
                "source_content_hash": row["source_content_hash"],
                "canonical_question": row["canonical_question"],
                "domain": row["domain"],
                "knowledge_type": row["knowledge_type"],
                "scope_models": row["models"],
                **intent,
                "knowledge_key": build_knowledge_key(intent),
                "provider": PROVIDER,
                "model": LLM_MODEL,
                "attempt_count": attempts,
                "created_at": utc_now(),
            }
            return row, item, None
        except Exception as error:
            return row, None, error

    from concurrent.futures import ThreadPoolExecutor, as_completed

    rate_limit_marker_path = failures_path.with_name("knowledge_intent_rate_limit_v1.json")
    cooldown = recent_429_cooldown(failures_path, rate_limit_marker_path)
    # A previous run may have consumed part of the provider's rolling window.
    # Let that window clear before resuming after a 429.
    limiter = RequestRateLimiter(requests_per_minute, initial_delay=cooldown)
    executor = ThreadPoolExecutor(max_workers=1)
    futures = [executor.submit(classify_pending, row) for row in pending]
    rate_limited = False
    try:
        for future in as_completed(futures):
            row, item, error = future.result()
            key = (row["support_case_id"], row["source_content_hash"], row["canonical_question"])
            if item is not None:
                append_jsonl(checkpoint, item)
                completed[key] = item
                continue
            if "429" in safe_error_message(error):
                rate_limited = True
                break
            append_jsonl(failures_path, {
                "support_case_id": row["support_case_id"],
                "analysis_id": row["analysis_id"],
                "source_content_hash": row["source_content_hash"],
                "provider": PROVIDER,
                "model": LLM_MODEL,
                "attempt_count": MAX_RETRIES + 1,
                "error_message": safe_error_message(error),
                "created_at": utc_now(),
            })
    finally:
        if rate_limited:
            for future in futures:
                future.cancel()
        executor.shutdown(wait=not rate_limited, cancel_futures=rate_limited)
    if rate_limited:
        rate_limit_marker_path.write_text(
            json.dumps({"last_429_at": utc_now()}) + "\n", encoding="utf-8"
        )
        raise RuntimeError("OpenRouter rate limit active; checkpoint preserved for a later resume")
    return [completed[key] for key in input_by_key if key in completed], len(rows) - len(completed)


def counter_distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items(), key=lambda pair: pair[0].casefold()))


def model_distribution(items: list[dict]) -> dict[str, int]:
    counts = Counter(model for item in items for model in item.get("scope_models", []) if model)
    return dict(sorted(counts.items(), key=lambda pair: pair[0].casefold()))


def group_intents(intents: list[dict]) -> list[dict]:
    by_key = defaultdict(list)
    for item in intents:
        by_key[item["knowledge_key"]].append(item)
    groups = []
    for knowledge_key, members in by_key.items():
        members = sorted(members, key=lambda item: int(item["support_case_id"]))
        groups.append({
            "knowledge_key": knowledge_key,
            "frequency": len({int(item["support_case_id"]) for item in members}),
            "case_ids": [int(item["support_case_id"]) for item in members],
            "representative_questions": [item["canonical_question"] for item in members[:3]],
            "object_type_distribution": counter_distribution([item["object_type"] for item in members]),
            "model_distribution": model_distribution(members),
            "domain_distribution": counter_distribution([item["domain"] for item in members]),
            "knowledge_type_distribution": counter_distribution([item["knowledge_type"] for item in members]),
            "scope_models": sorted({model for item in members for model in item.get("scope_models", []) if model}, key=str.casefold),
        })
    return sorted(groups, key=lambda group: (-group["frequency"], group["knowledge_key"]))


def percent_distribution(values: dict[str, int], total: int) -> str:
    if not values:
        return "-"
    return ", ".join(f"{key} ({value}/{total})" for key, value in values.items())


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_review(summary: dict, groups: list[dict], top_n: int = 50) -> str:
    lines = [
        "# Knowledge Intent Review V1",
        "",
        "> Structured human-review intents only. No answers, engineer answers, Verified Knowledge, or production RAG changes were used.",
        "> Grouping is deterministic `GROUP BY knowledge_key`; one support case contributes one vote.",
        "",
        "## Run Summary",
        "",
        f"- processed cases: {summary['processed_cases']}",
        f"- failures: {summary['failures']}",
        f"- unique knowledge_keys: {summary['unique_knowledge_keys']}",
        f"- singleton keys: {summary['singleton_keys']}",
        f"- frequency >=2: {summary['frequency_ge_2']}",
        f"- frequency >=3: {summary['frequency_ge_3']}",
        f"- frequency >=5: {summary['frequency_ge_5']}",
        f"- frequency >=10: {summary['frequency_ge_10']}",
        f"- maximum frequency: {summary['maximum_frequency']}",
        "- knowledge_key generation: deterministic Python build from family/action/feature/problem/relation",
        "- LLM settings: temperature=0, enable_thinking=false, JSON only",
        "",
        "## All Knowledge Keys",
        "",
        "| knowledge_key | frequency |",
        "|---|---:|",
    ]
    for group in groups:
        lines.append(f"| `{md(group['knowledge_key'])}` | {group['frequency']} |")
    lines.extend(["", f"## Top {min(top_n, len(groups))}", ""])
    for group in groups[:top_n]:
        lines.extend([
            f"### `{md(group['knowledge_key'])}` — frequency {group['frequency']}",
            "",
            "Cases:",
            "",
        ])
        for case_id, question in zip(group["case_ids"], group["representative_questions"], strict=False):
            lines.append(f"- #{case_id} {md(question)}")
        lines.extend([
            "",
            "Object types: " + percent_distribution(group["object_type_distribution"], group["frequency"]),
            "",
            "Models: " + percent_distribution(group["model_distribution"], group["frequency"]),
            "",
            "Domains: " + percent_distribution(group["domain_distribution"], group["frequency"]),
            "",
            "Existing knowledge_types: " + percent_distribution(group["knowledge_type_distribution"], group["frequency"]),
            "",
            "Review status: pending",
            "",
        ])
    return "\n".join(lines)


def build_artifacts(args) -> dict:
    load_env_file(args.env_file)
    model = LLM_MODEL
    token_file = Path(os.environ.get("OPENROUTER_TOKEN_FILE", str(Path(__file__).with_name("openrouter"))))
    api_key = read_openrouter_token(token_file)
    timeout_seconds = float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "120"))
    requests_per_minute = int(os.environ.get("OPENROUTER_REQUESTS_PER_MINUTE", str(OPENROUTER_DEFAULT_RPM)))
    if requests_per_minute < 1:
        raise ValueError("requests per minute must be positive")
    if not api_key:
        raise RuntimeError("OpenRouter token is not available; refusing to fabricate knowledge intents")
    llm = OpenRouterLLM(api_key, timeout=timeout_seconds)
    rows = load_input(args.input)
    if not rows:
        raise ValueError("input contains no question rows")
    case_ids = [str(row.get("support_case_id")) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("input contains duplicate support_case_id values")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.data_dir / "knowledge_intents_v1.jsonl"
    failures_path = args.data_dir / "knowledge_intent_failures_v1.jsonl"
    intents, failures = classify_cases(
        rows, checkpoint, failures_path, llm, requests_per_minute
    )
    groups = group_intents(intents)
    summary = {
        "schema_version": 1,
        "artifact_type": "knowledge_intent_v1",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "provider": PROVIDER,
        "model": model,
        "processed_cases": len(intents),
        "failures": failures,
        "unique_knowledge_keys": len(groups),
        "singleton_keys": sum(group["frequency"] == 1 for group in groups),
        "frequency_ge_2": sum(group["frequency"] >= 2 for group in groups),
        "frequency_ge_3": sum(group["frequency"] >= 3 for group in groups),
        "frequency_ge_5": sum(group["frequency"] >= 5 for group in groups),
        "frequency_ge_10": sum(group["frequency"] >= 10 for group in groups),
        "maximum_frequency": max((group["frequency"] for group in groups), default=0),
        "grouping": {"method": "group_by_knowledge_key", "distinct_support_case_vote": True},
        "llm": {
            "provider": PROVIDER,
            "model": model,
            "temperature": 0,
            "enable_thinking": False,
            "json_only": True,
            "requests_per_minute": requests_per_minute,
        },
        "created_at": utc_now(),
        "intents": sorted(intents, key=lambda item: int(item["support_case_id"])),
        "knowledge_key_groups": groups,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_review(summary, groups), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--env-file", type=Path, default=Path("/etc/aihelper.env"))
    args = parser.parse_args()
    try:
        summary = build_artifacts(args)
    except Exception as error:
        print(f"knowledge intent build failed: {safe_error_message(error)}")
        return 1
    print(json.dumps({key: summary[key] for key in (
        "processed_cases", "failures", "unique_knowledge_keys", "singleton_keys",
        "frequency_ge_2", "frequency_ge_3", "frequency_ge_5", "frequency_ge_10",
        "maximum_frequency",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
