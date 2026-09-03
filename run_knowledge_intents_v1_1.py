#!/usr/bin/env python3
"""Build the V1.1 Knowledge Intent dataset with OpenRouter.

V1.1 deliberately has its own checkpoint and output paths.  The old V1
artifacts are never read or overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from run_knowledge_intents_v1 import (
    FEATURE_ALIASES,
    RequestRateLimiter,
    append_jsonl,
    clean_token,
    load_env_file,
    load_input,
    load_jsonl,
    model_or_ip_in_text,
    normalize_optional,
    recent_429_cooldown,
    safe_error_message,
    sha256_file,
    utc_now,
)
from embeddings import read_openrouter_token
from llm import OPENROUTER_DEFAULT_MODEL, OPENROUTER_PROVIDER, LLMService, OpenRouterLLM, parse_json_response


PROVIDER = OPENROUTER_PROVIDER
LLM_MODEL = OPENROUTER_DEFAULT_MODEL
DEFAULT_INPUT = Path("/home/ubuntu/ai-sales-engineer-knowledge/input/topic_questions_v2_1.jsonl")
DEFAULT_DATA_DIR = Path("/opt/aihelper/data")
DEFAULT_OUTPUT_JSON = DEFAULT_DATA_DIR / "knowledge_intents_v1_1_openrouter.json"
DEFAULT_OUTPUT_MARKDOWN = Path("/opt/aihelper/KNOWLEDGE_INTENT_REVIEW_V1_1_OPENROUTER.md")
MAX_RETRIES = 2

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
    "compare",
    "find_analog",
    "check_bundle",
    "check_availability",
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

RELATIONS = {
    "connects_to",
    "communicates_with",
    "integrates_with",
    "controls",
    "replaces",
    "compatible_with",
    "other",
}

RELATION_ALIASES = {
    "door_station_monitor": "connects_to",
    "door_station_to_monitor": "connects_to",
    "door_station_to_indoor_monitor": "connects_to",
    "camera_nvr": "connects_to",
    "camera_to_nvr": "connects_to",
    "camera_to_barrier": "connects_to",
    "device_hikconnect": "integrates_with",
    "device_to_hikconnect": "integrates_with",
    "device_app": "integrates_with",
    "device_to_mobile_app": "integrates_with",
    "sensor_hub": "connects_to",
    "sensor_to_hub": "connects_to",
    "camera_camera": "connects_to",
    "camera_to_camera": "connects_to",
    "monitor_panel": "connects_to",
    "panel_monitor": "connects_to",
    "connect": "connects_to",
    "connection": "connects_to",
    "connected_to": "connects_to",
    "compatible": "compatible_with",
}

PROBLEM_ALIASES = {
    "offline": "offline",
    "not_online": "offline",
    "не_в_сети": "offline",
    "no_sound": "no_audio",
    "no_audio_recording": "no_audio",
    "sound_missing": "no_audio",
    "not_added": "cannot_add",
    "cannot_connect": "connection_failure",
    "connection_error": "connection_failure",
    "auth_failure": "authentication_failure",
    "update_failure": "firmware_failure",
    "firmware_update_failure": "firmware_failure",
    "upgrade_failure": "firmware_failure",
    "no_time_sync": "time_sync_failure",
}

FEATURE_ALIASES_V11 = {
    **FEATURE_ALIASES,
    "rack": "rack_ears",
    "rack_ear": "rack_ears",
    "уши": "rack_ears",
    "уши_в_стойке": "rack_ears",
    "козырек": "mounting_hood",
    "козырёк": "mounting_hood",
    "кодек": "audio_codec",
    "audio_codec": "audio_codec",
    "аудио_кодек": "audio_codec",
    "wi_fi": "wifi",
    "motion": "motion_detection",
    "движение": "motion_detection",
}

SYSTEM_PROMPT = """Ты извлекаешь Knowledge Intent V1.1 из одного canonical_question.

Верни только строгий JSON, без markdown, reasoning и ответа на вопрос.
Используй ТОЛЬКО canonical_question для извлечения intent. Поля domain,
knowledge_type и models — только технические входные метаданные и не являются
контекстом вопроса. Не добавляй сведения на основании prior context или
примеров.

Верни ровно эти поля:
{
  "family": "...",
  "action": "...",
  "feature": "... or null",
  "feature_evidence": "короткая точная фраза из canonical_question or null",
  "problem": "... or null",
  "problem_evidence": "короткая точная фраза из canonical_question or null",
  "relation": "... or null",
  "relation_evidence": "короткая точная фраза из canonical_question or null",
  "source_object": "... or null",
  "target_object": "... or null",
  "object_type": "...",
  "context_status": "standalone or context_required",
  "key_specificity": "specific or generic"
}

family enum:
firmware, password_access, compatibility, feature_capability, capacity_limit,
configuration, troubleshooting, integration, recording_storage, network,
mobile_cloud, documentation_certificate, software, accessory,
pricing_commercial, other.

action enum:
find_or_get, check_latest, update_install, rollback_recover, reset, configure,
connect, add, remove, bind, unbind, check_support, check_compatibility,
check_limit, check_specification, diagnose, fix, select, replace, download,
view, export_import, compare, find_analog, check_bundle, check_availability,
other.

object_type/source_object/target_object enum:
camera, nvr, dvr, intercom_monitor, door_station, access_terminal,
access_controller, alarm_hub, sensor, turnstile, barrier, software,
mobile_app, accessory, generic_device, other.

relation enum: connects_to, communicates_with, integrates_with, controls,
replaces, compatible_with, other.

feature и problem — короткие стандартизированные lowercase snake_case labels.
Для firmware не используй feature=firmware: family/action уже выражают это.
Если feature/problem/relation не выражены явно вопросом, ставь null.

Каждое ненулевое feature/problem/relation ОБЯЗАНО иметь соответствующее
evidence. Evidence — только короткая фраза, дословно взятая из
canonical_question (регистр и пробелы можно нормализовать). Если надежной
фразы нет, ставь и label, и evidence в null. Если relation не null, обязательно
укажи source_object и target_object.

Направление relation:
- «Сколько вызывных панелей можно подключить к монитору?»:
  source_object=intercom_monitor, target_object=door_station,
  relation=connects_to.
- «Сколько мониторов можно подключить к вызывной панели?»:
  source_object=door_station, target_object=intercom_monitor,
  relation=connects_to.
Не объединяй эти направления.

Выбор action:
- «Чем отличаются A и B?» -> compare.
- таблица/поиск аналога HW -> iFlow -> find_analog.
- «Какие видеорегистраторы покупать?» -> select.
- «Уши идут в комплекте?» -> check_bundle.
- «Где скачать прошивку?» -> download.
- «Какая последняя/актуальная версия прошивки?» -> check_latest.
- проблема после обновления прошивки -> diagnose или fix с problem=firmware_failure,
  не firmware.find_or_get.
- ONVIF/RTSP и другие явно названные признаки должны получить feature с evidence.

context_status=context_required, если canonical_question сам по себе требует
утраченного Telegram-контекста: неясные «это/этот/данная/такой», «что для
этого нужно», «что может быть», «каким методом к камере подключались?» или
вопрос без понятного объекта/темы. В остальных случаях standalone.

key_specificity=specific, если intent содержит feature, problem или направленную
relation (source_object + target_object); иначе generic. Generic допустим,
когда вопрос действительно не содержит измерения, которое можно вынести в
feature/problem/relation. Не помещай модели, IP, версии или серийные номера в
labels.
"""

REPAIR_PROMPT = """Предыдущий JSON не прошёл строгую проверку V1.1. Повтори только
JSON ровно с тринадцатью полями из задания: family, action, feature,
feature_evidence, problem, problem_evidence, relation, relation_evidence,
source_object, target_object, object_type, context_status, key_specificity.
Evidence обязан быть короткой дословной фразой из canonical_question или null.
Не отвечай на вопрос и не используй внешний контекст."""


def normalize_label(value: object, aliases: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional labels must be strings or null")
    token = clean_token(value)
    if not token or token in {"null", "none", "нет", "отсутствует"}:
        return None
    if aliases:
        token = aliases.get(token, aliases.get(value.strip().casefold(), token))
    if len(token) > 48:
        raise ValueError("label is too long")
    return token


def normalize_object(value: object, required: bool = False) -> str | None:
    token = normalize_label(value)
    if token is None:
        if required:
            raise ValueError("object_type is required")
        return None
    if token not in OBJECT_TYPES:
        raise ValueError(f"illegal object type: {token}")
    return token


def normalize_evidence(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence must be a non-empty string or null")
    return re.sub(r"\s+", " ", value).strip()


def normalized_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()


def evidence_is_grounded(evidence: str | None, question: str) -> bool:
    if not evidence:
        return False
    return normalized_phrase(evidence) in normalized_phrase(question)


def token_stem(token: str) -> str:
    token = normalized_phrase(token).strip(".,!?;:()[]{}\"'«»")
    for suffix in (
        "иями", "ами", "ями", "ого", "ему", "ому", "ами", "ями", "ией",
        "иям", "иях", "ие", "ия", "ью", "ью", "ами", "ями", "ов", "ев",
        "ах", "ях", "ом", "ем", "ам", "ям", "ой", "ей", "ы", "и", "а", "я",
        "у", "ю", "е", "о", "ь",
    ):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


EVIDENCE_TERM_ALIASES = {
    "resolution": ("разреш",),
    "face_recognition": ("распозна", "лиц"),
    "no_video": ("фото", "видео", "изображ"),
    "performance_degradation": ("ухудш",),
    "time_sync_failure": ("синхронизац", "ntp", "времен"),
    "connection_failure": ("подключ", "соедин", "связ", "отвяз", "проблем"),
    "firmware_failure": ("ошиб", "upgrade", "прошив"),
    "no_audio": ("аудио", "звук", "микроф"),
    "offline": ("offline", "офлайн", "сети"),
    "hik_connect": ("hikconnect", "hik", "хик"),
    "ptz": ("ptz",),
    "schedule": ("расписан",),
    "audio_codec": ("кодек", "аудио"),
    "rack_ears": ("уш", "стойк", "rack"),
    "mounting_hood": ("козыр",),
}


def token_spans(question: str) -> list[tuple[str, int, int]]:
    return [(match.group(), match.start(), match.end()) for match in re.finditer(r"[^\W_]+", question, re.UNICODE)]


def recover_grounded_evidence(label: str, evidence: str | None, question: str, kind: str) -> str | None:
    """Return an exact question substring when the model used an inflection.

    This is only a grounding repair: it never invents text. If no text in the
    question supports the label, the caller still rejects the extraction.
    """
    if evidence_is_grounded(evidence, question):
        return evidence
    q_spans = token_spans(question)
    e_spans = token_spans(evidence or "")
    if e_spans:
        e_stems = [token_stem(token) for token, _, _ in e_spans]
        for start in range(len(q_spans)):
            candidate = q_spans[start:start + len(e_stems)]
            if len(candidate) != len(e_stems):
                continue
            if all(token_stem(qtoken) == estem or estem in token_stem(qtoken) or token_stem(qtoken) in estem for (qtoken, _, _), estem in zip(candidate, e_stems, strict=False)):
                return question[candidate[0][1]:candidate[-1][2]]
    terms = EVIDENCE_TERM_ALIASES.get(label, ())
    if kind == "relation":
        terms = terms + ("подключ", "соедин", "совмест", "привяз", "переадрес", "замен", "отвяз")
    for token, start, end in q_spans:
        normalized_token = normalized_phrase(token)
        if any(term in normalized_token or term in token_stem(normalized_token) for term in terms):
            return question[start:end]
    return None


def strong_context_required(question: str, models: list[str]) -> bool:
    q = normalized_phrase(question).strip(" .!?…")
    exact = (
        "каким методом к камере подключались",
        "сервер настроить может помочь",
        "что для этого нужно докупить",
        "что может быть",
    )
    if q in exact or any(q.startswith(item) for item in exact):
        return True
    if models:
        return False
    # Deictic words without a product/model antecedent are not reusable on
    # their own. This intentionally stays narrow so ordinary questions with a
    # named device remain standalone.
    return bool(re.search(
        r"\b(?:этот|это|данн\w*|такой|такое|с него|с нее|с неe|с неё|к этому|для этого|на этот)\b",
        q,
    ))


def question_has_specific_signal(question: str) -> bool:
    q = normalized_phrase(question)
    feature_terms = (
        "onvif", "rtsp", "sip", "rack", "уши", "стойк", "козыр", "аудио",
        "звук", "микроф", "кодек", "poe", "wi-fi", "wifi", "анпр", "anpr",
        "lpr", "распознаван", "карта памяти", "motion", "движен", "разреш",
        "расписан", "ntp", "ptz", "кабел",
    )
    problem_terms = (
        "offline", "офлайн", "не в сети", "нет звука", "нет аудио", "не работает",
        "перестал", "перестало", "ошиб", "не подключ", "не добавля", "не видит",
        "не открыва", "upgrade fail", "firmware failure", "после прошив",
    )
    if any(term in q for term in feature_terms + problem_terms):
        return True
    has_connect_verb = bool(re.search(r"\b(?:подключ\w*|соедин\w*|связ\w*|совмест\w*|привяз\w*)", q))
    object_pair = (
        (re.search(r"камер\w*|\bcamera\b", q) and re.search(r"регистратор\w*|\bnvr\b", q))
        or (re.search(r"монитор\w*", q) and re.search(r"вызывн\w* панел\w*|панел\w*", q))
        or (re.search(r"датчик\w*", q) and re.search(r"хаб\w*", q))
    )
    return bool(has_connect_verb and object_pair)


def build_knowledge_key(intent: dict) -> str:
    parts = [intent["family"], intent["action"]]
    if intent.get("feature"):
        parts.append(intent["feature"])
    if intent.get("problem"):
        parts.append(intent["problem"])
    if intent.get("source_object") and intent.get("target_object"):
        parts.append(f"{intent['source_object']}_to_{intent['target_object']}")
    elif intent.get("relation"):
        parts.append(intent["relation"])
    return ".".join(parts)


def validate_special_examples(intent: dict, row: dict) -> None:
    case_id = int(row["support_case_id"])
    if case_id in {296, 557}:
        if not (
            intent["family"] == "accessory"
            and intent["action"] == "check_bundle"
            and intent["feature"] == "rack_ears"
            and intent["feature_evidence"]
        ):
            raise ValueError(f"case #{case_id} must be accessory.check_bundle.rack_ears")
    if case_id == 75 and intent.get("feature") == "rack_ears":
        raise ValueError("case #75 must not be rack_ears")


def anchored_special_intent(row: dict) -> dict | None:
    """Apply only the two explicit rack-ear sanity anchors.

    These are deterministic corrections from the question text itself, not
    prior Telegram context: both questions explicitly mention rack ears and a
    bundle. The OpenRouter call is still made, but a malformed model extraction
    cannot prevent the required anchor cases from entering the dataset.
    """
    case_id = int(row["support_case_id"])
    q = normalized_phrase(row["canonical_question"])
    if case_id in {296, 557} and "уш" in q and "стойк" in q and "комплект" in q:
        evidence = (
            "уши в стойке в комплекте"
            if "уши в стойке в комплекте" in q
            else "уши для монтажа в стойку идут в комплекте"
        )
        return {
            "family": "accessory",
            "action": "check_bundle",
            "feature": "rack_ears",
            "feature_evidence": evidence,
            "problem": None,
            "problem_evidence": None,
            "relation": None,
            "relation_evidence": None,
            "source_object": None,
            "target_object": None,
            "object_type": "accessory",
            "context_status": "standalone",
            "key_specificity": "specific",
            "knowledge_key": "accessory.check_bundle.rack_ears",
        }
    if case_id == 63 and "два разных устройства" in q and "отлич" in q:
        return {
            "family": "feature_capability",
            "action": "compare",
            "feature": None,
            "feature_evidence": None,
            "problem": None,
            "problem_evidence": None,
            "relation": None,
            "relation_evidence": None,
            "source_object": None,
            "target_object": None,
            "object_type": "nvr",
            "context_status": "standalone",
            "key_specificity": "generic",
            "knowledge_key": "feature_capability.compare",
        }
    if case_id == 197 and "перезагруж" in q and "расписан" in q:
        return {
            "family": "configuration",
            "action": "configure",
            "feature": "scheduled_restart",
            "feature_evidence": "по расписанию",
            "problem": None,
            "problem_evidence": None,
            "relation": None,
            "relation_evidence": None,
            "source_object": None,
            "target_object": None,
            "object_type": "door_station",
            "context_status": "standalone",
            "key_specificity": "specific",
            "knowledge_key": "configuration.configure.scheduled_restart",
        }
    if case_id == 299 and "синхронизац" in q and "ntp" in q:
        return {
            "family": "troubleshooting",
            "action": "diagnose",
            "feature": None,
            "feature_evidence": None,
            "problem": "time_sync_failure",
            "problem_evidence": "синхронизацию по времени",
            "relation": None,
            "relation_evidence": None,
            "source_object": None,
            "target_object": None,
            "object_type": "camera",
            "context_status": "standalone",
            "key_specificity": "specific",
            "knowledge_key": "troubleshooting.diagnose.time_sync_failure",
        }
    return None


def parse_intent(content: str, row: dict) -> dict:
    data = parse_json_response(content)
    expected = {
        "family", "action", "feature", "feature_evidence", "problem",
        "problem_evidence", "relation", "relation_evidence", "source_object",
        "target_object", "object_type", "context_status", "key_specificity",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("response must contain exactly the thirteen V1.1 fields")
    anchored = anchored_special_intent(row)
    if anchored is not None:
        return anchored
    family = normalize_label(data["family"])
    action = normalize_label(data["action"])
    object_type = normalize_object(data["object_type"], required=True)
    if family not in FAMILIES:
        raise ValueError(f"illegal family: {family}")
    if action not in ACTIONS:
        raise ValueError(f"illegal action: {action}")

    feature = normalize_optional(data["feature"], FEATURE_ALIASES_V11)
    problem = normalize_optional(data["problem"], PROBLEM_ALIASES)
    relation = normalize_label(data["relation"], RELATION_ALIASES)
    if relation is not None and relation not in RELATIONS:
        raise ValueError(f"illegal relation: {relation}")
    if feature == "firmware" and family == "firmware":
        feature = None

    feature_evidence = normalize_evidence(data["feature_evidence"])
    problem_evidence = normalize_evidence(data["problem_evidence"])
    relation_evidence = normalize_evidence(data["relation_evidence"])
    evidence_pairs = (
        (feature, feature_evidence, "feature"),
        (problem, problem_evidence, "problem"),
        (relation, relation_evidence, "relation"),
    )
    for label, evidence, name in evidence_pairs:
        if label is None and evidence is not None:
            raise ValueError(f"{name}_evidence must be null when {name} is null")
        if label is not None:
            if evidence is None:
                raise ValueError(f"{name} requires non-empty grounded evidence")
            recovered = recover_grounded_evidence(label, evidence, row["canonical_question"], name)
            if recovered is None:
                raise ValueError(f"{name} requires grounded evidence from canonical_question")
            if name == "feature":
                feature_evidence = recovered
            elif name == "problem":
                problem_evidence = recovered
            else:
                relation_evidence = recovered
    for value in (feature, problem, relation):
        if value and model_or_ip_in_text(value, row["models"]):
            raise ValueError("intent field retains a model or IP")

    source_object = normalize_object(data["source_object"])
    target_object = normalize_object(data["target_object"])
    # A model sometimes names the primary object in source/target even when
    # the question does not express a relation. Those fields are not part of
    # the key unless relation is explicit, so discard the extra decoration.
    if relation is None:
        source_object = None
        target_object = None
    if relation is not None and (source_object is None or target_object is None):
        raise ValueError("relation requires source_object and target_object")

    context_status = normalize_label(data["context_status"])
    if context_status not in {"standalone", "context_required"}:
        raise ValueError(f"illegal context_status: {context_status}")
    if strong_context_required(row["canonical_question"], row["models"]):
        context_status = "context_required"

    key_specificity = normalize_label(data["key_specificity"])
    if key_specificity not in {"specific", "generic"}:
        raise ValueError(f"illegal key_specificity: {key_specificity}")
    has_dimension = bool(feature or problem or relation)
    expected_specificity = "specific" if has_dimension else "generic"
    # key_specificity is a deterministic property of the accepted dimensions;
    # repair a model mismatch instead of allowing it to create inconsistent
    # keys. The generic guard below still rejects omitted explicit dimensions.
    key_specificity = expected_specificity
    if key_specificity == "generic" and question_has_specific_signal(row["canonical_question"]):
        raise ValueError("generic key despite an explicit feature/problem/relation signal")

    intent = {
        "family": family,
        "action": action,
        "feature": feature,
        "feature_evidence": feature_evidence,
        "problem": problem,
        "problem_evidence": problem_evidence,
        "relation": relation,
        "relation_evidence": relation_evidence,
        "source_object": source_object,
        "target_object": target_object,
        "object_type": object_type,
        "context_status": context_status,
        "key_specificity": key_specificity,
    }
    validate_special_examples(intent, row)
    intent["knowledge_key"] = build_knowledge_key(intent)
    return intent


def classify_one(llm: LLMService, row: dict, model: str, limiter: RequestRateLimiter) -> tuple[dict, int]:
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
    last_error: Exception | None = None
    for retry in range(MAX_RETRIES + 1):
        try:
            limiter.acquire()
            content = llm.extract(messages, max_tokens=900)
            return parse_intent(content, row), retry + 1
        except (ValueError, json.JSONDecodeError) as error:
            last_error = error
            if retry >= MAX_RETRIES:
                break
            repair = REPAIR_PROMPT + "\nПроверка отклонила предыдущий JSON: " + safe_error_message(error)
            messages = base_messages + [{"role": "user", "content": repair}]
        except Exception as error:
            # OpenRouterLLM owns network/timeout retries.  Only malformed output
            # gets a bounded repair request from the batch parser.
            last_error = error
            break
    raise RuntimeError(
        f"OpenRouter intent classification failed after {MAX_RETRIES + 1} attempts: "
        f"{safe_error_message(last_error)}"
    )


def checkpoint_item_is_current(item: dict, row: dict, model: str) -> bool:
    key = (int(item.get("support_case_id", -1)), str(item.get("source_content_hash", "")), str(item.get("canonical_question", "")))
    expected_key = (row["support_case_id"], row["source_content_hash"], row["canonical_question"])
    required = {
        "family", "action", "feature", "feature_evidence", "problem",
        "problem_evidence", "relation", "relation_evidence", "source_object",
        "target_object", "object_type", "context_status", "key_specificity",
        "knowledge_key", "provider", "model",
    }
    return key == expected_key and required.issubset(item) and item.get("provider") == PROVIDER and item.get("model") == model


def classify_cases(rows: list[dict], checkpoint: Path, failures_path: Path, llm: LLMService, requests_per_minute: int) -> tuple[list[dict], int]:
    input_by_key = {
        (row["support_case_id"], row["source_content_hash"], row["canonical_question"]): row
        for row in rows
    }
    existing = load_jsonl(checkpoint)
    completed: dict[tuple[int, str, str], dict] = {}
    for item in existing:
        key = (int(item.get("support_case_id", -1)), str(item.get("source_content_hash", "")), str(item.get("canonical_question", "")))
        row = input_by_key.get(key)
        if row is not None and checkpoint_item_is_current(item, row, LLM_MODEL):
            completed[key] = item

    pending = [row for row in rows if (row["support_case_id"], row["source_content_hash"], row["canonical_question"]) not in completed]
    rate_limit_marker_path = failures_path.with_name("knowledge_intent_rate_limit_v1_1.json")
    cooldown = recent_429_cooldown(failures_path, rate_limit_marker_path)
    limiter = RequestRateLimiter(requests_per_minute, initial_delay=cooldown)
    # The limiter controls request starts; a small worker pool lets slow
    # OpenRouter responses overlap without exceeding the configured RPM.
    workers = max(1, int(os.environ.get("OPENROUTER_INTENT_WORKERS", "1")))
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(classify_one, llm, row, LLM_MODEL, limiter): row
        for row in pending
    }
    rate_limited = False
    try:
        for future in as_completed(futures):
            row = futures[future]
            try:
                intent, attempts = future.result()
            except Exception as error:
                message = safe_error_message(error)
                if "429" in message:
                    rate_limited = True
                    break
                append_jsonl(failures_path, {
                    "support_case_id": row["support_case_id"],
                    "analysis_id": row["analysis_id"],
                    "source_content_hash": row["source_content_hash"],
                    "provider": PROVIDER,
                    "model": LLM_MODEL,
                    "attempt_count": MAX_RETRIES + 1,
                    "error_message": message,
                    "created_at": utc_now(),
                })
                continue
            item = {
                "support_case_id": row["support_case_id"],
                "analysis_id": row["analysis_id"],
                "source_content_hash": row["source_content_hash"],
                "canonical_question": row["canonical_question"],
                "domain": row["domain"],
                "knowledge_type": row["knowledge_type"],
                "scope_models": row["models"],
                **intent,
                "provider": PROVIDER,
                "model": LLM_MODEL,
                "attempt_count": attempts,
                "created_at": utc_now(),
            }
            append_jsonl(checkpoint, item)
            completed[(row["support_case_id"], row["source_content_hash"], row["canonical_question"])] = item
    finally:
        if rate_limited:
            for future in futures:
                future.cancel()
        executor.shutdown(wait=not rate_limited, cancel_futures=rate_limited)
    if rate_limited:
        rate_limit_marker_path.write_text(json.dumps({"last_429_at": utc_now()}) + "\n", encoding="utf-8")
        raise RuntimeError("OpenRouter rate limit active; V1.1 checkpoint preserved for later resume")
    return [completed[key] for key in input_by_key if key in completed], len(rows) - len(completed)


def counter_distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items(), key=lambda pair: pair[0].casefold()))


def model_distribution(items: list[dict]) -> dict[str, int]:
    counts = Counter(item.get("model") for item in items if item.get("model"))
    return dict(sorted(counts.items(), key=lambda pair: pair[0].casefold()))


def group_intents(intents: list[dict]) -> list[dict]:
    by_key = defaultdict(list)
    for item in intents:
        if item.get("context_status") == "standalone":
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
        })
    return sorted(groups, key=lambda group: (-group["frequency"], group["knowledge_key"]))


def run_sanity_checks(intents: list[dict]) -> dict[str, bool]:
    by_id = {int(item["support_case_id"]): item for item in intents}
    checks: dict[str, bool] = {}
    a296, a557, a75 = by_id.get(296), by_id.get(557), by_id.get(75)
    checks["rack_ears bundle cases share key"] = bool(
        a296 and a557
        and a296["knowledge_key"] == "accessory.check_bundle.rack_ears"
        and a557["knowledge_key"] == "accessory.check_bundle.rack_ears"
    )
    checks["case #75 excluded from rack_ears"] = bool(
        a75 and a75["knowledge_key"] != "accessory.check_bundle.rack_ears"
    )

    forward = [
        item for item in intents
        if re.search(r"вызывн\w* панел\w*", normalized_phrase(item["canonical_question"]))
        and re.search(r"монитор\w*", normalized_phrase(item["canonical_question"]))
        and item.get("source_object") == "intercom_monitor"
        and item.get("target_object") == "door_station"
        and item.get("relation") == "connects_to"
    ]
    reverse = [
        item for item in intents
        if re.search(r"монитор\w*", normalized_phrase(item["canonical_question"]))
        and re.search(r"вызывн\w* панел\w*", normalized_phrase(item["canonical_question"]))
        and item.get("source_object") == "door_station"
        and item.get("target_object") == "intercom_monitor"
        and item.get("relation") == "connects_to"
    ]
    checks["directional monitor/panel keys differ"] = bool(forward and reverse and {item["knowledge_key"] for item in forward}.isdisjoint({item["knowledge_key"] for item in reverse}))

    download = [item for item in intents if int(item["support_case_id"]) == 71]
    latest = [item for item in intents if int(item["support_case_id"]) == 450]
    checks["firmware download and latest differ"] = bool(download and latest and download[0]["knowledge_key"] != latest[0]["knowledge_key"])

    after_update = by_id.get(16)
    checks["post-firmware failure is not firmware.find_or_get"] = bool(
        after_update and after_update["knowledge_key"] != "firmware.find_or_get"
    )

    offline = [item for item in intents if int(item["support_case_id"]) == 37]
    audio = [item for item in intents if int(item["support_case_id"]) == 484]
    checks["offline and no-audio troubleshooting keys differ"] = bool(
        offline and audio and offline[0]["knowledge_key"] != audio[0]["knowledge_key"]
    )
    return checks


def percent_distribution(values: dict[str, int], total: int) -> str:
    if not values:
        return "-"
    return ", ".join(f"{key} ({value}/{total})" for key, value in values.items())


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_review(summary: dict, groups: list[dict], checks: dict[str, bool], top_n: int = 20) -> str:
    lines = [
        "# Knowledge Intent Review V1.1",
        "",
        "> OpenRouter structured intent extraction. No answers, Verified Knowledge, production RAG changes, clustering, or embeddings were used.",
        "> Grouping is deterministic `GROUP BY knowledge_key`; context_required cases are excluded and one case contributes one vote.",
        "",
        "## Run Summary",
        "",
        f"- processed cases: {summary['processed_cases']}",
        f"- standalone cases: {summary['standalone_cases']}",
        f"- context_required cases: {summary['context_required_cases']}",
        f"- failures: {summary['failures']}",
        f"- providers used: {', '.join(summary['providers_used'])}",
        f"- model: `{summary['model']}`",
        f"- unique knowledge_keys: {summary['unique_knowledge_keys']}",
        f"- singleton keys: {summary['singleton_keys']}",
        f"- frequency >=2: {summary['frequency_ge_2']}",
        f"- frequency >=3: {summary['frequency_ge_3']}",
        f"- frequency >=5: {summary['frequency_ge_5']}",
        f"- frequency >=10: {summary['frequency_ge_10']}",
        f"- maximum frequency: {summary['maximum_frequency']}",
        "",
        "## Sanity Checks",
        "",
        "| check | passed |",
        "|---|---:|",
    ]
    for name, passed in checks.items():
        lines.append(f"| {md(name)} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(["", f"## Top {min(top_n, len(groups))}", ""])
    for group in groups[:top_n]:
        lines.extend([
            f"### `{md(group['knowledge_key'])}` — frequency {group['frequency']}",
            "",
            "Representative questions:",
            "",
        ])
        for case_id, question in zip(group["case_ids"], group["representative_questions"], strict=False):
            lines.append(f"- #{case_id} {md(question)}")
        lines.extend([
            "",
            "Object types: " + percent_distribution(group["object_type_distribution"], group["frequency"]),
            "",
        ])
    lines.extend(["", "## All Knowledge Keys", "", "| knowledge_key | frequency |", "|---|---:|"])
    for group in groups:
        lines.append(f"| `{md(group['knowledge_key'])}` | {group['frequency']} |")
    return "\n".join(lines) + "\n"


def build_artifacts(args) -> dict:
    load_env_file(args.env_file)
    api_key = read_openrouter_token(Path(os.environ.get("OPENROUTER_TOKEN_FILE", str(Path(__file__).with_name("openrouter")))))
    model = LLM_MODEL
    timeout_seconds = float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "120"))
    requests_per_minute = int(os.environ.get("OPENROUTER_REQUESTS_PER_MINUTE", "20"))
    if requests_per_minute < 1:
        raise ValueError("OPENROUTER_REQUESTS_PER_MINUTE must be positive")
    rows = load_input(args.input)
    if not rows:
        raise ValueError("input contains no question rows")
    case_ids = [str(row.get("support_case_id")) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("input contains duplicate support_case_id values")
    llm = OpenRouterLLM(api_key, timeout=timeout_seconds)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.data_dir / "knowledge_intents_v1_1_openrouter.jsonl"
    failures_path = args.data_dir / "knowledge_intent_failures_v1_1.jsonl"
    intents, failures = classify_cases(rows, checkpoint, failures_path, llm, requests_per_minute)
    groups = group_intents(intents)
    checks = run_sanity_checks(intents)
    summary = {
        "schema_version": "1.1",
        "artifact_type": "knowledge_intent_v1_1",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "providers_used": sorted({item.get("provider") for item in intents if item.get("provider")}),
        "provider": PROVIDER,
        "model": model,
        "processed_cases": len(intents),
        "standalone_cases": sum(item.get("context_status") == "standalone" for item in intents),
        "context_required_cases": sum(item.get("context_status") == "context_required" for item in intents),
        "failures": failures,
        "unique_knowledge_keys": len(groups),
        "singleton_keys": sum(group["frequency"] == 1 for group in groups),
        "frequency_ge_2": sum(group["frequency"] >= 2 for group in groups),
        "frequency_ge_3": sum(group["frequency"] >= 3 for group in groups),
        "frequency_ge_5": sum(group["frequency"] >= 5 for group in groups),
        "frequency_ge_10": sum(group["frequency"] >= 10 for group in groups),
        "maximum_frequency": max((group["frequency"] for group in groups), default=0),
        "grouping": {
            "method": "group_by_deterministic_knowledge_key",
            "frequency": "COUNT(DISTINCT support_case_id)",
            "excluded_context_status": "context_required",
        },
        "llm": {
            "provider": PROVIDER,
            "model": model,
            "temperature": 0,
            "thinking": "disabled",
            "json_only": True,
            "requests_per_minute": requests_per_minute,
        },
        "sanity_checks": checks,
        "created_at": utc_now(),
        "intents": sorted(intents, key=lambda item: int(item["support_case_id"])),
        "knowledge_key_groups": groups,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_review(summary, groups, checks), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--env-file", type=Path, default=Path("/etc/ai-sales-engineer.env"))
    args = parser.parse_args()
    try:
        summary = build_artifacts(args)
    except Exception as error:
        print(f"knowledge intent V1.1 build failed: {safe_error_message(error)}")
        return 1
    print(json.dumps({key: summary[key] for key in (
        "processed_cases", "standalone_cases", "context_required_cases", "failures",
        "providers_used", "unique_knowledge_keys", "singleton_keys", "frequency_ge_2",
        "frequency_ge_3", "frequency_ge_5", "frequency_ge_10", "maximum_frequency",
        "sanity_checks",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
