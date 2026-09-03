import re
from collections.abc import Iterable

MODEL_RE = re.compile(r"(?=[A-Za-z0-9./()\-]*\d)[A-Za-z0-9][A-Za-z0-9./()\-]{2,}")
ALIASES = (
    (re.compile(r"сброс|восстанов(?:ить|ление)|забыл.*парол|парол.*забыл", re.I), "password reset forgot password"),
    (re.compile(r"\bhik[\s-]?connect\b|хик[\s-]?коннект|хикконнект", re.I), "Hik-Connect hikconnect hcserver server address server IP"),
    (re.compile(r"hikvision|хиквижн|хика|hiwatch|хайвоч|iflow|айфлоу", re.I), "Hikvision HiWatch iFlow"),
    (re.compile(r"пальц|палец|отпечат", re.I), "fingerprint fingerprints"),
    (re.compile(r"добав|добавл", re.I), "add adding"),
    (re.compile(r"оборудован|устройств|терминал|панел", re.I), "device terminal equipment panel"),
    (re.compile(r"лиц|фейс", re.I), "face"),
    (re.compile(r"карт|пропуск", re.I), "card access card"),
)

CONTEXT_ONLY_TERMS = frozenset({
    "iflow",
    "hikvision",
    "hiwatch",
    "hik connect",
    "hik-connect",
    "hikconnect",
    "guardingvision",
    "guarding vision",
})

ROUTE_RULES = (
    ("inventory", re.compile(r"налич|доступ\w*|есть в наличии|на складе|купить|цена|стоимост|available|availability|stock|price", re.I)),
    ("compatibility", re.compile(r"совместим|подключ\w*|работать вместе|интеграц|подходит|compatible|\bconnect(?:ed|ion)?\b|\bintegrat\w*\b", re.I)),
    ("fault", re.compile(r"не работает|ошиб|проблем|почему|завис|не видит|не открыва|не отправля|слом|troubleshoot|error|doesn.?t work", re.I)),
    ("operation", re.compile(r"как |как сделать|настро|добав|удал|сброс|восстанов|прошив|скачать|войти|how to|configure|reset|update", re.I)),
    ("parameter", re.compile(r"сколько|какой|какие|есть ли|поддержива|характерист|разрешен|канал|размер|протокол|what|which|support", re.I)),
)


def language(text: str) -> str:
    counts = {"ru": len(re.findall(r"[А-Яа-яЁё]", text)), "zh": len(re.findall(r"[\u3400-\u9fff]", text)), "en": len(re.findall(r"[A-Za-z]", text))}
    name, count = max(counts.items(), key=lambda pair: pair[1])
    return name if count else "und"


def identifiers(text: str) -> list[str]:
    return sorted({item.upper() for item in MODEL_RE.findall(text)}, key=lambda item: (-len(item), item))


def _is_version_identifier(value: str) -> bool:
    """Keep document versions/dates from being treated as product models."""
    return bool(re.fullmatch(r"V?\d+(?:[.\-_]\d+)*", value, re.I)) or value.isdigit()


def retrieved_models(hit: dict) -> list[str]:
    """Extract product scope from retrieved metadata, never from the answer."""
    if hit.get("source_type") in {"verified_knowledge", "case_memory"}:
        scope = hit.get("scope") if isinstance(hit.get("scope"), dict) else {}
        models = identifiers(" ".join(_clean_scope_values(scope.get("models"))))
        if models:
            return models
    metadata_model = str(hit.get("product_model") or "")
    if metadata_model:
        return identifiers(metadata_model)

    # The migrated chunks may not have product_model populated.  Their document
    # title is still metadata, so use its first model-like token and ignore
    # version/date suffixes.
    for value in MODEL_RE.findall(str(hit.get("title") or "")):
        value = value.upper()
        if not _is_version_identifier(value):
            return [value]
    return []


def _model_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _model_relation(user_model: str, document_model: str) -> str:
    user_key = _model_key(user_model)
    document_key = _model_key(document_model)
    if user_key == document_key:
        return "exact"
    if user_key.startswith(document_key) or document_key.startswith(user_key):
        return "family"
    return "conflict"


def scope_match(user_models: list[str], document_models: list[str]) -> str:
    """Classify whether retrieved product scope matches the user's scope."""
    if not user_models:
        return "unspecified"
    if not document_models:
        return "unspecified"

    relations = [
        min((_model_relation(user, document) for user in user_models), key=("exact", "family", "conflict").index)
        for document in document_models
    ]
    if "conflict" in relations:
        return "conflict"
    if all(relation == "exact" for relation in relations):
        return "exact"
    return "family"


def verified_scope_match(question: str, verified_scope: dict | None) -> str:
    """Classify applicability of a published knowledge record.

    ``generic`` is deliberately separate from ``unspecified``: the former is a
    usable scope result for a knowledge record with no model restriction (or a
    question without a model), while the latter is used by document retrieval's
    conditional-answer behavior.
    """
    scope = verified_scope if isinstance(verified_scope, dict) else {}
    user_models = identifiers(question)
    knowledge_models = identifiers(" ".join(_clean_scope_values(scope.get("models"))))
    if not user_models:
        return "generic" if not knowledge_models else "unspecified"
    if not knowledge_models:
        return "generic"
    relations = [
        min((_model_relation(user, model) for model in knowledge_models), key=("exact", "family", "conflict").index)
        for user in user_models
    ]
    if "conflict" in relations:
        return "conflict"
    if all(relation == "exact" for relation in relations):
        return "exact"
    return "family"


def _clean_scope_values(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def scope_details(question: str, evidence: list[dict]) -> dict:
    """Return deterministic query/document scope data for answer generation."""
    user_models = identifiers(question)
    document_model_values = []
    for hit in evidence:
        document_model_values.extend(retrieved_models(hit))
    document_models = sorted(set(document_model_values), key=lambda item: (-len(item), item))
    return {
        "explicit_user_models": user_models,
        "retrieved_document_models": document_models,
        "scope_match": scope_match(user_models, document_models),
    }


def qualify_model_specific_answer(answer: str, model: str) -> str:
    """Make a model-specific answer conditional when the user omitted a model."""
    clarification = "Если у вас другая модель, укажите её — порядок может отличаться."
    return f"Если речь о {model}, порядок такой: {answer.rstrip()} {clarification}"


def apply_scope_to_answer(answer: str, scope: dict) -> str:
    if not scope["explicit_user_models"] and scope["retrieved_document_models"]:
        return qualify_model_specific_answer(answer, scope["retrieved_document_models"][0])
    return answer


def matching_aliases(text: str, aliases: Iterable[dict] | None = None) -> list[dict]:
    """Return approved database aliases that occur in the user question.

    Aliases are used as an interpretation aid.  They are deliberately kept
    separate from the answer evidence so a terminology match cannot become a
    factual claim by itself.
    """
    if not aliases:
        return []
    normalized = text.casefold()
    matches = []
    for row in aliases:
        alias = str(row.get("alias") or "").strip()
        if alias and alias.casefold() in normalized:
            matches.append(dict(row))
    return matches


def static_alias_terms(text: str) -> list[str]:
    """Return deterministic retrieval terms for built-in product terminology."""
    return [term for pattern, expansion in ALIASES if pattern.search(text) for term in expansion.split()]


def is_context_only_question(text: str) -> bool:
    """Identify a bare brand/platform mention without treating it as a question."""
    normalized = re.sub(r"[\W_]+", " ", text.casefold(), flags=re.UNICODE).strip()
    normalized_terms = {
        re.sub(r"[\W_]+", " ", term.casefold(), flags=re.UNICODE).strip()
        for term in CONTEXT_ONLY_TERMS
    }
    return bool(normalized) and normalized in normalized_terms


def alias_knowledge_keys(text: str, aliases: Iterable[dict] | None = None) -> set[str]:
    return {
        str(row.get("knowledge_key")).strip()
        for row in matching_aliases(text, aliases)
        if str(row.get("knowledge_key") or "").strip()
    }


def expanded(text: str, aliases: Iterable[dict] | None = None) -> str:
    terms = static_alias_terms(text)
    for row in matching_aliases(text, aliases):
        terms.extend([str(row.get("concept") or ""), str(row.get("alias") or "")])
    terms = [term.strip() for term in dict.fromkeys(terms) if term.strip()]
    return f"{text} {' '.join(terms)}" if terms else text


def route_question(question: str, models: list[str] | None = None) -> str:
    """Deterministically classify the customer task before retrieval."""
    for route, pattern in ROUTE_RULES:
        if pattern.search(question):
            return route
    return "parameter" if models else "unknown"
