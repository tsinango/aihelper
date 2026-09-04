# aihelper

Current architecture status, confirmed production issues, and the recommended
remediation plan are documented in
[`TECHNICAL_STATUS_AND_REMEDIATION.md`](TECHNICAL_STATUS_AND_REMEDIATION.md).

The service uses OpenRouter as its only model provider. `llm.complete()`,
`llm.extract()`, and `llm.judge()` are the internal model interface. The raw
OpenRouter token is kept in the protected `openrouter` file (or supplied via
`OPENROUTER_API_KEY`).

Configure the following non-secret variables in `/etc/aihelper.env`:

```dotenv
OPENROUTER_TOKEN_FILE=/opt/aihelper/openrouter
OPENROUTER_TIMEOUT_SECONDS=120
OPENROUTER_RERANK_ENABLED=true
INBOX_WORKER_NAME=aihelper-inbox-worker
INBOX_WORKER_HEARTBEAT_INTERVAL_SECONDS=10
INBOX_WORKER_HEALTHY_THRESHOLD_SECONDS=45
```

## 本机 Qwen 评测（仅影子评测）

本机 Qwen3.5 GGUF 只用于离线生成器比较，不接入生产问答路径。评测时
runner 会逐个启动 2B/4B，并把结果写入专用的 `local_model_eval_*` 表和 JSON
文件；生产服务仍只使用 OpenRouter。模型默认从 Hugging Face 缓存自动发现：
`Qwen3.5-2B-Q4_K_M.gguf` 和 `Qwen3.5-4B-Q4_K_M.gguf`。

`data/golden_set.json` 是 135 条固定样本，问题和参考答案均引用真实的
`data/telegram_knowledge_review.json` 线程，包含 80 条直接回答、条件限制、
15 条型号混淆、10 条证据不足和 10 条多知识命中。若对应 knowledge key 尚未
发布，runner 使用标记为 `golden_reference` 的评测专用证据，不会写入生产知识。

有数据库时运行两个模型的完整 golden benchmark：

```bash
.venv/bin/python evaluate_local_qwen.py \
  --env-file /etc/aihelper.env \
  --models 2b,4b --limit 135 --mode golden \
  --output data/local_qwen_golden_v2.json
```

数据库暂时不可用时，使用同一套真实 artifact，仅写 JSON：

```bash
.venv/bin/python evaluate_local_qwen.py \
  --env-file /etc/aihelper.env \
  --models 2b,4b --limit 135 --mode golden --no-database \
  --output data/local_qwen_golden_v2.json
```

评测关闭 reasoning，固定 temperature=0；记录加载时间、生成时间和
tokens/s。结果同时给出 `status_pass`、`source_selection_pass` 和
`golden_pass`，`answer_pass` 仍保留为人工复核字段，不用自动判断替代领域专家。
只有在完整 golden set 上，4B 在复杂条件、型号混淆和正确拒答等综合指标明显
高出约 3–5 个百分点，才考虑生产保留 4B；否则删除其评测依赖并保持 2B 或
现有 OpenRouter 架构。评测期间 OpenRouter embedding 429 会回退到已发布知识
快照，不会改变知识或阻塞生产审批。

Generation uses `nvidia/nemotron-3-ultra-550b-a55b:free`, embeddings use
`nvidia/nemotron-3-embed-1b:free` (2048 dimensions), and retrieval optionally
uses `nvidia/llama-nemotron-rerank-vl-1b-v2:free`. Batch checkpoints are
preserved on persistent failures and OpenRouter rate limits. The V1.1 intent
batch is an offline preprocessing job: Nemotron extracts structured intent
dimensions and deterministic knowledge keys so similar cases can be reviewed
once as a group. It does not approve or publish knowledge by itself. The
resume command and quota behavior are documented in `OPERATIONS.md`.

```bash
python -m pip install -r requirements.txt
DATABASE_URL=postgresql://... python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The service keeps the Worker routes `/health`, `/api/v1/query`, `/api/v1/documents`,
`/api/support-cases`, and `/api/v1/support-cases`. Upload is deliberately disabled
for this copy-only migration; existing R2 documents are stored in `data/documents`.

## Query pipeline

`/api/v1/query` keeps its original request shape (`{"question":"..."}`) and
now returns optional `route`, `answer_status`, `citations`, `request_id`, and
`review_required` fields. Requests pass through deterministic product/model
recognition, bound aliases and approved few-shot examples, then route to
structured facts, Verified Knowledge, official-manual RAG, or a future live
inventory adapter before Nemotron composes the Russian answer.

Historical Telegram conversations are available through the `case_knowledge_memory`
layer after migration `003` as recall/reviewer evidence only. Unverified history
cannot directly answer customers; only published `verified_knowledge` is
authoritative.

## Review all Telegram history

The deterministic organizer covers all 602 imported conversations before any
document-learning step:

```bash
python organize_telegram_knowledge.py
python apply_migration.py migrations/004_complete_telegram_review.sql
python import_review_candidates.py --input data/telegram_knowledge_review.json
python apply_migration.py migrations/010_review_provenance_async_embedding.sql
python apply_migration.py migrations/011_review_group_modes.sql
python import_case_memory.py
```

Open `/review` to edit each proposed answer, message role, applicability level,
product scope, conditions, exceptions, and warnings alongside the complete
Telegram thread. The default grouping is deterministic and needs no model
request; V1.1/knowledge_key and OpenRouter semantic grouping are optional views.
Approval creates an auditable Verified Knowledge record and publishes it
immediately to the production bot. Embedding is queued as `pending` and can be
completed later by `reembed.py`; quota exhaustion never rolls back approval.
Older published versions with the same knowledge key are archived automatically.
