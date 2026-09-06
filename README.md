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

## V2 当前状态（截至 2026-09-05）

V2 已完成 Phase 2.2 的生产闭环并正式收口，完成 Phase 3.0（收口 + 评测基线），
并实现 Phase 3.1 Read-only Internal QA：内部工程师问答只读已确认 Knowledge。
下一阶段是 Phase 3.2 feedback / Experience / Retest（尚未开始）。

```text
Raw Evidence
  → OpenRouter Structured Learning
  → 语义归并、俄语规范化
  → Pending Knowledge
  → 用户确认
  → 可追溯 Knowledge
  → 确定性精确 Entity 关联（默认不再调用 LLM 做结构整理）
```

**Phase 3.0 组织层收口**：确认 Knowledge 后默认只执行确定性的精确 Entity 关联
（`organization.py` 中不依赖 LLM 的 baseline），不再自动调用 LLM 整理 relation
或实体结构。已有 Entity / Relation 数据、provenance、防环校验和人工
tree/move/prune API 全部保留；未删除任何表或历史数据，未新增 relation 类型。
内部部署开关 `V2_ORGANIZATION_LLM_ENABLED`（默认关闭）仅用于回退验证，不是产品
设置。

**技术失败 ≠ 业务歧义**：compare judge 的 provider/解析/contract 失败现在标记
`technical_failure`，Inbox job 会失败并可重试，不会再伪装成产品澄清问题生成
无价值的提问。真正的语义不确定仍走澄清流程。

Learning extraction 默认使用免费的
`openai/gpt-oss-20b:free`（可通过 `V2_LEARNING_MODEL` 配置，但必须保持
OpenRouter free endpoint）。正常路径只进行一次 LLM 调用；只有 JSON、俄语、
技术标记或 evidence contract 校验失败时才进行一次 repair。持续失败会保留
Raw Evidence，但不会把原文直接写入 Knowledge。

Knowledge 页面提供简单维护能力：搜索、编辑、软删除、恢复、调整 Entity、查看
来源和修改历史。人工修改 Knowledge 不调用 LLM，保持 Knowledge ID 和 provenance；
修改内容会清空旧 embedding，后续由 `reembed.py` 重新生成。Raw Evidence 不会被
编辑或删除。

Entity Tree 只提供轻量组织层。用户可以手动删除完全为空的 Entity 分支；只要该
Entity 或其子树仍被 active 或 deleted Knowledge 引用，后端都会拒绝 prune。结构
删除是 `active=false` 的软删除，不会物理删除 Entity、relation 或 Knowledge。

当前 V2 生产进程：

- `aihelper.service`：FastAPI Web 服务
- `aihelper-inbox-worker.service`：PostgreSQL durable Inbox worker

运行状态通过 `/health` 和 `/ready` 检查；`/ready` 同时验证 PostgreSQL、必要 schema
和 worker heartbeat。部署、迁移和故障排查见 [`OPERATIONS.md`](OPERATIONS.md)。

V2 当前已应用迁移 `013`–`020`，包括 bulk intake、durable job、worker heartbeat、
轻量 Entity organization、Knowledge history 和安全的 Entity pruning。

未完成的 Chat Grounded QA、全局 Knowledge Gardening、复杂 taxonomy、Ontology、
Knowledge Graph editor 和 Telegram V2 接入不会进入当前 V2 主导航或主流程。

## V2 页面

- `/inbox`：提交资料、查看处理状态、确认或编辑 Pending Knowledge。
- `/knowledge`：维护已保存 Knowledge、来源、历史和 Entity Tree。
- `/documents`：查看现有文档资产；文档学习管道仍未作为 V2 Learning 主入口。
- `/chat`：内部工程师问答页（Internal engineer draft）。`POST /api/v2/answers`
  只从 eligible Knowledge（active + official_source/user_confirmed + accepted
  supports 来源 + 无已知型号/版本冲突）生成答案草稿、澄清或拒答，并保存
  `v2_answer_runs`（含 evidence snapshot）；`GET /api/v2/answers/{id}` 查看
  历史。回答只读 Knowledge，不学习、不产生 Experience、不修改 Knowledge。
  学习资料仍进入 Inbox。

## V2 评测基线（Phase 3.0）

从 Phase 3.0 起评测是上线门槛。`data/golden_set.json`（135 条）保持不可变；
V2 特有的映射、人工预期、改写问法和禁止断言保存在 sidecar
`data/v2_eval_cases.json`（固定 30 条：15 可回答 / 5 必须澄清 / 5 无依据 /
5 型号/条件边界），不覆盖原 golden 文件。当前 sidecar 的 V2 Knowledge 映射、
改写问法仍 `pending_expert_mapping`，由领域专家在 Phase 3.1 前补齐，缺口由
runner 如实报告，不伪造。

```bash
# 无网络/无模型：完整性与固定选样检查，写 data/v2_eval_report.json
.venv/bin/python evaluate_v2.py

# 有数据库时追加 V2 Knowledge 资格与词法检索基线
.venv/bin/python evaluate_v2.py --database-url postgresql://...
```

`evaluate_v2.py` 只做完整性检查、sidecar 校验和（可选的）retrieval/trust 基线；
Phase 3.0 没有 V2 Answer Service，不生成、不伪造任何 V2 答案准确率。第一次
真实端到端 QA baseline 在 Phase 3.1 实现 answer service 后产生。

## 测试

完整回归测试：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m unittest discover -p 'test_*.py'
.venv/bin/python -m compileall -q app.py llm.py reembed.py worker.py v2
```

PostgreSQL 集成测试需要设置 `V2_TEST_DATABASE_URL`；测试使用外层事务并回滚写入。

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
.venv/bin/python -m pip install -r requirements.txt
DATABASE_URL=postgresql://... .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
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
