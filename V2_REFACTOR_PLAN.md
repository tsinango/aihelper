# aihelper V2 重构计划

更新日期：2026-09-05

## 当前实现状态（2026-09-05）

本计划中的 Phase 1、Phase 2 和 Phase 2.2 已在当前 `main` 完成，Phase 2.2 已
正式收口，当前处于 **Phase 3.0**（Organization 收口 + UX Gate + 评测基线，
详见 [astra.md](astra.md)）。后续阶段（Phase 3.1 Read-only Internal QA、
3.2 纠正与复测、Phase 4/5）按 `astra.md` 执行；本文不再包含“禁止进入 Phase 3”
的限制。V2 Answer Service 尚未实现。当前生产闭环是：

```text
Raw Evidence
  → 一次 OpenRouter Structured Learning
  → claims / semantic consolidation / 俄语 canonical knowledge
  → Pending Knowledge
  → 用户确认
  → Knowledge + provenance
  → 确定性精确 Entity 关联（自动 LLM organization 已默认关闭）
```

已完成的 V2 能力包括：

- `migrations/013`–`020`：V2 skeleton、comparison、bulk intake、durable Inbox job、
  worker heartbeat、轻量 Organization、Knowledge history 和安全 Entity pruning。
- `aihelper.service` 与 `aihelper-inbox-worker.service` 两进程生产架构；`/health`、
  `/ready` 和 worker heartbeat 检查。
- Learning 使用 `openai/gpt-oss-20b:free` Structured Outputs。正常路径一次调用，
  contract 失败最多一次 repair；失败时只保留 Raw Evidence，不做原文 fallback。
- **Phase 3.0 组织层收口**：确认 Knowledge 后不再自动调用 LLM 做结构整理
  （内部开关 `V2_ORGANIZATION_LLM_ENABLED` 默认关闭，仅用于回退验证）。
  精确 Entity 关联、Entity/Relation 数据、provenance、防环和人工 tree/move/prune
  API 全部保留；不删除表、不迁移 relation、不新增 relation 类型。
- **技术失败与业务歧义分离**：compare judge 的 provider/解析/contract 失败标记
  `technical_failure`，Inbox job 失败可重试，不再伪装成产品澄清问题。
- **评测基线**：`evaluate_v2.py` + `data/v2_eval_cases.json` sidecar 建立固定
  30 条评测输入规范与完整性检查；Phase 3.0 不伪造 V2 答案基线，第一次真实
  QA baseline 在 Phase 3.1 产生。
- Knowledge 页面支持搜索、编辑、软删除、恢复、Entity 移动、来源和简单历史。
  人工编辑不调用 LLM，内容变化会使 embedding 失效。
- Entity Tree 保持局部、轻量和可审计。只有完全没有 active/deleted Knowledge 引用
  的结构才允许用户手动软删除；不做全局重排或自动 prune。

本文件后续的历史 inventory 和阶段说明仍用于解释迁移背景；若与本节冲突，以本节
和 `README.md`、`OPERATIONS.md`、`astra.md` 中的当前状态为准。

## 目标和边界

V2 的唯一产品目标是：用户把资料或一句话丢进 Inbox，AI 理解并主动提问，经过原子化复述和明确确认后内化为可追溯 Knowledge；客户问答只使用 `official_source` 或 `user_confirmed`，没有证据就拒答并生成 unresolved gap。

V2 不把用户暴露给 `candidate`、`knowledge_key`、`scope`、taxonomy、topic 或 publish workflow。V1 数据和代码先保留，V2 通过独立表、模块和路由隔离主流程。

## Phase 0 inventory

### 仓库和运行方式

- 仓库是单个初始提交的 Python FastAPI 服务；HTTP 主入口是 [app.py](app.py)，当前约 3900 行，职责包含 Telegram、Grounded QA、retrieval、review API 和数据库初始化。
- [README.md](README.md) 和 [TECHNICAL_STATUS_AND_REMEDIATION.md](TECHNICAL_STATUS_AND_REMEDIATION.md) 记录的生产模型入口为 OpenRouter；`llm.py` 固定 Nemotron 生成模型，`embeddings.py` 固定 2048 维 OpenRouter embedding，`rerank.py` 是可选 OpenRouter rerank。
- 仓库的 `.gitignore` 忽略 `data/`、`d1-export/`、Telegram 历史导出、`openrouter` 和 `tgtoken`。这些本地资产不能提交；token 只读，不在文档或日志中复制。
- 完整测试命令应使用 `.venv/bin/python -m unittest discover -p 'test_*.py'`；当前基线为 71 tests，全部通过。系统 `python` 命令不存在。

### 数据库和数据资产

基础 [schema.sql](schema.sql) 包含：

- 产品和资料：`products`、`documents`、`document_chunks`、`product_attributes`、`features`、`feature_aliases`、`product_features`、`verified_facts`。
- 问答和导入：`questions`、`import_batches`。
- Telegram：`support_cases`、`support_case_analysis`、`support_case_analysis_failures`。本地导出和技术状态文档显示已有 602 个派生 support case；完整原始 Telegram 归档不在当前仓库。
- V1 学习审核：`verified_knowledge_candidates`、`verified_knowledge_candidate_*`、`knowledge_review_*`、`knowledge_aliases`、`knowledge_learning_examples`、`verified_knowledge`、`case_knowledge_memory`，以及 topic abstraction / grouping 相关表。
- 评测：`local_model_eval_runs`、`local_model_eval_results`。本地 Qwen 只应继续作为影子评测资产，不进入 V2 production provider。

已有迁移 001–012 是 V1 增量迁移，涉及审核候选、发布版本、case memory、Telegram provenance、2048 维向量、review groups 和本地模型评测。V2 不修改或删除这些表；迁移 `013`–`020` 只增加或扩展 V2 表、索引和审计字段。

本地 ignored 数据资产包括：

- `data/golden_set.json`：固定 135 条 golden-v2 样本，复用而不是重建。
- `data/telegram_knowledge_review.json`：602 条历史 case 的派生 review artifact，作为保留和离线预提炼输入。
- `d1-export/`：产品、文档、support case 和历史知识的 SQL 导出。
- 历史 V1.1 intent、topic 和 local Qwen 输出：保留作审计/评测输入，但停止进入 V2 主流程。

当前没有连接本地 PostgreSQL，也没有把生产数据加载到测试中；schema 和 SQL 迁移必须继续支持 additive rollout。

### 现有在线流程

- `/api/v1/query` 在 `app.py` 中识别型号和路由，合并产品事实、已发布 `verified_knowledge`、`case_knowledge_memory`、文档 chunks 和 learning examples，再交给 OpenRouter 生成。
- 当前文档 retrieval 已有 exact/lexical、全文和 embedding 混合召回，并可调用 rerank；embedding 维度为 2048，技术状态文档说明当前 pgvector 索引条件不足时使用 exact scan。
- `/telegram/webhook` 校验 secret，把文本放入 background task，最终复用 `query()` 和 `customer_facing_text()` 回复 Telegram。这个入口是生产资产，Phase 3.5 再切换到 V2 Grounded QA；Phase 1–2 不改变 V1 webhook 行为。
- `/review` 和 `/review/published` 是 V1 复杂审核 UI，暴露了 scope、knowledge key、候选、分组、批准发布等概念。它们在 V2 初期保留，但不作为 V2 Inbox。

### 主要风险

1. V1 `case_knowledge_memory` 有 `ai_derived`、`verified` 等旧状态，不能映射后直接当作 V2 四种 trust；迁移必须默认 `provisional`，除非能提供 official 或明确用户确认 provenance。
2. V1 retrieval 的候选集合包含 case memory；V2 客户问答必须使用独立的 trust filter，不能复用“选中 case memory 即可传给 LLM”的路径。
3. V1 scope 和 topic/group 逻辑会诱发单型号到系列的泛化；V2 初期只保存 `entity_name`，不自动抽象系列范围。
4. V1 review approval 是发布动作；V2 user confirmation 是 Knowledge trust 变化，二者不等价。
5. 文档“未提到”不能生成 negative fact；只有原文明确否定或用户确认否定才允许保存否定陈述。

## KEEP / MIGRATE / REWRITE / DEPRECATE

### KEEP

- PostgreSQL、现有 additive migration 机制、`products` / `documents` / `document_chunks` 原始资料表。
- `support_cases` 及其原始 `messages` JSON、Telegram webhook secret/token 保护、日志脱敏。
- `llm.py` 的 OpenRouter-only 固定模型、`embeddings.py` 的现有 embedding 模型/维度和 exact-scan 兼容性。
- `data/golden_set.json` 135 题及现有 `test_*.py`；golden schema 和参考答案继续复用。
- V1 review 页面和历史候选/发布表，作为 legacy 审计和安全回退。

### MIGRATE

- 官方 PDF/文档元数据、原文 chunks 和 source reference 迁移为 V2 `raw_evidence` 的关联来源；原文不删除。
- Telegram 原文和 602 个 case 迁移为 raw evidence / inbox context；历史 AI 提炼默认 `provisional`，不直接回答客户。
- 已明确人工确认、已发布且 provenance 完整的知识逐条映射到 V2 Knowledge；迁移记录 source relation，不能批量假定全部可信。
- 产品型号、别名和历史问题用于 entity matching 和检索输入，但不建立新的 taxonomy/dimension registry。
- golden-v2 作为 V2 Grounded QA baseline/eval 输入，不重建 100 题。

### REWRITE

- Chat-style Inbox、学习会话、主动提问、原子复述确认、学习总结。
- 四种 trust 和 raw evidence/provenance 模型。
- 只允许 confirmed evidence 的 Grounded QA retrieval/answer gate。
- refusal → unresolved gap 的闭环（Phase 3 才进入主实现）。
- V2 简单页面和 API：`Inbox`、`Knowledge`、`Documents`、`Chat`。

### DEPRECATE

以下保留代码和表，但从 V2 主流程停止接入：

- Dimension Registry、Knowledge Gardening、Ontology、Knowledge Graph、complex taxonomy、topic abstraction。
- Knowledge intent V1/V1.1、knowledge key、review groups、candidate review workflow、publish workflow。
- 复杂 scope hierarchy、自动系列泛化、多阶段 agent/router、provider fallback。
- LongCat、NVIDIA direct、Groq、Cloudflare Workers AI、local Qwen production provider 和 multi-provider framework。
- V1 `/review` 作为 V2 用户入口；V2 入口是 `/inbox`。

## 最小 V2 schema

第一版只增加以下表；它们是内部实现细节，UI 不显示技术字段。

```sql
v2_raw_evidence
  id, evidence_type, author_role, content, raw_payload, source_label,
  source_locator, external_id, conversation_id, document_id,
  document_chunk_id, telegram_chat_id, telegram_message_id,
  reply_to_telegram_message_id, evidence_status, captured_at, created_at

v2_knowledge
  id, title, content, entity_name, entity_id, trust, active,
  embedding, embedding_model, created_at, updated_at

v2_knowledge_sources
  id, knowledge_id, raw_evidence_id, source_kind, relation, source_role,
  excerpt, active, resolution, created_at

v2_inbox_threads
  id, thread_type, origin, status, external_thread_id, created_at, updated_at

v2_inbox_messages
  id, thread_id, sequence_no, role, message_type, content, raw_evidence_id,
  message_status, created_at

v2_learning_proposals
  id, thread_id, source_message_id, question_message_id, fact_text,
  entity_name, proposed_trust, status, confirmed_knowledge_id,
  resolution_message_id, created_at, updated_at

v2_learning_sessions
  id, thread_id, session_type, status, question_budget, questions_asked,
  summary, started_at, completed_at, created_at, updated_at

v2_learning_batches
  id, thread_id, raw_evidence_id, raw_source, total_segments,
  processed_segments, failed_segments, clear_facts, unclear_items,
  conflicts, status, confirmation_message_id, created_at, updated_at

v2_inbox_processing_jobs
  id, thread_id, raw_evidence_id, user_message_id, idempotency_key,
  status, error_message, attempts, created_at, started_at, completed_at,
  updated_at

v2_inbox_workers
  worker_name, last_seen_at

v2_entities
  id, name, normalized_name, entity_type, active, deactivated_at,
  created_at, updated_at

v2_entity_relations
  id, parent_entity_id, child_entity_id, relation_type, source_id,
  provenance, provenance_kind, active, deactivated_at, created_at, updated_at

v2_knowledge_history
  id, knowledge_id, action, before_json, after_json, created_at
```

约束：`trust` 只能是 `official_source`、`user_confirmed`、`provisional`、`conflicted`；`active=false` 可让被用户否定的 source/knowledge 退出生产 retrieval，但原始 evidence 和 resolution history 永久保留。`v2_knowledge_sources` 是多来源 provenance，不把来源文本塞进 Knowledge 内容后丢失关系。

`v2_learning_proposals` 不是用户需要维护的 candidate workflow，而是 Inbox 对话中等待确认的短暂内部状态。长文本可以先语义归并为多个 Knowledge Unit，用户可以在确认前查看并编辑或删除本批次 proposal；确认后写入 `v2_knowledge` 和 source relation，跳过则记录并停止机械重复。`v2_learning_sessions.question_budget` 保存主动提问预算，默认值通过应用配置传入。

## Phase 1–2 vertical slice

### Phase 1：V2 Skeleton

1. 增加 `migrations/013_v2_skeleton.sql`，只创建上述最小表和 trust/status/message type 约束。
2. 增加 `v2/` 下的 db/repository、页面路由和简单模板；不改 V1 `/api/v1/query`、`/telegram/webhook`、`/review`。
3. 建立 `/inbox`、`/knowledge`、`/documents`、`/chat` 和 `/api/v2/inbox`、`/api/v2/knowledge`、`/api/v2/documents`、`/api/v2/chat` 最小可用入口。
4. Inbox 顶部只显示知识库数、本周新增、待确认和未解缺口（Phase 1 未解缺口为 0 或兼容读取），提供文字输入框和会话消息流。
5. 页面不暴露 candidate、knowledge key、scope、taxonomy 或发布按钮。

验收：空库可以打开四个页面；提交文字能建立 Inbox thread/message；V1 测试和旧路由保持通过。

### Phase 2：最小学习闭环（已完成）

1. 仅处理手工文字，不处理 PDF、Excel、CSV、Telegram webhook 或历史批处理。
2. 接收输入并保存 raw evidence + user message；先按实体和规范化内容查 active V2 Knowledge，避免重复创建。
3. 若无法理解或发现需要专家判断，OpenRouter 返回一个或多个语义 Knowledge Unit；UI 只在必要时提出确认或澄清。
4. 用户可以回答、不知道、跳过、以后再说。跳过写入 session/proposal 状态，后续没有新 evidence/context 时不重复问。
5. AI 将相关 claims 归并为语义 Knowledge Unit，并一次展示可确认的整理结果。只有用户明确确认后才创建或更新 `user_confirmed` Knowledge；随手输入和 AI 推断均为 `provisional`。
6. 不同型号、独立参数、不同条件/版本和矛盾事实必须分开；同一能力的解释和直接效果可以合并。没有显式证据不做系列泛化，不把手册未提到转成不支持。
7. 对新输入执行内部 `NEW` / `CONFIRM` / `ENRICH` / `CONFLICT` / `UNCLEAR` 判定；低 trust 不能直接 enrich 高 trust，冲突保留双方 source 并继续提问。
8. 明显学习会话结束时输出“今天我学会了… / 还有…未确认”的自然语言总结。

验收重点是 UX：至少 20 条真实产品知识人工喂入，覆盖明确、模糊、Telegram 碎片、冲突、型号/系列、可能、大概、否定、新旧版本和连续追问。Phase 2 测试通过后进入人工 UX Gate；当前不自动做 Phase 3。

### Phase 2.2：轻量组织与 Knowledge 维护（已完成）

1. `entities` 和 `entity_relations` 只表达局部 `belongs_to` 结构；Fact Layer 的 Knowledge 内容不因组织变化而搬迁或重写。
2. 每次确认后只对当前 Entity 附近做 local review；大多数结果为 `NO_CHANGE`，失败不影响 Knowledge 保存。
3. Relation 必须有 `official_source` 或 `user_confirmed` provenance；保留 inactive relation 历史并阻止 cycle 和无证据推断。
4. Knowledge 页面提供直接编辑、软删除、恢复、Entity 移动、来源和历史；内容编辑会清空旧 embedding，Raw Evidence 保持不可变。
5. Entity 结构只允许用户手动 prune 完全为空的 active subtree。active 或 deleted Knowledge 引用、非空子树和无法安全判断的情况都会拒绝操作。

Phase 2.2 已收口并进入 Phase 3.0。后续阶段按 `astra.md` 的顺序执行：
Phase 3.1 Read-only Internal QA → 3.2 纠正与复测 → Phase 4 → Phase 5。

## 提交和发布纪律

- Phase 0：本文件和 [V2_FUTURE_DESIGN_NOTES.md](V2_FUTURE_DESIGN_NOTES.md) 单独 commit 并 push。
- Phase 1：skeleton 代码、migration、页面和测试单独 commit 并 push。
- Phase 2：学习闭环代码、trust 测试和测试报告单独 commit 并 push。
- 每个 commit 前运行 `.venv/bin/python -m unittest discover -p 'test_*.py'`，检查 `git diff --check` 和 `git status`；不提交 ignored secrets、data export 或临时文件。
- Phase 2.2 已收口；Phase 3.0（组织层收口 + UX Gate + 评测基线）已实施，后续按
  `astra.md` 进入 Phase 3.1，不再停留在“禁止 Phase 3”的状态。
