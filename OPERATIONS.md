# aihelper operations

Production consists of two systemd processes:

- `aihelper.service` — FastAPI web backend on `127.0.0.1:8000`.
- `aihelper-inbox-worker.service` — durable PostgreSQL-backed Inbox worker.

Start the web service:

```bash
sudo systemctl start aihelper
```

Start the Inbox worker:

```bash
sudo systemctl start aihelper-inbox-worker
```

Stop both services before maintenance:

```bash
sudo systemctl stop aihelper-inbox-worker
sudo systemctl stop aihelper
```

Restart:

```bash
sudo systemctl restart aihelper
sudo systemctl restart aihelper-inbox-worker
```

Status:

```bash
systemctl status aihelper
systemctl status aihelper-inbox-worker
```

Logs:

```bash
journalctl -u aihelper -f
journalctl -u aihelper-inbox-worker -f
```

## Service and environment migration

The checked-in units are `deploy/aihelper.service` and
`deploy/aihelper-inbox-worker.service`. The Python executable below is the
confirmed existing `/opt/aihelper/.venv/bin/python` path; it resolves to
`/usr/bin/python3.10` in this checkout.

If the old environment file is still the only copy, preserve it and create the
new protected file:

```bash
sudo cp /etc/ai-sales-engineer.env /etc/aihelper.env
sudo chown root:ubuntu /etc/aihelper.env
sudo chmod 640 /etc/aihelper.env
sudo stat -c '%n %U:%G %a' /etc/aihelper.env
```

Do not store the environment file or its secrets in Git. Keep the old file as
a backup until the migration has been verified; it is not used by the new
units.

Apply the additive heartbeat and organization migrations before starting the
new worker:

```bash
sudo -u ubuntu /opt/aihelper/.venv/bin/python /opt/aihelper/apply_migration.py \
  --env-file /etc/aihelper.env migrations/017_v2_inbox_worker_heartbeat.sql
sudo -u ubuntu /opt/aihelper/.venv/bin/python /opt/aihelper/apply_migration.py \
  --env-file /etc/aihelper.env migrations/018_v2_organization.sql
sudo -u ubuntu /opt/aihelper/.venv/bin/python /opt/aihelper/apply_migration.py \
  --env-file /etc/aihelper.env migrations/019_v2_knowledge_history.sql
sudo -u ubuntu /opt/aihelper/.venv/bin/python /opt/aihelper/apply_migration.py \
  --env-file /etc/aihelper.env migrations/020_v2_entity_pruning.sql
```

Install and switch units without allowing both web services to bind port
8000. Stop the old Web unit first. If an old worker unit exists, stop and
disable it too; otherwise inspect the exact old worker PID and stop it before
starting the replacement:

```bash
sudo systemctl stop ai-sales-engineer.service
sudo systemctl stop ai-sales-engineer-worker.service 2>/dev/null || true
ps -ef | rg '[w]orker.py'
# If a worker PID remains after the unit stop, verify it is the old
# /opt/aihelper/worker.py process before stopping it:
sudo kill <old-worker-pid>
sudo install -o root -g root -m 0644 deploy/aihelper.service /etc/systemd/system/aihelper.service
sudo install -o root -g root -m 0644 deploy/aihelper-inbox-worker.service /etc/systemd/system/aihelper-inbox-worker.service
sudo systemctl daemon-reload
sudo systemctl enable aihelper.service aihelper-inbox-worker.service
sudo systemctl start aihelper.service
sudo systemctl start aihelper-inbox-worker.service
sudo systemctl status aihelper --no-pager
sudo systemctl status aihelper-inbox-worker --no-pager
# Only after the new units are verified, prevent the old units from returning:
sudo systemctl disable ai-sales-engineer.service ai-sales-engineer-worker.service 2>/dev/null || true
```

Do not remove the old units or environment backup until `/health`, `/ready`,
and Inbox polling have been verified after the switch.

After a reboot, verify both units are enabled and active, then check the
endpoints and recent journal output:

```bash
systemctl is-enabled aihelper aihelper-inbox-worker
systemctl is-active aihelper aihelper-inbox-worker
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
journalctl -u aihelper -b --no-pager -n 50
journalctl -u aihelper-inbox-worker -b --no-pager -n 50
```

If the web service is active but Inbox jobs do not progress, inspect the
worker status and journal first. Durable jobs remain in PostgreSQL and can be
checked with:

```bash
systemctl status aihelper-inbox-worker --no-pager
journalctl -u aihelper-inbox-worker -n 100 --no-pager
sudo -u postgres psql -d ai_sales_engineer -c \
  "select id,status,attempts,updated_at from v2_inbox_processing_jobs order by updated_at desc limit 20"
```

Use the Inbox retry action after the worker is healthy; do not run a second
worker manually against the same queue.

Health checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Knowledge maintenance is human-initiated.  Deleting a Knowledge item is a
soft delete, so its raw evidence, sources, and Entity remain available for
audit or restore.  The Knowledge page only offers `删除` for an Entity whose
entire active subtree has no Knowledge references; both active and deleted
Knowledge references block pruning.  A prune request is checked again inside
the database transaction and returns a conflict if the state changed after the
page was loaded.  Pruning only sets `active=false` and `deactivated_at` on
Entities and relations; it never physically deletes rows.

PostgreSQL check:

```bash
sudo -u postgres psql -d ai_sales_engineer -c "select count(*) from document_chunks;"
sudo -u postgres psql -d ai_sales_engineer -c "select count(*) from document_chunks where embedding is not null;"
```

OpenRouter configuration uses the raw token file
`/opt/aihelper/openrouter` (mode 600) and the protected environment
file `/etc/aihelper.env` (root:ubuntu, mode 640 so the systemd service user can
read it). OpenRouter is the only provider. V1 generation, embeddings, and
optional reranking use the fixed free slugs in source; V2 Learning extraction
defaults to the free Structured Outputs model below and must not be changed to
a paid fallback.

Required variables:

```text
OPENROUTER_TOKEN_FILE=/opt/aihelper/openrouter
OPENROUTER_TIMEOUT_SECONDS=120
OPENROUTER_RERANK_ENABLED=true
V2_LEARNING_MODEL=openai/gpt-oss-20b:free
INBOX_WORKER_NAME=aihelper-inbox-worker
INBOX_WORKER_HEARTBEAT_INTERVAL_SECONDS=10
INBOX_WORKER_HEALTHY_THRESHOLD_SECONDS=45
# Phase 3.0 internal rollback lever only; automatic LLM organization after
# Knowledge confirmation stays off in production. Do not set to true unless
# verifying the pre-3.0 behavior.
# V2_ORGANIZATION_LLM_ENABLED=false
```

## V2 评测基线（Phase 3.0 起为上线门槛）

`data/golden_set.json`（135 条）不可变；V2 评测输入规范与固定选样保存在 sidecar
`data/v2_eval_cases.json`（30 条：15 可回答 / 5 必须澄清 / 5 无依据 / 5
型号/条件边界），只补充 V2 映射、人工预期、改写问法和禁止断言，不覆盖 golden。
当前映射状态为 `pending_expert_mapping`，由领域专家在 Phase 3.1 前补齐。

无网络、无模型、无数据库即可运行的完整性与选样检查（退出码非零表示失败）：

```bash
.venv/bin/python evaluate_v2.py
```

有数据库时追加 V2 Knowledge 资格（trust、accepted supports 来源）与词法检索
召回基线：

```bash
.venv/bin/python evaluate_v2.py \
  --database-url postgresql://... \
  --report data/v2_eval_report.json
```

Phase 3.0 尚无 V2 Answer Service：runner 不生成、不伪造 V2 答案准确率，
`golden_reference` 不会被当作模型答案。第一次真实端到端 QA baseline 在
Phase 3.1 实现 answer service 后产生，成为后续版本的比较基准。

## V2 内部问答评测（Phase 3.1）

`evaluate_v2_answers.py` 把固定的 30 条 sidecar 用例和可选的补充问题送入
只读 Answer Service（真实模型与 embedding），记录 answer status、evidence
snapshot、latency 和确定性 critical flags；`human_verdict` 始终留空，由领域
专家人工填写，模型不给自己打分。每次运行写入带 tag 的 `v2_answer_runs`
行（对 Knowledge 只读），不会导入 golden_reference，也不会创建 Knowledge：

```bash
.venv/bin/python evaluate_v2_answers.py \
  --database-url postgresql://... --env-file /etc/aihelper.env \
  --extra-questions data/v2_eval_supplementary_questions.json \
  --report data/v2_eval_phase31b_report.json
```

只跑 retrieval triage、不调用模型：

```bash
.venv/bin/python evaluate_v2_answers.py \
  --database-url postgresql://... --retrieval-only
```

`--tag` 覆盖幂等键前缀（默认 `phase31-YYYYMMDD`）；同一 tag 重跑会命中幂等
返回已存 run，不重复调用模型。

## V2 纠正闭环（Phase 3.2）

Chat 回答卡新增纠正入口：`POST /api/v2/answers/{id}/feedback`
（`reply_only` 仅本次使用不碰 Knowledge；`save_experience` 暂存提案；
另有缺资料/召回失败/生成失败/现场成功/失败等缺口分类），
`POST /api/v2/feedback/{id}/confirm`（明确确认 Experience 为
`user_confirmed`，幂等，更新目标需 revision 一致），
`POST /api/v2/feedback/{id}/retest`（永远建新 run，不覆盖旧 run），
`PATCH /api/v2/answers/{id}/verdict`（人工 pass/fail），
`GET /api/v2/feedback/unresolved`（Inbox 未解决缺口筛选）。
生产验收证据：`data/phase32_acceptance.json`（gitignored，不入库）。

回退：停用 Chat 纠正按钮与上述 feedback 入口即可回到 3.1 只读问答；
已确认的 Experience 与历史记录保留，不删除。迁移 022 只加表/加列，
无回填，不阻塞旧版本代码读取（新列均有 DEFAULT）。

## V2 文档接入（Phase 4.1）

`POST /api/v2/documents`（multipart：`file` + `document_key` 必填，
`version_label` 默认为文件名，`title`/`applicability`/`source_authenticity`
可选）上传 PDF/PPTX；同 key+label+相同字节幂等返回已存版本，不同字节
409。文件按内容 hash 存于 `data/documents/v2/`，原文件名仅展示。限制：
单文件 50MB、PDF 500 页、PPTX 500 页、图片资源 100MB/版本；
超限 400，魔法字节校验失败 400。解析为后台任务（`v2_document_jobs`），
单个 worker 在 Inbox 空闲时执行，`GET /api/v2/document-jobs/{id}` 轮询，
失败可 `POST .../retry`（瞬时错误退避重试，最多 5 次后转 failed 并标版本
`parse_failed`）。Documents 页可查看版本、结构块（含待人工原因）与下载
原文（PPTX 返回正确 MIME）。

回退：停用上传入口及 worker 文档分支，Phase 3 与旧 Inbox 不受影响；
保留文件、块、来源。图片页/未知图表只标 `needs_review`，不进入回答。

## V2 文档提炼（Phase 4.2）

`POST /api/v2/documents/versions/{id}/learn` 按小节/相关幻灯片排队提炼
任务（每上下文一个 job，幂等）；worker 逐个执行一次有界提炼
（`EXTRACT_MAX_TOKENS=4000`），引用/标识符/结构由代码校验，不合格驳回；
提案在 Documents 版本详情中整项确认（可编辑文本）后成为 `validated`
可回答 Knowledge。确认幂等，`GET .../proposals` 查看整项与来源块。

注意：默认学习模型 `openai/gpt-oss-20b:free`（`V2_LEARNING_MODEL` 未设置时）
已在 OpenRouter 免费层下线，worker 提炼任务会以 404 重试至失败，版本标
`learning_failed`（旧文本学习同样受影响，需人工决策替换模型后 `retry`）。
提炼入口本身与模型无关，验收时可用任一可用免费模型手动提炼，结果以
`prompt_version` 记录为准。

## V2 全文覆盖（Phase 4.3）

`GET /api/v2/documents/versions/{id}/coverage` 返回各状态块数、未完成块
与任务状态；`complete` 仅表示每块有去向且无开放提炼任务。提炼后未被引用
的文本块标 `evidence_only`（保留为证据）；`POST .../learn` 默认只排队
pending 块，`{"reset_evidence_only": true}` 重开一轮（已完成/失败的提炼
任务行会被新一轮代替，结果已在提案中故不丢失）。worker 每次迭代只做一步，
Inbox 永远优先于文档任务（`pump_once` 回归覆盖）；崩溃后 processing 任务
回 queued，按上下文检查点续跑，不重传整本文件。

## V2 原文回读（Phase 5.1）

`POST /api/v2/answers` 新增 `check_sources`（Chat“核对原文”按钮）：无合格
Knowledge 时读合格原文回答；已有草稿时做一次核对，矛盾记
`knowledge_document_conflict` 不自动选边。调用上限：一次初始生成加一次
回读。原文资格：版本真实性已确认（上传时 `source_authenticity` 设为
`official_vendor`/`confirmed_copy`，未核实只供人工看）、解析未失败、版本
范围不冲突、块非待人工/失败；整块引用，超长章节转澄清不截断。
回退：关闭 `check_sources` 入口即回到纯 Knowledge 问答；无关 schema 变更。

## V2 文档版本重验（Phase 5.2）

`GET /api/v2/documents/versions/{id}/impact?previous_version_id=N` 只读对比
（新增/变化/删除/失配小节 + 受影响旧 Knowledge，不停用任何单元）；
`POST .../revalidate {"previous_version_id": N}` 记录版本前驱与差异摘要。
新版学习沿用 4.2（`/learn` + 整项确认，`origin` 指向新版）；旧单元只在其
自身证据失效或被明确纠正时才动。带 `document_version_id` 上下文的问答会
排除旧版链单元；`PATCH /api/v2/knowledge/{id}` 可切 `validation_status`
（pending/validated/needs_revalidation，记 `revalidate` 历史）。
回退：删除版本前驱关联（置空）即回到无链状态；知识行不受影响。

## V2 失败分类与召回门限（Phase 5.3）

`GET /api/v2/failures?days=7&limit=50` 返回近期未判定失败的六类划分与默认
动作，外加召回门限进度；`evaluate_v2.py --database-url ... --failure-report
<path>` 写同口径 JSON。给漏召回问题登记 `expected_knowledge_ids`
（`POST .../feedback` 仅 `retrieval_failure` 可带）：只有被登记且当前仍合格
的才计入门限，满 10 例才允许动检索（当前 0/10，检索保持不动）。
失败分类只标注不建知识；连续 5 个工作日、20+ 真实问题的试用仍需人工完成。

## 本机 Qwen 离线评测

本机 Qwen3.5-2B/4B 是临时的影子评测模型，不由 systemd 托管，也不接收
Telegram 或 `/api/v1/query` 流量。`evaluate_local_qwen.py` 会在同一轮中
逐个启动模型、完成请求后释放进程；它只写 `local_model_eval_runs`、
`local_model_eval_results` 和指定的 JSON 结果文件。固定 golden set 位于
`data/golden_set.json`，共 135 条真实线程引用。

先确认生产仍由 OpenRouter 提供模型，再运行完整 golden benchmark：

```bash
sudo systemctl status aihelper --no-pager
.venv/bin/python evaluate_local_qwen.py \
  --env-file /etc/aihelper.env \
  --models 2b,4b --limit 135 --mode golden \
  --output data/local_qwen_golden_v2.json
```

PostgreSQL 不在线时可使用 `--no-database`；该模式只读 golden set 和
`data/telegram_knowledge_review.json`，只输出 JSON，不写评测表：

```bash
.venv/bin/python evaluate_local_qwen.py \
  --models 2b,4b --limit 135 --mode golden --no-database \
  --output data/local_qwen_golden_v2.json
```

离线 JSON 模式每完成一条结果都会更新 output；如果进程中断，可用同样的
参数加 `--resume` 继续，已完成的 model/sample 不会重算：

```bash
.venv/bin/python evaluate_local_qwen.py \
  --models 2b,4b --limit 135 --mode golden --no-database --resume \
  --output data/local_qwen_golden_v2.json
```

默认模型路径会从 `/home/ubuntu/.cache/huggingface/hub` 自动发现；也可用
`--model-2b`、`--model-4b` 显式指定 GGUF。默认使用本地 `llama serve`、
127.0.0.1 的临时端口、2 个 CPU 线程、单并发和关闭 reasoning。若只想测一个：

```bash
.venv/bin/python evaluate_local_qwen.py \
  --env-file /etc/aihelper.env \
  --models 2b --limit 2 --mode smoke \
  --output data/local_qwen_2b_smoke.json
```

若只需快速验证服务，可继续使用 `--limit 2 --mode smoke`。正式比较应使用
`--limit 135 --mode golden`，不要用 2 条 smoke test 决定删除或保留 4B。
评测结果可以这样查看：

```bash
jq '{run_id,summary,decision}' data/local_qwen_golden_v2.json
sudo -u postgres psql -d ai_sales_engineer -c \
  "select run_id,model_name,sample_key,actual_answer_status,structure_pass,applicability_pass,generation_ms,tokens_per_second from local_model_eval_results order by id desc limit 20"
```

`retrieval_mode=golden_reference_snapshot` 表示对应 knowledge key 尚未发布，
评测使用了真实审核线程的只读参考证据；`published_vk_by_knowledge_key` 表示
命中了已发布 VK。两者都不会改变生产检索或审批行为。评测完成后应确认没有
残留本地进程：

```bash
ps -ef | rg 'llama serve|evaluate_local_qwen' || true
```

Batch Knowledge Builder throttling is controlled by
`OPENROUTER_REQUESTS_PER_MINUTE`. OpenRouter free endpoints have daily/rate
limits, so a full run may span multiple days. Checkpoints are preserved for
resume; there is no provider or model fallback. The current account's stated
daily ceiling is 1,000 requests. When OpenRouter returns HTTP 429, the V1.1
batch records a rate-limit marker, cancels queued work, exits, and preserves
the checkpoint. Re-run it on the next quota window; it will skip valid
OpenRouter results and retry unresolved cases.

To resume the V1.1 intent batch with the accelerated configuration used in
production (eight workers, twenty request starts per minute):

```bash
sudo bash -c 'set -a; . /etc/aihelper.env; set +a; export OPENROUTER_INTENT_WORKERS=8; export OPENROUTER_REQUESTS_PER_MINUTE=20; exec env PYTHONUNBUFFERED=1 /opt/aihelper/.venv/bin/python /opt/aihelper/run_knowledge_intents_v1_1.py --env-file /etc/aihelper.env'
```

The batch input currently contains 591 cases. Check progress without opening
the full sensitive artifacts:

```bash
wc -l data/knowledge_intents_v1_1_openrouter.jsonl data/knowledge_intent_failures_v1_1.jsonl
rg -c '"provider": "openrouter"' data/knowledge_intents_v1_1_openrouter.jsonl
cat data/knowledge_intent_rate_limit_v1_1.json
```

Update code from this checkout, then run
`sudo systemctl restart aihelper`. The service working directory is
`/opt/aihelper`; there is no `oracle/` or `public/` source directory
in this repository.

## Knowledge Review

Internal review UI: `/review` and `/review/published` (served by FastAPI through the web proxy).
Enter the existing API key in the page; it is kept only in browser session
storage and is not bundled into the frontend. Review APIs require the same
`x-api-key` authentication as the existing service.

Apply the additive migration and import the pilot queue with:

```bash
python apply_migration.py migrations/001_knowledge_review.sql
python apply_migration.py migrations/002_verified_knowledge_publish.sql
python apply_migration.py migrations/003_mvp_rag_memory.sql
python organize_telegram_knowledge.py
python apply_migration.py migrations/004_complete_telegram_review.sql
python import_review_candidates.py --input data/telegram_knowledge_review.json
python apply_migration.py migrations/010_review_provenance_async_embedding.sql
python apply_migration.py migrations/011_review_group_modes.sql
python import_case_memory.py
python reembed.py
```

`organize_telegram_knowledge.py` deterministically converts all 602 imported
Telegram support cases into a review workbook and one safe, case-scoped review
candidate per case. It preserves the full thread, separates follow-up questions
into atomic Q&A records, labels message roles, records customer confirmation,
and classifies the proposed scope as generic, brand, model, conditional, or
unspecified. It does not call an LLM or write to PostgreSQL. The generated
artifacts are `data/telegram_knowledge_review.json` (the import source) and
`TELEGRAM_KNOWLEDGE_REVIEW.md` (a readable audit workbook).

Migration `004` and the explicit import are idempotent. Re-import refreshes only
untouched `CASE-*` rows that remain pending/draft/non-production; it does not
overwrite human-reviewed candidates or message-role overrides.

`import_case_memory.py` is idempotent and materializes every extracted
Telegram case (including cases not present in the V2.1 artifact). It preserves
the complete thread and imports native/inferred message relations while never
overwriting manual relations. Historical answers are recall/reviewer evidence
only and cannot directly answer customers. Review approval synchronizes the
related memory to `verified`; rejection disables it. Run `reembed.py` separately
to fill OpenRouter Nemotron vectors used by retrieval; it covers document
chunks, case memory, Verified Knowledge, and approved learning examples.

Review actions keep generated snapshots and human overrides separately. Approve
creates and publishes the `verified_knowledge` record immediately after conflict
checks; there is no separate Publish step. Embedding is recorded as `pending`
and is filled asynchronously by `reembed.py`, so OpenRouter quota or embedding
failure cannot roll back human approval. Production retrieval accepts only
`published AND production_answer_allowed=true`. Published edits create a new
draft version. `knowledge_evidence` keeps case/message provenance and separates
successful and failed confirmations. `/review` also offers deterministic
grouping by default, optional V1.1/knowledge_key or OpenRouter grouping, related
knowledge suggestions, explicit merge, and optional manual message relations.
The `/review` page supports queue filters, full Telegram threads,
field/role/evidence corrections, split, negative pairs, bound aliases, and
audit/learning-example inspection. The migration runner records checksums in
`schema_migrations` and rejects changed or out-of-tree migration files.
