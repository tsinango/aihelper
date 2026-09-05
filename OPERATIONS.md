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
read it). Provider and model slugs are
fixed in source.

Required variables:

```text
OPENROUTER_TOKEN_FILE=/opt/aihelper/openrouter
OPENROUTER_TIMEOUT_SECONDS=120
OPENROUTER_RERANK_ENABLED=true
INBOX_WORKER_NAME=aihelper-inbox-worker
INBOX_WORKER_HEARTBEAT_INTERVAL_SECONDS=10
INBOX_WORKER_HEALTHY_THRESHOLD_SECONDS=45
```

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
