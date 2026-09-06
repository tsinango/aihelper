# Phase 3–5 统一交付报告

日期：2026-09-06。范围：`astra.md` Phase 3.1 硬化 → Phase 5.3。分支 `main`，
8 个阶段各自独立 commit + push，无堆积提交。V1 路由、数据与测试全程未动。

结论先行：闭环机器已建成并经生产验证（提问→有据回答→纠正→经验→复测→
原文回读→版本重验→失败分类），但**尚未达到“值得工程师真实日常使用”**，
缺三个人工门：可用学习模型决策、真实手册/PPTX 全文验收、连续 5 工作日试用。
详见 §7、§10。

## 1. Commit timeline

所有 push 均成功（`To aihelper:tsinango/aihelper.git main -> main`）。

| Phase | Commit | Push 后 | 变更文件 |
| --- | --- | --- | --- |
| 3.1 hardening | `7be2bb1` Harden Phase 3.1 answer evidence gate | ✓ | `astra.md`（Step 0/1 记录）、`test_production_hardening.py`（去掉过时 Chat 导航排除断言） |
| 3.2 | `6cc1100` Complete Phase 3.2 correction learning loop | ✓ | `migrations/022`、`v2/feedback.py`（新）、`v2/answering.py`、`v2/learning.py`、`v2/retrieval.py`、`v2/service.py`、`app.py`（7 个 API）、`templates/chat.html`（纠正 UI）、`templates/inbox.html`（缺口筛选）、`test_v2_feedback{,_postgres}.py`、文档 4 件 |
| 4.1 | `ea64ab2` Add Phase 4.1 structured PDF and PPTX intake | ✓ | `migrations/023`、`v2/documents.py`（新）、`v2/document_processing.py`（新）、`worker.py`（inbox 优先分支）、`requirements.txt`（pin 解析器）、`templates/documents.html`（上传/结构）、`test_v2_documents.py`、`ready()` 新增表检查、文档 4 件 |
| 4.2 | `352db94` Add Phase 4.2 complete document knowledge units | ✓ | `migrations/024`、`v2/document_learning.py`（新）、learn 分支/API（提炼/提案/确认）、回答 `validated` 门槛、`templates/documents.html`（提案确认）、`templates/knowledge.html`（步骤展示）、`test_v2_document_learning.py`、文档 4 件 |
| 4.3 | `e2dbd09` Complete Phase 4 document learning coverage | ✓ | `worker.py`（`pump_once` 可测单步）、`v2/document_processing.py`（evidence_only 归档）、`v2/documents.py`（coverage）、`v2/document_learning.py`（重学重置）、`v2/service.py`（适用范围/结构变更补历史）、coverage API+UI、`test_v2_document_versions.py`（新）、文档 4 件。无新迁移 |
| 5.1 | `b9ba0b6` Add Phase 5.1 bounded source fallback | ✓ | `v2/retrieval.py`（`retrieve_document_evidence` + 确定性触发）、`v2/answering.py`（1+1 有界组合）、`check_sources` 参数 + Chat 核对按钮 + 文档引用展示、`test_v2_document_fallback.py`（新）、文档 4 件。无新迁移 |
| 5.2 | `db10d92` Add Phase 5.2 document version revalidation | ✓ | `migrations/025`、`v2/documents.py`（对比/影响/lineage/记录）、`v2/retrieval.py`（请求版本门槛）、`v2/answering.py`（版本上下文 + 快照来源）、`v2/service.py` + PATCH（验证状态切换）、impact/revalidate API + UI、`test_v2_document_versions.py` 扩展、文档 4 件 |
| 5.3 | `e5ec6d6` Complete Phase 5 failure-driven improvement loop | ✓ | `migrations/026`、`v2/feedback.py`（六分类/门限计数/expected ids）、`GET /api/v2/failures`、Inbox 失败视图、`evaluate_v2.py --failure-report`、`test_v2_failures.py`（新）、文档 4 件 |

每个 commit 单独可理解、可回退（回退办法见 OPERATIONS 各节），不混入下一阶段代码。

## 2. Final architecture

真实流程（全部已实现，括号内为实际载体）：

```
Sources（Inbox 文本 / 纠正 / PDF+PPTX 版本+结构块）
  → Knowledge / Experience（v2_knowledge：trust × unit_kind × revision ×
     validation × origin；提案经人工确认才可信）
  → Answer（retrieve_for_answer 资格门槛 → 一次有据生成 → 引用校验；
     v2_answer_runs 存不可变证据快照）
  → Correction（Chat 纠正：reply_only / save_experience / 5 种缺口；
     v2_answer_feedback，亦为缺口队列）
  → Retest（永远新 run，经 retest_of/feedback_id 链接；人工 verdict）
  → Document learning（版本→结构块→小节上下文→整单元提案→确认 validated）
  → Source fallback（无知识/明确核对/高风险时读合格原文块，最多再生成一次）
  → Version revalidation（新版只增不改；章节对比→影响清单→记录前驱；
     请求点名新版时排除旧版链单元）
  → Failure loop（六分类只标注；10 例漏召回门限未满，检索保持不动）
```

单 FastAPI + 单 PostgreSQL + 单 worker + OpenRouter-only。无独立向量库、
无 agent 框架、无 ontology/GraphRAG、无模型路由、无自动客户回复。

## 3. Database changes

迁移 `apply_migration.py` 校验 SHA-256，生产库与测试库均应用到 026
（`schema_migrations` 26 行）。022–026 全部 additive，只加表/加列/放宽
CHECK，无回填、无 V1 表变更：

- `021`（既有）：`v2_answer_runs`（幂等键唯一、证据快照）。
- `022`：`v2_answer_feedback`（纠正+缺口队列，幂等键唯一）；Knowledge/
  提案加 `unit_kind`/`applicability`/`revision`，提案 `thread_id` 可空；
  runs 加 `retest_of`/`feedback_id`/人工判定列；history 动作加 `confirm`。
- `023`：`v2_document_versions`（不可覆盖：key+label 唯一）、
  `v2_document_blocks`（结构+原文外键+处理状态）、`v2_document_jobs`
  （检查点+幂等+退避）；Knowledge 加 `origin_document_version_id`/
  `validation_status`（NULL=非文档行，走旧门槛）。
- `024`：Knowledge + 提案加 `details_json`，提案加
  `origin_document_version_id`。
- `025`：版本加 `previous_version_id` + `change_summary`；history 动作加
  `revalidate`。
- `026`：feedback 加 `expected_knowledge_ids`（门限计数用）。

## 4. Tests

基线：无 DB 时 417 passed + 47 skipped（PostgreSQL 集成 skip），
`V2_TEST_DATABASE_URL` 下 464 passed；`unittest discover` 与 `compileall`
全绿，`git diff --check` 全绿。每阶段提交前均执行。

- 3.1：evidence_status 门限 pg 回归；30 例 evaluation（见 §5）。
- 3.2：`test_v2_feedback.py` 27（校验/幂等/版本冲突/缺口/判定/路由）+
  `test_v2_feedback_postgres.py` 8（暂存不可答→确认可答、重复提交、409、
  retest 链接、verdict、revision）。
- 4.1：`test_v2_documents.py` 14（in-test 生成 PDF/PPTX fixture：标题/表格/
  合并单元格/备注/图片/空 slide/扫描页标记；持久化/幂等/冲突/损坏失败/
  路径穿越/路由）。
- 4.2：`test_v2_document_learning.py` 20（上下文分组、提炼形状、引用逐字/
  标识符/数字/结构校验、渲染确定性、确认幂等、验证门限、toy 全链路）。
- 4.3：`test_v2_document_versions.py`（`pump_once` inbox 优先、崩溃续跑、
  evidence_only 归档、重学去重、编辑一致性、fixture 问答集）。
- 5.1：`test_v2_document_fallback.py` 16（触发、预算、整块装配、资格门、
  不降级、矛盾呈现、版本作用域、pg 救援/未核实排除）。
- 5.2：版本对比（增/改/删/失配、页码不敏感、全局保守影响）、lineage 门限、
  验证切换与历史、快照来源（pg）。
- 5.3：`test_v2_failures.py` 8（六分类、列表/门限、expected 校验、路由、
  evaluate 标志）。
- UI：`test_v2_ui.py` 契约断言 + node 内联脚本语法检查；Playwright 真实
  客户端 A–G 轮（导航、问答、纠正全环、上传解析、覆盖率、原文核对）0 页面错误。

## 5. Real QA evaluation

模型：`nvidia/nemotron-3-ultra-550b-a55b:free`；问答 prompt `v2-answer-2`；
提炼 prompt `v2-doc-extract-1`（一例 one-off，见下）。`human_verdict`/
`reviewer_verdict` 全部由人填写，模型不自评。

- 3.1 基线（`data/v2_eval_phase31_step1.json`，报告 gitignored）：30/30
  `pending_expert_mapping`，28 次调用，status_match 4/30（全为预期
  unsupported），mechanical critical_flags 全 0。**0 不代表答案全对**（多为
  正确 fail-closed）；4/30 不代表损坏（sidecar 无专家映射）。
- 3.2 真实验收（`data/phase32_acceptance.json`）：17 runs / 8 feedback。
  C1 reply_only 不污染；C2 新经验 K211（r2→r3 两次细化）复测复用
  （R177 引用 K211/K150/K151）；重复提交/确认幂等；C3 范围经验 K212 使改写
  与反例正确限定 TandemVu；C4 K143 适用范围更新 r2 后快照带 revision；
  相邻型号反例正确拒答；1 次 service_error 被隔离未判分。
  真实发现：decline 型经验对“是否”问句敏感（R174 verdict fail + F8 缺口，
  5.3 已闭环见下）；草稿偶带俄语脚手架碎片，非关键错误。
- 4.2 真实验收（`data/phase42_acceptance.json`，提炼系 one-off 调用，
  部署默认未动）：2 原始单元→1 驳回（无 trigger 的 rule）→P385 procedure
  与 P387 表格 fact 确认后问答命中引用；toy 行已停用（历史保留）。
- 4.3（`data/phase43_acceptance.json`）：E2E 版本覆盖率 3/6 与 4/7，
  complete=false 如实；失败任务带模型 404 终态。
- 5.1（`data/phase51_acceptance.json`）：run 186 核对保持草稿 + trace 留痕；
  rescue 在 pg 证明，生产无合格文档故诚实未证。
- 5.2（`data/phase52_acceptance.json`）：E2E-MANUAL v2 对比（增 Appendix/
  Warnings、改 Guide、删 Fingerprints）与前驱记录完成；冒烟问答正常。
- 5.3：30 天 148 例未判定失败如实分类（§7 前）；R179 重试解决、R174 经
  R177/R190 复测通过、F8 关闭留因；门限 0/10。

完整 answer/reason/citation/snapshot/latency/verdict 见上述 JSON（本机保留，
供人工统一验收）。

## 6. Document evaluation

- 解析：PDF（页/标题层级/段落/表格/图片定位）与 PPTX（顺序/标题/文本框/
  表格合并格/备注/图片/空 slide）in-test fixture 全绿；扫描式图片页标
  `image_only_page`，未知图形标 unexplained，空 slide/页标待人工——无一冒充理解。
- 生产 E2E（fixture，非业务资料）：E2E-MANUAL 6 块（5 pending + 1 待人工），
  E2E-TRAINING 7 块（6 pending + 1 待人工）；图片资源落盘可下载；解析器版本
  已记录。提炼：3 上下文→4 提案（1 驳回、1 空上下文 0 产出无幻觉）；确认 2
  个后问答引用命中；版本 v2 对比与前驱记录完成。
- 失败页/slide：损坏 PDF 任务退避重试后终态失败，版本标 parse_failed，错误可查。
- 溯源：单元→块→原文证据→版本全链；重传/重学指纹去重无重复知识；中断后按
  上下文检查点续跑（pg 证明）；inbox 永远优先（`pump_once` 回归）。
- 版本：v1→v2 增/改/删/失配全检出；页码移动不算变化；全局节变化保守全影响；
  旧单元未被触碰；请求点名新版时旧版链排除（pg 证明）。

## 7. Remaining manual gates

以下必须由人完成，模型与代码不能替代：

1. **可用学习模型决策**：默认 `V2_LEARNING_MODEL=openai/gpt-oss-20b:free`
   已在 OpenRouter 免费层下线（404，worker 任务如实失败），旧文本学习同受
   影响。选定可用免费模型（或单模型复用问答模型）后 retry 失败任务。
2. **真实手册/PPTX 全文验收**：仓库无真实业务资料。需 ≥1 真实 PDF + ≥1 真实
   PPTX 走完上传→提炼→确认→问答→复测（含 4.2 要求的门禁主题与 ≥20 文档问题）。
3. **连续 5 工作日试用**：≥20 真实问题，≥5 失败走完纠正与人工复测。
4. **10 例漏召回证据**：当前 0/10，满额前不得改检索。
5. **30 题 sidecar 专家映射**：当前 30/30 `pending_expert_mapping`；
   `human_verdict` 待领域专家统一填写。

## 8. Scope audit

- 独立向量库：无（沿用 pgvector 精确扫描；新依赖仅 pdfplumber/python-pptx 解析器）。
- ontology/knowledge graph/GraphRAG/agent/multi-hop：无（命中均为否定性文字与
  禁止名单测试）。
- 模型路由/多模型选择：无（问答与提炼各单一模型；one-off 验收模型如实记录，
  未改部署默认）。
- 自动客户回复：无（V1 `query()` 未动；仅内部工程师草稿可读可信 Knowledge）。
- 破坏性 V1 迁移：无（022–026 只碰 `v2_` 表与 history CHECK；V1 表零变更；
  `legacy_document_id` 只读关联）。
- 自动提 trust/自动合并/自动清理/静默截断：无（确认全经人工；去重只作用于
  同版本同指纹提案；超限报错；整块引用）。
- app.py 相对 3.1 基线 +686/−5，5 处删除全为扩展（import/Form、ready 表、
  edit 参数、documents 信封），V1 行为零变更（回归全绿）。

## 9. Remaining risks

1. 学习模型下线导致新学习停滞（含旧文本路径），直到人工选定替代模型。
2. judge 对 decline 型经验的“是否”问句敏感：知识清单式问法可答，直接问句
   可能拒答（已分类、可复现、待生成侧处理）。
3. 草稿偶带俄语脚手架碎片（`Если речь о ...`），质量瑕疵，非事实错误。
4. 仓库尚无真实业务文档：解析器在真实扫描件/复杂版式上的表现未经验证；
   失败会如实标记但需人工兜底。
5. 148 例历史未判定失败多为未覆盖的真实问题：这是诚实的覆盖缺口 backlog，
   不是回归；需持续补资料/经验。
6. 单人维护：表增至 20+（`v2_` 15 张），但每阶段回退点独立，风险可控。

## 10. Recommended next decision

**aihelper 尚未达到“值得工程师真实日常使用”。** 机器已就绪，差三个人工门
（§7.1–7.3）。建议顺序：先定学习模型（解冻学习与重跑失败任务）→ 上传 1 份
真实核心手册走完 4.3 全文验收（同时产出 sidecar 专家映射与文档问题集）→
开始 5 天试用计数。**不要自行设计 Phase 6**：下一步是人的验收，不是新功能。
试用达标前，不扩检索、不扩模型、不接对外自动回复。
