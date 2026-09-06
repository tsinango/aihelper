# aihelper 渐进开发计划：Phase 3–5

日期：2026-09-05。基于当前仓库的快速代码检查，以及本轮确定的产品目标。

本文是待执行的开发计划，不是已实现功能说明。本次只新增此文件，不修改运行代码、数据库或部署。以下新模块、字段、API、配置和测试名称均为拟议接口；实施时按小步骤落地，不一次性创建全部结构。

## 1. 先决定顺序

路线：**Phase 2.2 收口 → Phase 3 内部 Answer → Correction → Learn → Retest → Phase 4 PDF/PPTX 主动学习完整技术单元 → Phase 5 原文回读、文档更新和真实失败驱动的改进。**

| 关键问题 | 决定 |
| --- | --- |
| Organization 在哪里收口？ | Phase 3.0：保留精确 Entity 关联、已有关系、人工维护、来源和防环；停止确认后自动调用 LLM 整理关系，不新增关系种类和层级推断。 |
| Phase 3 先做文档还是问答闭环？ | 先完成一个仅使用已确认 V2 Knowledge/Experience 的内部问答闭环。用少量真实知识启动，不等待文档管道。 |
| 完整 Knowledge Unit 在哪个阶段？ | Phase 3 允许完整的人工 Experience 文本；Phase 4 正式增加文档用的完整流程、规则、参数单元。不能先把文档接进原子事实提取器，再以后修粒度。 |
| learning.py 什么时候改？ | Phase 3 只收口副作用和增加明确人工确认入口；Phase 4 增加文档提炼入口，绕开不适合文档的拆句、合句、单片段引用和语言硬校验。旧文本入口保持兼容。 |
| 什么时候升级 retrieval？ | Phase 3 就做可信与适用范围硬过滤；Phase 4 保留完整单元；Phase 5 完成原文回读后，只有重复的、已标注的召回失败才能触发一次有针对性的改进。 |
| 评测何时成为硬门槛？ | 从 Phase 3.0 建立基线，Phase 3.1 首次开放内部问答前执行。后续每次上线都执行，不等到 Phase 5。 |

旧 [V2_REFACTOR_PLAN.md](V2_REFACTOR_PLAN.md) 的 Phase 2.2 UX Gate 仍作为 Phase 3.0 的实际验收项目，不能用“代码已完成”替代人工试用。本文重新定义之后的 Phase 3–5 顺序；实施第一个代码提交时同步 README、旧计划与 AGENTS 的阶段描述，避免以后继续被过时的“禁止开始 Phase 3”文字误导。本轮要求已经授权编写后续计划，不需要为讨论这些阶段再次确认；并未要求本轮直接开发上线。

## 2. 当前仓库与目标的差距

以下为代码事实；没有查询生产数据库、运行付费/免费模型，或重新验证历史文档中的数据快照。

| 当前实现 | 可复用资产 | 需要补齐或改变 |
| --- | --- | --- |
| `app.py:query()` 是 V1 问答；`GET /api/v2/chat` 返回 Inbox 快照 | V1 的证据提示、俄语输出、回答状态、引用索引检查；`templates/chat.html` 的页面外壳 | V2 独立问答服务和运行记录。Chat 当前发送路径仍是学习入口，不能仅把页面加入导航就称为 Grounded QA。 |
| `v2/service.py:summary()` 的 `unresolved_gap_count` 固定为 0 | Inbox thread/message、Knowledge 维护和来源 UI | 保存真实问题、纠正和复测结果；缺口不是固定计数，也不是一个新的复杂工单平台。 |
| `v2/retrieval.py:retrieve_learning_knowledge()` 对 active Knowledge 排序，trust 仅参与排序 | 词项匹配、型号识别、同模型向量、向量失败仍可词法检索 | 增加独立的 answer 检索入口，在排名前限制 trust、来源资格和适用范围；不得直接复用学习候选作为回答证据。 |
| `v2/learning.py` 有提炼、语义拆合、俄语校验、逐 fact compare、确认和组织调用 | Raw Evidence、proposal、来源关系、明确确认、跳过、有限 repair | 完整流程不能经过拆句再拼接。一次 extraction 不等于整个学习过程一次模型请求。 |
| `v2/organization.py` 已有局部 LLM 组织、关系校验和安全维护 | 精确关联、已有来源、历史、防环和空分支软删除 | 停止自动组织扩张；不删除已有 Entity/Relation 数据。 |
| `documents` 有文件 hash/version/metadata；`document_chunks` 是 V1 检索资产 | 既有文件目录、鉴权下载习惯、只读旧资料 | 新的 V2 文件版本与结构片段。V1 上传仍为 501；V2 Documents 目前只列元数据。 |
| `worker.py` 只处理 `v2_inbox_processing_jobs` | 单进程、独立 heartbeat、重启恢复、数据库任务领取 | Inbox job 强依赖 thread/raw evidence/user message，且以之后首条 assistant 消息推测完成结果；不适合直接承载整本文档。 |
| `v2_knowledge_history` 和 `edit_knowledge()` 保存人工修改快照 | stable Knowledge ID、软删除恢复、编辑后向量失效 | 回答必须保存当时使用的内容/来源快照；完整单元的新字段必须进入历史和向量失效判断。 |
| 已有 golden set 和 `evaluate_local_qwen.py` | 真实来源、sample key、状态/来源选择/人工答案标签设计 | 新增轻量 V2 端到端 runner；不能启动本地模型比较来代替产品评测，也不能把 golden_reference 注入在线回答。 |

关键现有文件：[app.py](app.py)、[learning.py](v2/learning.py)、[retrieval.py](v2/retrieval.py)、[service.py](v2/service.py)、[processing.py](v2/processing.py)、[worker.py](worker.py)、[schema.sql](schema.sql)、[migrations/013](migrations/013_v2_skeleton.sql)、[migrations/016](migrations/016_v2_inbox_processing_jobs.sql)、[migrations/019](migrations/019_v2_knowledge_history.sql)。

## 3. 全阶段共用的实施规则

- 保留 V1 API、Telegram webhook、review UI、数据和现有测试。新的内部问答使用 V2 数据，不能调用 V1 `query()` 后再包装成 V2。
- 新迁移按实际实施时的下一个编号追加；当前目录到 020。不要修改已应用迁移或预先提交未来阶段空表。
- Experience 是 `v2_knowledge` 的一种内容类型，不建立第二套经验库、审批系统或向量库。
- 四种 trust 暂不改名、不重建。来源真实性、人工确认、版本适用性分别检查；模型不能提升 trust。
- 一个 FastAPI、一个 PostgreSQL、一个 worker、OpenRouter。新业务代码进入小模块；不趁机拆完整个 `app.py` 或重写 `learning.py`。
- 使用原有统一模型接口、有限重试和 fail-closed。没有模型服务时仍能看来源、编辑、保存人工确认内容；不能因为 embedding 失败而回滚确认。
- 文档、历史聊天、模型输出均作为数据，不执行其内嵌指令。继续使用 x-api-key、日志脱敏和忽略 secrets/data 的规则。
- 模型调用前保存任务/请求；模型调用后短事务提交结果。新增链路不跨网络请求持有数据库行锁或长事务。
- “停用”首先指停止在新路径调用；本计划没有必须物理删除的旧表、旧记录或来源文件。代码清理仅针对已证明无人调用的内部函数，单独提交。
- 各小版本分别上线，关闭新入口可以回退，保留新增数据。数据库迁移回退优先停用新功能，不删表、不逆改历史迁移。
- 同步检查 `app.py` 的启动/schema/readiness 检查及 `OPERATIONS.md`：只在新功能启用时要求其新增 schema 就绪，避免文档功能尚未启用就阻塞旧 V1/Inbox；worker heartbeat 的既有含义保持稳定。

### 3.1 评测规则：从 Phase 3 起执行

复用 `data/golden_set.json` 的 sample key 和真实来源。文件实际存在性、数量、标签质量在 Phase 3.0 检查；README 所说的 135 条不是已经全部经过领域专家认可的假设。

新增一个轻量 `evaluate_v2.py`，读取原 golden set 和本地 sidecar `data/v2_eval_cases.json`。sidecar 只补充 V2 Knowledge/来源映射、人工预期、版本、改写问题和禁止出现的断言，不覆盖原 golden 文件。不新增评测服务或本地模型生命周期管理。

3.0 先完成数据/标签核对、检索基线和 runner 的输入输出约定；当前没有 V2 问答，不能伪造 V2 答案基线。3.1 实现 answer service 后保存第一次真实端到端结果，成为 3.2 及后续版本的比较基准。

两种检查明确分开：

| 检查 | 作用 | 上线条件 |
| --- | --- | --- |
| 无网络回归：mock LLM、临时文件、测试 PostgreSQL | trust、来源、状态、并发、幂等、解析和历史完整性 | 所有相关回归通过；涉及迁移与事务时必须实际执行隔离 PostgreSQL 集成测试，不能把 skipped 当通过。 |
| 真实模型 + 领域人工判定 | 实际答案是否正确、有条件、有用 | 使用固定输入/证据快照，记录模型 ID、prompt 版本、数据版本、请求数、耗时和人工结论。无额度则保留结果等待续跑，不改用另一模型假装通过。 |

Phase 3 初始硬门槛：30 条经专家复核的问题，至少含 15 条可回答、5 条需澄清、5 条无依据、5 条型号/条件边界问题。若现有 V2 可信内容不足，先由人工确认少量真实材料补齐测试范围，不能直接把旧 golden_reference 发布为可信知识。

- 关键样本中，错误型号、编造操作、遗漏已标注的危险条件、未确认来源进入答案、越界引用：0 个。
- 15 条可回答样本至少 12 条无需实质性技术改写；全部拒答不算成功。
- 必须澄清/拒答的问题状态正确；服务失败单列，不能从统计分母悄悄删掉，也不能当作业务拒答通过。
- 每个纠正至少关联原问题、一个人工确认的同义问法，以及一个不该套用的相邻型号/条件问题。
- 语义正确与引用 ID 存在分别计分；引用检查通过不代表事实受支持。关键答案由人判定，不用生成模型自评分替代。
- 30 条属于开发集。另保留少量未用于调整 prompt 的真实问题，按线程/场景分组隔离，不把改写问题误当完全独立样本。
- 后续阶段在原基线上追加各自的文档、版本样本；原有已通过样本不得出现新的关键错误。保存适用性不满足或暂无证据样本的明确原因，不虚报整套 golden 全通过。

上述数量和比例是本项目初始验收约定，不是对当前准确率的测量，也不是上线后永远不变的性能承诺。

## 4. Phase 3：让真实问题产生可复用的纠正

### 4.1 只解决什么，以及为什么现在做

只解决：内部工程师可以提问、查看证据、纠正答案、明确保存 Experience，并验证下次能否正确复用。

现在必须做，因为文档学习的质量需要由真实问答检验。先有这个窄闭环，Phase 4 每增加一份资料都能立即测出收益，而不是继续用“新增多少 Knowledge”作为效果指标。

### 4.2 分三个可独立上线的小版本

| 小版本 | 修改边界 | 独立验收/上线结果 |
| --- | --- | --- |
| 3.0 收口与基线 | Organization 默认不再调模型；20 条现有 Inbox UX 用例；建立固定评测映射与基线；记录仍存在的学习错误 | 原有学习仍能用，确认不触发自动结构整理，新增基线报告。失败的 UX 项先修具体阻塞，不扩展整理功能。 |
| 3.1 只读内部问答 | V2 answer retrieval、answer service、运行记录和 Chat 页面 | 内部可问已有可信知识，看到适用条件和可追溯出处；不自动学习答案。 |
| 3.2 纠正与复测 | feedback、人工 Experience 确认、Retest | 同一真实错误能完成 correction → confirmed experience → 新一次问答 → 人工复测判定。 |

### 4.3 模块与数据结构

**Organization 收口。** 修改 `v2/learning.py:_run_local_organization_review()` 的调用边界，保留 `organization.py` 中不使用 LLM 的精确 Entity 关联。增加一个默认关闭的部署开关，仅用于回退验证；不向产品用户提供模型整理设置。已有 tree、移动和 prune API 保持兼容，维护操作沿用 provenance、防环和引用保护。关系不参与事实继承，父节点知识不会自动适用于所有子型号。

**问答。** 新增 `v2/answering.py`，负责一次受证据约束的回答，不承载学习流程。`app.py` 只做鉴权、参数校验和调用。参照 V1 `build_decision_messages()`、`normalize_llm_decision()` 的事实边界和状态约定；可复用的纯校验小函数单独抽取并同时跑 V1 测试，不导入整个 `app.py` 造成循环依赖，也不搬迁整条 V1 pipeline。

`v2/retrieval.py` 增加 `retrieve_for_answer()`，与 `retrieve_learning_knowledge()` 明确分开：

- 排名之前筛 `active=true` 和 `trust IN ('official_source','user_confirmed')`，并要求存在有效、可追溯的被接受支持来源。当前旧记录缺少 accepted 标记的，先诊断和人工补确认，不整体自动升级。
- 已知型号/版本/地区冲突直接排除；未知条件不能当通用。范围信息不足时给有针对性的澄清。保留少量明确通用知识的显式适用范围，不通过无型号字段推断通用。
- 词法路径可独立工作。复用现有 compatible embedding 和 exact scan；不新建向量服务、不加 reranker。
- 只保留相关候选，不为了凑固定来源配额加入证据。未确认候选仍可供学习比较使用，不能进入回答证据。

**最小新增持久化：**

| 对象 | 最小字段/变化 | 理由 |
| --- | --- | --- |
| `v2_answer_runs` 新表 | question、thread_id、context_json、request_key、execution_status、answer_status、answer_text、clarifying_question、reason_code、evidence_snapshot、trace_json、model/prompt_version、retest_of、feedback_id、人工判定、时间 | 原答案不可被纠正覆盖；证据快照含 Knowledge ID、revision、当时文本、来源 ID/摘录。服务故障也留记录。执行状态和 answered/needs_clarification/unsupported/service_error 四种业务输出分开。 |
| `v2_answer_feedback` 新表 | answer_run_id、feedback_kind、correction_text、applicability、raw_evidence_id、proposal_id/knowledge_id、status、现场结果、reviewer_label、时间 | 区分修改本次回复、保存经验、缺资料、召回失败、生成失败、现场成功/失败。一个反馈对象兼作轻量待处理事项。 |
| `v2_knowledge` 增量字段 | `unit_kind` 默认 `fact`，CHECK 从首次迁移就允许 `fact / procedure / rule / experience`，Phase 3 仅使用 fact/experience；`applicability JSONB NOT NULL DEFAULT '{}'`；`revision` 默认 1 | Experience 和 Knowledge 共用保存、检索、来源、历史；不增加第二个知识库。版本字段为明确已知信息，不强制填完整型号本体。 |
| 既有 history/proposal | proposal 追加 `unit_kind DEFAULT 'fact'`、`applicability JSONB DEFAULT '{}'`、`revision DEFAULT 1`，kind 使用相同 CHECK；history 扩充内容快照 | 编辑条件也是知识变更；旧字段和旧客户端默认行为不破坏。 |

`applicability` 仅约定当前用得到的 models、firmware/software version、region、conditions；空值表示未说明。先支持精确值和原文范围，暂不写复杂版本范围解析器；无法确定的比较交给澄清/人工。

所有修改内容/适用范围/信任资格的入口更新 revision；对已有 `_confirm()`、人工编辑、删除恢复逐一检查。历史保留旧值，内容或适用范围变化清空旧 embedding。请求或确认携带预期 revision，过期提交返回 409，避免用户确认的不是眼前那版内容。

**纠正与 Experience。** 新增小模块 `v2/feedback.py`。复用 Raw Evidence、`v2_learning_proposals`、来源关联、history 和确认事务；给出“明确提交这份完整文本”的接口，不把工程师已编辑的流程再次送去拆句、翻译和逐 fact compare。

- `reply_only`：只保存本次修改，不创建可信 Knowledge。
- `save_experience`：展示现象、环境/条件、操作或结论、现场结果、来源；用户确认确切文本后保存 `user_confirmed`。可以保持自然语言完整段落，不要求每项都拆字段。
- 若需要模型帮忙整理，结果仍是提案；模型失败可人工完成，不强制等待模型。
- 新增经验或更新某个已知 Knowledge 由用户明确选择；更新必须指定目标 ID 和 revision。不同型号、条件不得自动覆盖。未解决冲突保留双方证据，相关问题不能当作无冲突直接答。
- “工程师确认答案正确”和“客户现场验证成功”分开存；未知结果就是未知，不能自动声称已解决。
- 确认按钮就是授权保存，不再发一轮“你确定吗”的对话。保留确认记录，不用裸 `yes` 猜测多条提案中的目标。
- 共用 API key 时 `reviewer_label` 只是记录标签，不声称它提供独立身份认证；本阶段不造账号系统。
- Telegram 先支持人工粘贴一个相关完整线程作为经验来源，保存能够取得的原文、消息 ID/回复关系、时间和最终反馈；缺失的元数据不编造。复用 Raw Evidence 和同一个 Experience 确认入口，3.2 手工验收至少包含一例。自动导出导入器、媒体归档和 webhook 学习另行排期，不成为这个闭环的前置条件。

实现时遵守现有 CHECK：新增人工经验提案采用 `proposed_trust=provisional + comparison_result=NEW + status=pending_confirmation`，不能提前设为 user_confirmed；明确确认时在同一事务内完成 Knowledge、proposal、被接受来源和反馈关联。明确更新已有 Knowledge 时走指定 ID/revision 的维护操作，不借 NEW 自动覆盖。当前 `_confirm()` 带有 embedding/organization 副作用，新人工入口只调用拆出的数据库保存部分，向量补算在提交后执行。重复确认返回同一个结果，不能新增一份确认。只需补这些小边界，不改变旧 schema 的信任约束来迁就新入口。

### 4.4 API 与 UI

拟新增接口：

| API | 行为 |
| --- | --- |
| `POST /api/v2/answers` | question、可选 thread/context、Idempotency-Key；返回 run ID、互斥回答状态、答案/澄清、引用与缺少条件。先使用有超时上限的同步回答，不建新异步问答调度器。 |
| `GET /api/v2/answers/{id}` | 展示历史回答和当时证据快照。 |
| `POST /api/v2/answers/{id}/feedback` | 保存本次纠正，区分 reply_only 与经验提案；写入独立 raw evidence。 |
| `POST /api/v2/feedback/{id}/confirm` | 明确确认完整经验文本/适用条件；幂等，并检查预期修订版本。 |
| `POST /api/v2/feedback/{id}/retest` | 基于当前可用知识重新回答，创建新的 answer_run，不覆盖原记录。 |
| `PATCH /api/v2/answers/{id}/verdict` | 人工标记复测通过/失败及原因；不能由生成模型自判通过。 |

`v2_answer_runs.request_key` 加数据库 UNIQUE；当前单公司共享鉴权下，客户端生成全局 UUID，不仅在线程内唯一。创建运行采用短事务 INSERT ON CONFLICT 后读取已有行，仅成功创建者调用模型；同 key 不同 question/context 返回 409，同 payload 返回原 run/进行中状态。feedback 确认也设置明确的幂等边界。同一请求键重试不重复建运行或经验；模型调用前记录 request，异常时标 service_error。耗时调用不占持有锁的事务。重启遗留请求标中断，显式重试创建关联旧运行的新请求键，不自动无上限补发。

改造 `templates/chat.html`：真实提问入口、答案草稿、引用、纠正、保存经验、复测和前后对比；保留 sessionStorage API key。学习资料仍进入 Inbox。新增独立答案 POST，不改变旧 `GET /api/v2/chat` 的兼容响应。

`templates/inbox.html` 增加轻量“待解决问题”筛选，读取 feedback；`summary()` 改为实际未解决数量。只有业务缺口进入这个队列，服务错误显示可重试状态。主导航增加“问答”，产品说明明确“内部工程师草稿”。

### 4.5 保留、简化和不做

保留：原始输入、明确确认、四种 trust、来源/历史、编辑/删除恢复、V1 接口、现有文本学习、有限重试、worker 和异步补 embedding。

简化/停用：自动 LLM Entity organization；新 Experience 路径中的拆句合句、强制俄语入库和逐事实比较；缺口不另外建设自动 taxonomy 或工单引擎。底层原文/经验可保留原语言，客户答案继续俄语。

`compare.py` 对 provider/格式故障的处理在 3.2 增加可辨别的错误结果：worker/UI 标为处理失败可重试，不能伪装成产品条件不明确。真正的语义不确定仍走澄清；不因为服务失败把可疑内容直接放行。

明确不做：PDF/PPT 上传、自动历史 Telegram 批量导入、V2 Telegram 对外自动回复、复杂多轮项目规划、库存报价、全局检索升级、重写已有文本 extractor。

### 4.6 Done Definition

- [ ] 3.0 的 20 条 Inbox UX 用例完成并记录；自动组织关闭后，确认成功且精确 Entity 关联仍在。
- [ ] Phase 3 固定 30 题通过第 3.1 节门槛；未确认/已删除/冲突/范围错误内容不会被作为答案依据。
- [ ] 至少 5 个真实纠正完成保存，分别通过原问题、同义问法、边界反例复测；复测不向模型注入参考答案。
- [ ] reply_only 不改变知识；经验确认可在模型/embedding 不可用时完成；重复点击不产生重复知识。
- [ ] 原答案、纠正原文、确认、Knowledge 修订、复测结果能连起来查看。
- [ ] 俄语生成失败与证据不足分开显示；至少一个有真实缺失字段的澄清能承接用户补充重新回答。
- [ ] 相关单测、完整 unittest、隔离 PostgreSQL 集成测试、UI 手工截图和 `git diff --check` 通过。

**完成后真实新增能力：** 工程师可在系统里处理一个问题，并使同类问题下一次更容易解决；系统能展示这个改善的证据。

### 4.7 手工验收与回退

1. 准备一条已确认知识、一条未确认知识、一个不同型号的相似条目。
2. 在问答页提问，检查答案只引用合格条目，并能打开当时来源。
3. 输入一个缺型号/条件的问题，补充信息后重问；不能把检索到的型号当成用户型号。
4. 将一个错误答案改为完整正确流程，先选“仅本次回复”，确认知识没有新增。
5. 再保存为经验，检查确认版本和来源；运行原问、改写和相邻型号反例，人工判结果。
6. 在测试环境模拟 LLM 超时、embedding 429、重复提交，验证不会变成伪澄清或重复保存。
7. 修改该 Knowledge 后打开旧答案，旧证据快照仍保持原样。

回退：关闭新增问答/feedback 入口，旧 Inbox/V1 仍可用；保留记录和迁移。3.1、3.2 可以分别停用，不能删除用户已确认的经验。

## 5. Phase 4：PDF/PPTX 主动学习完整技术单元

### 5.1 只解决什么，以及为什么现在做

只解决：上传核心 PDF 和 PPTX 后，后台主动检查整本文档，生成可审阅、可复用、有完整来源的技术单元，并让 Phase 3 问答直接复用。

此时已有真实问答与复测工具，可以判定提炼是否保留了关键条件。不能把 Phase 4 做成单纯上传/切块/展示，然后宣称已具备文档学习。

### 5.2 独立上线顺序

| 小版本 | 范围 | 独立验收/上线结果 |
| --- | --- | --- |
| 4.1 文件与结构 | PDF + PPTX 一起支持；版本、页/幻灯片/表格/图片定位；失败标记；单 worker 文档任务 | 工程师能上传、定位和检查资料，包括明确显示无法理解的图片页。尚未自动进入回答。 |
| 4.2 完整单元 | 文档专用提炼、proposal 预览、编辑确认、来源对照；与现有 Knowledge 共用保存检索 | 完整配置流程和条件规则成为可回答 Knowledge；提炼不按句子逐条确认。 |
| 4.3 全文覆盖与质量 | 跨章节明确引用、全局限制、处理清单、部分续跑、PDF/PPT 问答验收 | 全文每个章节有明确处理结果，核心问题通过复测；完成 Phase 4。 |

### 5.3 文件、版本与结构：从第一天建来源，不以后补

新增 `v2/documents.py` 负责上传/版本/结构读取和保存；新增 `v2/document_processing.py` 负责有限步骤的后台执行。现有 V1 `documents/document_chunks` 仅读，不写入新 V2 学习内容，防止未验证资料进入 V1 检索。旧文件可由 V2 版本记录通过 `legacy_document_id` 只读关联，不必复制文件或迁移旧表。

| 新对象/变化 | 最小字段/用途 |
| --- | --- |
| `v2_document_versions` | id、document_key（同一手册的版本组，不按文件名猜）、version_label、sha256、文件路径/类型/大小、title、applicability、source_authenticity、parser_version、status、legacy_document_id、时间。文件内容不可覆盖，同一个文件 hash 可复用物理文件。 |
| `v2_document_blocks` | version_id、block_key、parent_block_id、顺序、章节路径、页码/slide 编号、block_type、raw_evidence_id、layout/assets、content_hash、processing_state/reason。正文保存在关联 Raw Evidence，避免维护两份可编辑原文。 |
| `v2_document_jobs` | version_id、stage、block_id/检查点、幂等键、status、attempts、next_run_at、result/proposal IDs、error、时间。按版本+步骤+解析/提炼版本幂等，结果通过明确 ID 判断完成。 |
| `v2_knowledge` / proposal 文档字段 | `origin_document_version_id` 外键、`validation_status`（pending/validated/needs_revalidation）；非文档旧记录来源版本为空。每个文档版本生成独立单元，不跨版本共享同一个 Knowledge ID。 |
| 既有来源关系 | 每个 Knowledge 可关联多个 block 的 Raw Evidence；主步骤、前置条件、全局警告分别保留来源。 |

文件保存在受控的 `data/documents/v2/` 下，服务生成存储名，原文件名只作显示。上传大小、解压量/条目数、解析时间设上限；检查实际格式并拒绝路径穿越，不执行宏或主动内容。鉴权下载和结构接口不暴露任意本机路径。文件与数据库分别备份，并记录恢复步骤。

不要把 V2 version/block ID 写进现有 `v2_raw_evidence.document_id/document_chunk_id`，这两个 FK 仍指向 V1 表。新来源通过 `v2_document_blocks.raw_evidence_id` 的明确外键反查版本，相关查询建立索引。新文件下载根据 PDF/PPTX 返回正确 MIME；不能直接复用目前固定 PDF 响应的 V1 `document_file()`。

版本不能等到 Phase 5 才加：Phase 4 就要求绑定具体文档版本；新版本知识不会默认继承旧版本信任与适用范围。文档单元回答资格须同时满足 trust、`validation_status=validated`、来源版本与请求适用性；只有明确检查确认后设置 validated。文档出版版本与设备固件版本不是同一个字段，由文档 applicability 记录已知对应关系。多个版本可能影响结论而用户版本未知时先澄清。Phase 5 增加的是变化影响分析和复用，不是第一次支持版本隔离。

### 5.4 PDF 与 PPTX 的明确支持边界

先用少量真实样本验证轻量解析：PDF 可从 `pdfplumber` 起步，PPTX 从 `python-pptx` 起步；只安装最终选中的实现，锁定兼容项目 Python 3.10 的版本。PDF 的文本/表格结构提取适用于文本型材料，扫描图像不能假装提取成功；PPTX 的表格要保留合并单元格语义，不能只遍历显示文本。依据：[pdfplumber 官方说明](https://github.com/jsvine/pdfplumber/blob/stable/README.md)、[python-pptx 表格说明](https://python-pptx.readthedocs.io/en/latest/user/table.html)。

- PDF：保留页、目录/标题、段落顺序、表格表头/单位/脚注及图片定位。无法识别的结构标 `needs_review`，必要时让用户校正章节边界和表格内容，保留原始解析结果。
- PPTX：保留 slide 顺序、标题、文本框、讲者备注、表格、图片及其位置；组合形状按顺序/布局收集。图表、SmartArt、动画含义无法可靠恢复时明确标记，不生成产品结论。
- 原 PPTX 永久保留可下载；页面展示文本/表格/图片及 slide 定位，不承诺 python-pptx 能渲染整张幻灯片。完整视觉对照可使用关联的人工导出 PDF；不为了预览先部署 Office 服务。
- `.pptx` 是 Phase 4 的直接学习入口，不能要求用户先全部转 PDF。旧二进制 `.ppt` 明确提示先在 PowerPoint/LibreOffice 导出 PPTX；本阶段不实现服务器自动转换。
- 扫描 PDF、图片中的接线逻辑：保留原图/原页并显示未解析，允许人工填写经确认的说明。Phase 4 不承诺全自动 OCR/视觉理解；这类页仍必须出现在处理清单，不能算作已学会。
- 首份资料验证 parser 的能力后才锁定依赖。如果真实核心资料普遍超出轻量解析能力，停止在该具体问题上评估一个替代解析器，不同时引入几套管道。

PPTX block 最小输出契约：`slide_number`、`slide_title`、`block_key`、`block_type`、`order`、`bbox`、`raw_evidence_id`、`needs_review_reason`；slide 级记录 `speaker_notes`，table block 保存 `rows/columns/cells[{row,col,row_span,col_span,text}]`，image block 保存 `asset_ref` 和 `interpretation_status`（未解释不能标已理解）。4.1 用带备注、合并单元格和图片的固定 PPTX fixture 校验这些字段，4.2 的提炼测试验证备注、表头/合并关系和相邻 slide 上下文实际传入模型，而不是只保存在数据库中。

### 5.5 完整 Knowledge Unit 合约

扩展现有 `v2_knowledge`，不新建独立知识存储：`unit_kind` 支持 `fact / procedure / rule / experience`，增加可选 `details_json`。Phase 3 的 Experience 可继续用完整自然语言；旧 fact 无需全量回填。

| 字段 | 含义 |
| --- | --- |
| title、unit_kind | 一个可复用技术任务/规则，而不是一个句子。 |
| applicability | 明确型号、版本、地区、工作模式与其他条件；未知保留未知。 |
| details_json | 可选 prerequisites、ordered_steps、expected_result、exceptions/warnings；rule 可以保存明确的触发条件和结果，experience 可以保存现象、尝试、观察结果。只填原文提供的项。 |
| content | 面向阅读和检索的完整文字。结构化单元由 details 确定性渲染；编辑结构同步更新 content，不维护两个互相矛盾的权威版本。自由文本旧条目继续兼容。 |
| sources | 多个来源 block、具体摘录、支撑/上下文关系和文档版本；一个单元可引用跨页/跨章节证据。 |

proposal 增加对应的 unit/details 元数据，并用现有 pending/confirm/edit/reject 流程。UI 显示整项任务，工程师可以一次确认或编辑；高影响条件和未明确推导需要单独检查，不把全部内容机械按原子事实逐条提问。

`service.py` 的编辑、sources、history、序列化和 `reembed.py` 同步支持新字段。结构字段和适用范围变化必须增加 revision、进入历史并使 embedding 失效。对新单元改用明确长度/大小校验：超过上限显示未完成，不能沿用 `_clean(...12000)` 静默截断。超长任务保持完整逻辑，可引用有来源的子流程，不为长度硬切步骤。

### 5.6 文档学习算法：主动全面，但不做复杂 agent

新增 `v2/document_learning.py`，通过现有 `llm.extract_structured()` 调用文档专用 schema。继续单一 OpenRouter 默认学习模型，不建设模型路由。

1. 代码建立文档 block 清单和目录，登记全局适用范围/警告所在位置；人工修正无法可靠判断的结构。
2. 按完整小节或相邻幻灯片主题组成一个任务上下文，附文档范围及明确引用的前置小节。不能直接调用 `segment_bulk_text()` 把整书拆成编号句子。
3. 每个上下文一次提炼多个完整单元；模型输出可定位的证据引用、保留的条件和未理解内容。输出 token 不足/被截断时留下检查点并重做该上下文，不算完成。
4. 代码验证引用存在、摘录对应、数值/型号没有非法变化、结构字段齐全；不能以这些检查代替语义正确判断。归一化只作用于比较视图，原文和坐标不改写。
5. 明确相同的来源/内容做确定性去重；相似单元只给编辑者提示，不自动跨型号、版本合并。不对每个步骤调用 compare。确有候选冲突时，最多按当前主题做一次比较提案。
6. 合格提案写入现有 proposal 并展示来源；失败单元独立保留待重试状态，合格单元可以继续检查，不要求重传整本文件。
7. 后台继续处理后续章节，最后核对原始 block 清单：已提炼、仅保留证据且有原因、待人工、失败。`处理覆盖=100%` 只表示每段有去向，不能显示为“知识正确率100%”。

文档专用路径不经过 `learning.py:_postprocess_semantic_units()`、`_split_obvious_conjoined_unit()`、强制俄语 canonical 校验和单一连续 `source_excerpt` 合约。通过提取少量共有保存/确认函数复用事务，不复制整套状态机，也不修改旧文本提炼的默认含义。

可信晋升采用保守而可操作的方式：官方来源身份由人核实；模型提炼仍是待检查内容。本阶段对首批核心技术单元采用人工对照后确认，沿用 `user_confirmed` 并在 sources 记录官方来源。不要仅因 PDF 标题含品牌就自动创建 `official_source` Knowledge；以后是否降低简单条目的人工负担，以审核时间和错误数据决定。

### 5.7 Worker、API、UI

复用 `worker.py` 和 heartbeat，在主循环增加文档任务的固定分支，不建设通用 workflow engine。现有 Inbox job 表和恢复语义不改变。新任务每次只处理一页/一个小节/一个有限步骤并提交检查点，在步骤间优先处理 Inbox；为防文档饥饿设置小的固定交替配额。一次模型调用仍可能占到超时上限，但不能一口气占用 worker 处理整本手册。

429 按有限重试和 `next_run_at` 延后，持久化已完成部分；不在轮询循环无限重放。重新执行通过检查点和明确 result ID 幂等；长网络调用不持事务锁。文档任务恢复与 Inbox 恢复分开判断，不能用“之后出现了一个 assistant 消息”判定文档完成。

拟新增接口：

- `POST /api/v2/documents`：上传 PDF/PPTX，选择/创建 document_key，登记版本和来源；返回 version/job ID。
- `GET /api/v2/documents/versions/{id}`、`.../{id}/file`、`.../{id}/blocks`：状态、原文件、结构/定位和处理清单。
- `POST /api/v2/documents/versions/{id}/learn`：幂等开始或继续后台学习；上传默认排队，不必等用户逐章触发。
- `GET /api/v2/document-jobs/{id}`、`POST .../{id}/retry`：明确步骤状态与重试。
- proposal 编辑/确认继续走现有 V2 入口，增加完整单元 payload 和 revision 校验，不创建另一套 review UI。

`GET /api/v2/documents` 兼容原返回 envelope，区分只读 legacy 资产与新 V2 版本，不把两类 ID 混用。`templates/documents.html` 增加上传、版本、原文定位、进度、待检查内容；`templates/inbox.html` 复用完整单元确认组件；`templates/knowledge.html` 展示步骤、条件和多来源。

### 5.8 保留、简化和不做

保留：Phase 3 闭环、原始资料、source/history、明确确认、Entity 精确关联、单 worker 和数据库任务、现有 embedding 模型与补算机制。

简化/停用：文档不走旧 bulk 原子切分；不全书逐 fact compare；不强制所有知识先译成俄语；不在确认后整理实体层级。旧学习接口和旧数据不删除。

明确不做：自动视觉理解全部接线图、服务器 Office 转换、全量 Telegram 历史迁移、跨文档自动知识合并、自动大规模知识清理、增强 rerank、文档更新的精细增量重编译。问答本阶段优先使用已确认文档单元；自动回原文由 Phase 5 接通，缺少单元时明确说明并可人工打开来源。

### 5.9 Done Definition

- [ ] 至少一份真实核心 PDF 手册与一份真实 PPTX 培训资料完成从上传到 Knowledge 再到问答的全过程；不能只用人工粘贴文本演示。
- [ ] 两份资料所有章节/slide 有处理状态；扫描页、未知图表、失败段不能静默丢失或标为已理解。
- [ ] 门禁手册至少验证添加用户、添加指纹、权限、常开、反潜回、联动、排查这些实际存在的主题；缺失主题明确记为原文未提供，不生成补全知识。
- [ ] 工程师事先从资料中标出的关键前置条件、顺序、例外、数值、型号/版本边界全部保留；每个关键单元可以回到正确页或 slide。
- [ ] 追加至少 20 条人工编写的文档问题，包括跨章节条件、PPT 表格/备注、组合功能与未说明问题；可回答样本至少 80% 无需实质性技术改写，关键错误为 0。需要图像解释而尚未支持的问题必须诚实退出，不能混进可回答样本稀释标准。
- [ ] 同版本重复上传/重复学习不重复产生确认知识；中途停止 worker 后从检查点续跑，Inbox 不被整书任务长期占住。
- [ ] 人工修改单元后，history/details/content/embedding 状态一致；没有被默认长度限制截去步骤。
- [ ] Phase 3 回归与真实问题集无新增关键错误；新表/来源关系/恢复行为通过隔离 PostgreSQL 集成测试。

**完成后真实新增能力：** 工程师上传常用手册或培训 PPTX，系统主动整理整份资料中的可复用技术单元；确认后即可用于真实问题和复测。

### 5.10 手工验收与回退

1. 在上传前，专家独立写出至少 10 个关键条件/例外和一组客户问题，避免只用模型自己生成的问题验收。
2. 上传 PDF 和 PPTX，逐页/slide 抽查标题、表格、脚注、备注和图片位置；检查纯图片页被明确标记。
3. 检查一个完整“添加用户并获得权限”的实际流程：前置步骤、顺序、范围、结果和来源必须一起出现。
4. 检查“常开与反潜回是否同时生效”之类组合问题；若手册未说明，不能由两个独立能力推断答案。
5. 确认一个单元，在问答页用原问和改写提问；纠正其中一项，再复测，验证仍使用 Phase 3 同一闭环。
6. 重传文件并中断一次后台任务，确认没有重复知识且可以继续；上传新版时，旧版知识不会无条件用于新版。

回退：停用文档学习/上传入口及 worker 文档分支，Phase 3 和旧 Inbox 可继续用。若某批提炼有质量问题，按 document version 暂停相关单元的回答资格；保留文件、提案、来源和历史，不批量物理删除。

## 6. Phase 5：补足原文回读与文档变化后的可靠复用

### 6.1 只解决什么，以及为什么现在做

只解决：已有 Knowledge 无法覆盖问题时能读取合格原文；原文更新后不错误复用旧知识；真实失败能驱动少量有证据的修复。

此时系统已经有回答运行记录、文档结构和来源依赖，才具备定位“缺知识、漏召回、错理解、错版本”的条件。Phase 5 不作为复杂 retrieval 的默认建设阶段。

### 6.2 独立上线顺序

| 小版本 | 范围 | 独立验收/上线结果 |
| --- | --- | --- |
| 5.1 原文回读 | 在既有 answer service 增加有限的文档证据读取和引用 | 知识未覆盖但原文明确写出的内容可以回答；未知/无法解析的图示仍退出。 |
| 5.2 更新与重验 | 版本差异、派生知识影响清单、重新学习/确认及版本选用 | 上传新版不会污染旧项目，也不会继续拿过期知识回答新版问题。 |
| 5.3 日常使用与针对性修复 | 高频失败列表、纠正复测、必要时一个检索改进 | 记录同类错误实际下降；没有召回问题证据时，不增加检索组件也可以完成阶段。 |

### 6.3 原文回读：一个答案流程，不造第二个机器人

在 `v2/retrieval.py` 增加 `retrieve_document_evidence()`，在 `v2/answering.py` 组合证据。沿用 Phase 4 block/source/version，不重新建设 document_chunks 副本。

先取可信且适用的 Knowledge；以下情形读取对应原文：

- 没有合格单元，或命中单元明确标注有未覆盖细节；
- 问题明确要求核对原文、表格、具体界面或版本；
- 涉及组合功能、关键前置条件或高影响操作，需要检查关联原文；
- 初次回答发现依据不足，允许一次有界补充读取。

不只依赖模型自报置信度触发。提供“核对原文”按钮，便于工程师对任何答案进行检查。

原文路径同样有资格门槛：来源真实性已经确认，解析质量满足该类内容，版本/型号匹配，完整章节及其前置/全局限制可用。不能因为 `evidence_type=document` 就直接当 official，也不能把 Phase 4 未审阅的 LLM 摘要当作原文。

选择小节后读取其完整内容、必要的父标题、同表表头/脚注及已经记录的明确引用；初版只沿显式来源关联展开一次，设置上下文上限。装不下且不能完整表达时澄清或转人工，不静默切掉最后几步。不实现递归探索、动态多 agent 或通用 multi-hop。

模型调用预算：通常组好证据后一次生成；只有确实需要补读时允许再生成一次，总计最多两次答案生成，不无限“检索—反思”。使用同一个默认模型。

答案记录保留 `knowledge` 或 `document_block` 类型、版本、引用和当时快照。Knowledge 与新原文矛盾时显示冲突，不按“经验更重要”或“新文件更新”自动选真。若关键版本未知，先询问设备实际版本。

### 6.4 文档更新与最小影响分析

扩展 `v2/documents.py` 和 `v2/document_processing.py`，无需新的图谱服务：

1. 新文件新增 version，旧文件/blocks/Knowledge 保留。按 document_key 和人工确认的版本关系分组，不能从文件名猜新旧。
2. 先按章节与规范化内容 hash 找新增、变化、删除和无法匹配的范围；页码变化不等于内容变化，原始 bytes/hash 仍保留。
3. 通过 Knowledge → Raw Evidence → block/version 关联找影响项。全局型号范围、警告或前置章节变化会影响其所有依赖单元。
4. 依赖不明确时，对该新版本保守地重新验证整本文档单元，不猜测哪些安全。
5. 新版本适用的单元在重验前不得继承旧版本回答资格；旧版单元仍可用于明确的旧项目。不得只修改一个全局 active 标记导致所有旧项目失去可用知识。
6. 未变内容也要检查依赖和范围；确认可复用时，为新版本建立独立 Knowledge 行、明确来源和适用范围，再标 validated。保留 Phase 4 的每版本独立单元机制，不为了去重引入复杂继承。
7. 经人工核对后启用新版本知识。删除说明不等于功能不支持；文档之间矛盾保留待确认。

最小新增字段：版本比较基准和受影响单元/blocks 清单；复用 Phase 4 的 `origin_document_version_id/validation_status`、jobs 和 proposal 保存重处理过程。新版本生成的单元从 pending 起步；同一版本重新解析/发现问题时，相关单元进入 needs_revalidation。旧版本单元不会仅因出现新版就被全局停用；只有其自身证据失效或被明确纠正时才撤销资格。本计划不增加跨版本共享 Knowledge 的关联/继承表。

V2 answer 资格检查必须同时考虑 Knowledge revision、来源状态和请求版本；历史答案证据快照不随来源更新改变。Phase 3 feedback/纠正继续有效，但旧版本的现场成功经验不能自动迁移为新版成功经验。

### 6.5 日常失败回流与更高级 retrieval 的门槛

扩展 `v2/feedback.py`、Inbox 待处理筛选和 `evaluate_v2.py`，按以下原因展示真实失败，不建设新的运营平台：

| 原因 | 默认动作 |
| --- | --- |
| 缺少资料/原文没有答案 | 补资料或形成待确认 Experience。 |
| 原文有，Knowledge 漏了条件 | 修改/重提炼具体单元，回跑受影响问题。 |
| 合格证据存在，但没召回 | 检查型号/别名、范围过滤和排序，保留诊断样本。 |
| 证据正确，答案错误 | 修改生成约束/完整流程呈现，复测；不重复新增同样知识。 |
| 版本错误/来源冲突 | 修适用性和版本资格，不提高相似度阈值掩盖问题。 |
| 服务失败 | 延后重试/提示服务状态，不制造产品知识提问。 |

更高级 retrieval 的进入条件：至少 10 个真实问题能证明“正确、完整、合格的证据已经在库，但现有检索未选中”，且不是解析/提炼/版本错误。记录预期来源 ID、当前候选和过滤原因。

每次只尝试一个最小变化：人工别名 → 更明确的章节/型号词法检索 → 必要时复用现有 `rerank.py`。顺序不是强制装完的清单；选择对应实际失败的那一项即可。先在固定样本上比较，只有预期证据召回提升、答案关键错误不增加、延迟/请求成本在试用可接受范围内才保留。无效果即关闭，不保留“以后可能有用”的分支。

只有当前 exact scan 在实际数据量下产生可重复的延迟问题，才讨论 PostgreSQL 内索引/SQL 优化；不换独立 vector DB，不因为 Knowledge 数量增加就默认加高级 pipeline。

### 6.6 API/UI、保留和不做

复用 `POST /api/v2/answers`，增加 `check_sources` 和明确的版本上下文；不创建第二个回答 API。Chat 显示原文页/slide、Knowledge 与文档版本以及“核对原文”。

Documents 增加 `GET /api/v2/documents/versions/{id}/impact`、`POST .../{id}/revalidate` 及版本对比/待重验列表；确认继续复用 proposal。Inbox 显示尚未通过复测的真实失败，可按次数和最近使用排序；不引入语义聚类服务。

保留：Phase 3/4 的单模型、信任门槛、明确确认、结构来源、完整单元、worker 和既有 UI。停用/删除：仅删除这一阶段 A/B 检验未带来收益的新实验分支；不清理历史知识或原文。

明确不做：自动全局合并/重写知识、所有产品线覆盖、实时库存报价、自动对客户承诺、对外 Telegram 切换、模型微调、多模型选择、默认 OCR/视觉平台。若真实核心问题反复卡在图像解析，可作为单独下一任务评估一个工具；不塞进 Phase 5 验收范围。

### 6.7 Done Definition

- [ ] 至少 5 个 Knowledge 未覆盖、合格原文明确有答案的真实问题，能够读取正确章节回答并引用；不得使用未确认摘要冒充原文。
- [ ] 至少 5 个原文未说明/图像无法解释/版本未知的问题，正确澄清或退出；组合功能不能靠单项支持推导。
- [ ] 用一份手册的两个版本完成真实对比：包含一项步骤变更、一项条件/全局限制变更，以及一个删除或无法匹配段落；全部受影响单元被识别或保守列入重验。
- [ ] 新版未重验时不继承旧知识；旧版项目仍能按旧证据回答；同名/同型号也不能把版本混用。
- [ ] 重新验证后原问题、改写、版本反例通过；历史答案能回看原版本。
- [ ] 至少连续 5 个实际工作日试用，记录不少于 20 个真实问题；至少 5 个失败完成纠正与人工复测，展示改善，不只展示总知识数。
- [ ] 累积固定集无新增关键错误；若引入 retrieval 变化，附前后同集比较、延迟/调用数和关闭办法。没有充分证据时“不加高级 retrieval”符合 Done。
- [ ] 原文读取不超过有界上下文/调用预算，版本任务失败可续跑，相关 PostgreSQL/单元/UI 回归通过。

**完成后真实新增能力：** 工程师既能复用已学知识，也能核对尚未提炼的原文；资料更新后系统知道哪些经验需要重新确认，而不是不断累积过期答案。

### 6.8 手工验收与回退

1. 选一个原文有答案但尚无 Knowledge 的问题，确认系统读取了对应完整章节，而不是给固定拒答。
2. 选一个组合功能但原文未说明的问题，确认不会将两个单项支持拼成组合保证。
3. 上传新版手册，检查影响列表；分别以旧版本、新版本和未知版本提问。
4. 在重验之前与之后各提问一次，检查新旧资格变化、来源版本及旧答案快照。
5. 对一个漏召回问题先查 trace；只有证据存在才做检索修改，并使用同一组问题比较前后结果。
6. 连续使用一周，检查真实失败有没有关闭、纠正是否能复用，以及新增审核工作是否少于节省的查资料/改答案时间。

回退：可以单独关闭原文回读，回到已确认 Knowledge 问答；版本重验任务可暂停，但不能恢复已经确认不适用于新版的旧知识资格。检索增强有独立关闭点，不依赖删除索引/数据来回退。

## 7. learning.py 的修改清单与时机

| 现有机制 | Phase 3 | Phase 4 | Phase 5 |
| --- | --- | --- | --- |
| `_run_local_organization_review()` | 停自动模型组织，保留精确关联 | 不恢复自动树整理 | 只有真实关联失败另案讨论 |
| `_model_facts()` 与俄语校验 | 旧输入保留；人工 Experience 确认绕开重新提炼 | 新文档 extractor 使用自己的单元 schema，保留原语言 | 输出答案俄语，不强制重译历史知识 |
| `_split_obvious_conjoined_unit()` / `_consolidate_related_units()` | 不扩大规则；完整 Experience 不经过 | procedure/rule 文档单元不经过 | 不回头批量重拆旧数据 |
| `segment_bulk_text()` | 仅保留现有手工长文本用途 | 文件按结构 block/小节学习，不复用文本猜段 | 原文读取沿用结构，不再切固定小块 |
| `_plan_fact()` 逐条比较 | 旧学习路径保持；人工确认经验显式处理冲突 | 文档只做确定性重复检查和必要的主题比较 | 不建立全库自动相似合并 |
| 单连续 excerpt、claims/coverage | 旧合约兼容，不降低已运行链路的引用检查 | 新文档单元多来源引用；coverage 基于解析清单，有失败状态 | 来源关联用于回读和影响分析 |
| 12,000 字符限制/静默清理 | 明确人工 Experience 限制，超出报错 | 新结构路径禁止静默截断，按完整子流程处理超长内容 | 检索不能再次丢步骤尾部 |
| `_confirm()` 与历史/向量 | 复用、加 revision/显式确认目标，经验可纯数据库保存 | 新字段进入确认、历史、embedding 失效和重算 | 修改后复测记录引用准确 revision |
| compare 故障变 UNCLEAR | 区分技术失败与知识歧义 | 文档保留失败任务，不向用户伪造澄清 | 同样纳入故障统计 |

重点：抽取小的共享保存/确认能力，不重写 2,000 多行学习流程；允许旧文本与新文档提炼入口暂时并存，共用 Knowledge、source、trust 和维护层。

## 8. 实施顺序、上线纪律与最大风险

推荐严格顺序：`3.0 → 3.1 → 3.2 → 4.1 → 4.2 → 4.3 → 5.1 → 5.2 → 5.3`。不要同时开发完整文档管道和全新问答流程。每个箭头前都保存演示结果和验收记录；Phase 4.1 的 parser 样本验证可以提前小规模调查，但不阻塞 Phase 3 闭环。

每个小版本：先 additive migration（如需）→ 兼容后端 → worker（如需）→ UI/入口开放 → 手工验收。记录 feature 开关、迁移号、备份位置、恢复步骤和 screenshots 到 `OPERATIONS.md`/交付说明。新增配置集中在阶段实际需要时加入，不一次性设计几十个选项。

实施阶段的基础检查：

```bash
.venv/bin/python -m unittest discover -p 'test_*.py'
git diff --check
git status --short
```

新增测试沿用仓库 `test_*.py` 风格，建议按业务边界建立 `test_v2_answering.py`、`test_v2_feedback.py`、`test_v2_documents.py`、`test_v2_document_learning.py`、`test_v2_document_versions.py`。复用/扩充 `test_v2_postgres.py` 做迁移、确认、来源、job 恢复集成测试；V2 真实模型 runner 的命令和输出位置在 Phase 3.0 实现时写明，本文不列不存在的可执行命令。

**风险最大的是 Phase 4.2–4.3：把原文变成可直接复用的技术单元。** 引用、JSON 和全部章节有状态，只能说明结构可追溯，不能证明条件与操作逻辑被正确理解。一次遗漏会被重复回答放大。控制方式是先用一份 PDF + 一份 PPTX 验证、专家先写关键条件清单、独立真实问题复测、保留原文、禁止自动扩大适用范围和跨版本合并；质量未达标则停在待确认提案，不扩量。

最高的数据维护风险是 Phase 5.2：全局限制变化却没有使派生知识重新验证。控制方式是 Phase 4 就记录版本和全部来源依赖；不确定时按新版本整本重验，先保守正确，再谈增量优化。

项目推进的最终判断不是表数、代码量、知识条数或模型大小，而是：工程师是否愿意每天打开，真实问题是否更容易解决，一次纠正是否减少了下一次错误，以及维护负担是否仍能由一个人承担。

## 9. 实施状态记录（开发过程中追加，只记事实）

- 2026-09-06 Step 0 inventory：main 到 817e4c2，迁移 001–021（最新 021_v2_answer_runs.sql），Phase 3.0/3.1 已实现（retrieve_for_answer、v2/answering.py、v2_answer_runs、POST/GET /api/v2/answers、Chat 问答页、evaluate_v2_answers.py），Phase 3.2/4/5 未开始。`v2_answer_feedback`、`v2/feedback.py`、`v2/documents.py`、`v2_document_versions` 等均不存在。
- 2026-09-06 Step 1 hardening：answer evidence gate（accepted supports 来源对应的 raw evidence 必须 `evidence_status='active'`，superseded/redacted 不得支撑新答案）已在 e271075 落地并 push；本轮只补回归测试腐蚀修复（test_production_hardening 去掉过时的 Chat 导航排除断言，导航契约以 test_v2_ui 为准）。
- 2026-09-06 Phase 3.1 evaluation（`data/v2_eval_phase31_step1.json`，full；`..._retrieval.json`，retrieval-only；gitignored，仅本机验收用）：30/30 `pending_expert_mapping`，模型 `nvidia/nemotron-3-ultra-550b-a55b:free`，prompt `v2-answer-2`，28 次 LLM 调用；status_match 4/30（命中的 4 条全是预期 unsupported 的无依据类），mechanical critical_flags 全部为 0。**这不表示“答案全部正确”，只表示确定性引用检查无触发；4/30 也不表示系统损坏——sidecar 尚无专家知识映射，fail-closed 到 unsupported 是正确行为。** 真实覆盖问题以前一轮 5 条 supplementary 真实问答验证为准（answered/cited），`human_verdict` 全为 null，待领域人工统一验收。
- 2026-09-06 Phase 3.2 完成（迁移 022，两库已应用）：`v2_answer_feedback`、`v2/feedback.py`、Knowledge `unit_kind/applicability/revision`、proposal 同名字段（`thread_id` 可空）、run 的 `retest_of/feedback_id/verdict` 列、history `confirm` 动作；5 个 API + 1 个缺口列表/关闭接口；Chat 纠正/确认/复测/对比/判定 UI；Inbox 未解缺口筛选；`summary()` 的缺口计数改为真实值。回归 383 passed（含 8 个 pg 集成）。生产验收（`data/phase32_acceptance.json`，gitignored）：C1 reply_only 不污染；C2 新经验 K211（r2→r3 更新细化）可被复测复用（R177 answered，引用 K211/K150/K151）；重复提交/重复确认幂等；C3 范围经验 K212 使改写与反例正确限定 TandemVu 机型；C4 K143 适用范围更新 r1→r2 后快照带 revision 2 仍 grounded；C5 缺口分类 + 现场成功记录；相邻型号反例正确 unsupported；1 次 `service_error` 被正确隔离未判分。**真实发现**： decline 型经验对“是否”问句式敏感（R174 拒答，记 verdict fail + generation_failure 缺口 F8，待 5.3 处理）；草稿偶带俄语脚手架碎片（`Если речь о K156/K157`），属生成质量问题，非关键错误。
- 2026-09-06 Phase 4.1 完成（迁移 023，两库已应用；`pdfplumber==0.11.4`、`python-pptx==1.0.2` pin 入 requirements）：`v2/documents.py`（版本/结构块/解析任务）、`v2/document_processing.py`（单 worker、inbox 优先交错）、上传/版本/原文/结构/任务 API、Documents 上传与结构页。回归全绿；生产 E2E（Playwright 真实上传 PDF+PPTX fixture → worker 解析 → 结构可见，0 页面错误）：E2E-MANUAL 6 块（1 待人工 image_only_page）、E2E-TRAINING 7 块（1 待人工 empty_slide），图片资源落盘，解析器版本已记录。**文档单元尚未进入回答**（4.1 无提炼入口，block 均为 pending/needs_review）。**明确人工门**：仓库无真实业务手册/PPTX，4.3 需用户上传至少 1 份真实 PDF + 1 份真实 PPTX 做端到端验收；不得伪造业务资料。
- 2026-09-06 Phase 4.2 完成（迁移 024，两库已应用）：`v2/document_learning.py`（小节上下文、一次提炼多完整单元、引用/标识符/结构代码校验、整项提案、明确确认进 validated Knowledge）、learn 任务与 worker learn 分支（inbox→parse→learn 优先级，无 LLM 时不领 learn 任务）、提炼/提案/确认 API、回答资格增加文档单元 `validated` 门槛、Knowledge/Documents 两页展示步骤与来源。回归全绿；生产验收（`data/phase42_acceptance.json`，gitignored，提炼模型为可用的 Nemotron，见下）：E2E fixture 提炼出 4 提案（1 个无 trigger 的 rule 被校验驳回、纯标题上下文 0 单元无幻觉），确认 procedure（K213）与表格 fact（K214）后真实问答引用命中；toy 行验收后已停用（历史保留）。**阻塞性发现**：代码默认 `V2_LEARNING_MODEL=openai/gpt-oss-20b:free` 已在 OpenRouter 免费层下线（404），worker 提炼与旧文本学习同受影响；本轮未更换模型（换模型属人工决策），worker 任务如实失败并记录，可 `retry`；另修 `EXTRACT_MAX_TOKENS` 2000→4000（推理模型长 JSON 被截断）。**明确人工门**：选定可用的免费学习模型（或沿用问答模型单模型运行）后 retry 失败任务，并用真实手册/PPTX 走完 4.3 全文覆盖验收。
