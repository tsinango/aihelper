# aihelper：技术现状、已知问题与整改方案

更新日期：2026-09-05（主体历史快照：2026-09-02）
适用环境：`/opt/aihelper` 当前生产迁移版本

## 当前 V2 附录（2026-09-05）

本文件主体保留 2026-09-02 的 V1 历史快照和整改记录；以下是当前 V2 运行状态，
避免把历史数据快照误认为最新产品状态。

- V2 迁移 `013`–`020` 已应用。`013`–`019` 提供 Inbox learning、bulk/durable job、
  worker heartbeat、轻量 Organization 和 Knowledge history；`020` 增加 Entity/Relation
  软删除所需的 `deactivated_at`。
- 生产由 `aihelper.service` 和 `aihelper-inbox-worker.service` 两个 systemd 进程组成，
  工作目录为 `/opt/aihelper`，环境文件为 `/etc/aihelper.env`。`/ready` 检查 PostgreSQL、
  schema 和 Inbox worker heartbeat。
- V2 Learning 使用 OpenRouter 的 `openai/gpt-oss-20b:free` Structured Outputs。
  正常路径一次调用，最多一次 repair；失败时保留 Raw Evidence，不将原文 fallback 成
  Knowledge。V1 客户问答仍使用本文后述的 Nemotron 生成链路。
- V2 Knowledge 保持 Russian canonical content、trust 和 provenance。人工维护不调用
  LLM；编辑内容会清空旧 embedding，删除/恢复/移动通过 stable Knowledge ID 和 history
  保留审计关系。
- Entity Tree 是局部、轻量组织层。只有整个 active subtree 没有 active 或 deleted
  Knowledge 引用时，用户才可以手动 prune；操作为软删除并在数据库事务内重新校验，
  不做自动 gardening 或全局重建。
- 当前完整回归结果：pytest `250 passed, 8 skipped`；unittest `258 OK, 8 skipped`。

日常运维和迁移命令以 [`OPERATIONS.md`](OPERATIONS.md) 为准。

## 1. 文档目的

本文记录项目截至 2026-09-02 的技术状态、迁移过程中形成的限制、已经通过生产请求确认的问题，以及建议的修复顺序。已实施的 OpenRouter/Nemotron 替换、审核分组改造和本机 Qwen 影子评测以本文的“当前状态”为准；它不替代 `OPERATIONS.md` 中的日常运维命令。

本次结论来自代码、迁移文件、D1 导出、生产 PostgreSQL 汇总数据和请求 `28`、`29`、`30` 的 retrieval trace。整改已应用增量迁移 `010`–`012`，并完成 602 条 case memory 的无模型重建。

## 2. 当前架构与数据流

服务是 Python FastAPI 应用，PostgreSQL 保存产品、文档、Telegram support case、审核数据和在线问答 trace。OpenRouter 是唯一模型入口：Nemotron 3 Ultra 负责生成，Nemotron 3 Embed 1B 负责 2048 维向量，Llama Nemotron Rerank VL 1B V2 负责候选重排。

在线问题经过以下链路：

1. Telegram webhook 或 `/api/v1/query` 接收问题。
2. `helpers.py` 提取型号、扩展别名并进行确定性路由。
3. 系统并行检索结构化产品事实、已发布 Verified Knowledge、官方文档和已批准学习样例；未审核 Telegram case memory 只用于离线召回/审核辅助。
4. scope 规则排除明确冲突的型号证据。
5. Nemotron 只能根据传入证据生成俄语答案；无法直接支持时必须 fail closed。
6. 问题、答案和 retrieval trace 写入 `questions` 表。

Telegram 原始聊天不会被在线查询直接读取。其链路是：

`Telegram 原始导出 -> support_cases -> support_case_analysis / knowledge intent -> case_knowledge_memory -> 人工审核候选 -> verified_knowledge -> 直接发布`

未审核 case memory 不能直接回答客户；只有人工审核并发布的 `verified_knowledge` 才能进入权威客户回答。

当前实现状态：服务已移除本机 embedding runtime、LongCat 和其它旧 LLM/provider 客户端；token 从 `/opt/aihelper/openrouter` 读取，生产环境开启 OpenRouter rerank。数据库迁移 `008`–`012` 已应用；五类 live embedding 列统一为 `vector(2048)`，并新增消息关系、知识证据、异步 embedding 状态和本机模型评测存储。当前 pgvector 安装无法为 2048 维列建立向量索引，因此在线检索暂时使用精确扫描，文本检索索引仍保留。旧 BGE/旧 provider 生成的主题和 pilot 文件已从活动产物中移除；已应用迁移文件中的历史算法名仅用于审计，不是运行时依赖。

审核分组默认使用不依赖模型的确定性规则（型号/范围/路由/词项），保留完整问题、事实、冲突和来源候选；V1.1/knowledge_key 与 OpenRouter 语义分组只是可选比较视图。批准组会直接发布，embedding 状态异步保持 `pending`，不阻塞人工审批。旧的 57 个分组是迁移前算法提案，不会被当作自动合并结果。

审核 payload 现在会在返回页面前合并完整候选字段；旧版本只保存部分人工覆盖字段时，也会从候选列补齐 claims、步骤、条件、例外和警告，避免再次保存时把未显示字段写成空值。

## 3. 生产数据快照

截至更新日期：

| 数据层 | 当前数量 | 说明 |
| --- | ---: | --- |
| `support_cases` | 602 | 从 Telegram 导入的派生 support case |
| `case_knowledge_memory` | 602 | 每个 Telegram case 一条 recall/evidence 记录 |
| 可回答 case memory | 2 | 仅已发布 Verified Knowledge 关联记录 |
| `verified_knowledge` | 2 | 已发布并允许生产回答 |
| `message_relations` | 1,344 | Telegram 原生/推断关系；人工关系优先级更高 |
| `knowledge_evidence` | 20 | 已发布知识的 Telegram 消息 provenance |
| 官方文档 | 3 | 文档元数据存在 |
| 文档 chunk | 0 | 当前库没有可检索 chunk，需要后续导入/OCR 流程 |
| 审核候选 | 659 | 包含原始候选和旧分组产生的合成候选 |
| 审核分组 | 57 | 全部为旧算法自动生成的 open 提案 |

这说明当前主要问题不是“case memory 没有导入或没有向量”，而是消息加工、知识分类、候选排序和证据选择的质量。

### 3.1 V1.1 意图批处理 checkpoint 状态

V1.1 的输入文件实际包含 591 个问题。当前 OpenRouter-only checkpoint 有 391 行，约覆盖输入的 66.2%；旧 provider 结果已从活动 checkpoint 和本地 V1.1 产物中移除。失败文件有 423 条失败尝试记录，这不是 423 个唯一 case，每条失败记录对应一次最多 3 次的批内格式修复重试，跨多次续跑同一 case 可能出现多条记录。

本轮使用 8 个 worker、每分钟最多启动 20 个请求。2026-09-02 04:56:40 UTC，OpenRouter 返回 429，批处理按设计停止并写入 `data/knowledge_intent_rate_limit_v1_1.json`；没有删除或覆盖已有 checkpoint。当前实现不区分 429 是每日额度还是端点速率额度，但两种情况都安全地按同一 checkpoint 续跑。下一次运行会跳过已经拥有当前 provider/model 且字段完整的结果，只处理未完成或失败的 case。

失败的主要原因是 Nemotron 返回空内容、非法 JSON 或不是严格的 13 个 V1.1 字段。每个 case 已有有限重试，最终失败会写入失败文件，不会阻塞其它 case；后续续跑可以再次尝试这些 case。批处理只生成意图、证据和 `knowledge_key`，不改变审核状态，也不会直接发布 Telegram 知识。

### 3.2 OpenRouter 额度耗尽时的审核行为

审核页面仍可以打开 case、查看完整线程、修改字段/消息角色/证据关系并保存修正。批准现在是纯 PostgreSQL 事务：先发布 `verified_knowledge`、写入 `knowledge_evidence` 并提交，再由 `reembed.py` 异步填充 OpenRouter Nemotron 向量。额度耗尽时 embedding 保持 `pending`，不会回滚批准，且文本检索仍可工作。

### 3.3 本机 Qwen 影子评测（2026-09-02）

为比较本机 Qwen3.5 量化模型与 OpenRouter 生产生成器，新增迁移
`012_local_model_evaluations.sql` 以及 `evaluate_local_qwen.py`。这不是第二条
生产问答链路：评测 runner 逐个启动本机 `llama serve`，只读取已发布且允许生产
回答的 Verified Knowledge，并只写专用评测表和 JSON artifact。生产 FastAPI、
Telegram webhook 和 `/api/v1/query` 仍只使用 OpenRouter。

此前数据库只有 2 条可用于生产回答的 Verified Knowledge，因此已完成的 smoke
run `qwen-20260902T092818Z-aa7f356f` 仅用于验证管道，不作为 4B 去留结论。现在
已补充 `data/golden_set.json` v2：135 条来自真实审核线程的固定样本，供 2B/4B
正式比较；对应 VK 尚未发布的样本使用 `golden_reference` 只读评测证据，不进入
生产问答：

| 模型 | 样本 | 生成速度中位数 | 结构化输出 | 知识适用性 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 2B Q4_K_M | 2 | 6.59 tokens/s | 2/2 | 1/2 |
| Qwen3.5 4B Q4_K_M | 2 | 3.13 tokens/s | 2/2 | 2/2 |

2B/4B 的生成时间分别为 `[10.98s, 26.49s]` 和 `[18.96s, 60.82s]`；这组
smoke 数据样本太小。正式 benchmark 使用 135 条样本，覆盖复杂条件、型号冲突、
必须澄清、unsupported 和拒答，并由人工填写 `answer_pass`。只有 4B 在复杂条件、型号混淆
和正确拒答等综合结果明显高出约 3–5 个百分点，才考虑保留；否则移除 4B 评测
依赖，保持简单架构。

本次 smoke 的 embedding 请求触发了 OpenRouter 429，runner 按设计使用已发布
知识快照完成生成比较；这不影响生产状态，也不改变人工审批。正式评测结果应
保存为 `data/local_qwen_golden_v2.json`，日常命令见 `OPERATIONS.md`。

## 4. 迁移阶段仍然存在的能力缺口

### 4.1 实时库存没有迁移

库存路由仍返回 `inventory_adapter_not_migrated`。系统不得从历史聊天推断库存或价格。需要独立的实时库存适配器、权限模型和 freshness 策略。

### 4.2 文档上传和 OCR 管道没有迁移

当前 `/api/v1/documents` 上传保持禁用，只复制了 3 份现有 PDF。新的手册不会自动进入文档库，除非补建上传、病毒检查、解析、分块、元数据抽取、embedding 和失败重试流程。

### 4.3 结构化产品知识覆盖接近空白

源 D1 的产品、属性、feature 和 verified fact 基本没有可用数据。大量请求只能依赖英文手册和未审核 Telegram 历史回答，准确率和可解释性受限。

### 4.4 Verified Knowledge 覆盖过小

当前只有 1 条已发布知识，来源是早期 pilot 包。审核、发布和版本控制能力已经存在，但内容覆盖远小于 602 个历史 case。

### 4.5 原始 Telegram 归档没有保存在当前服务器

`import_batches` 记录原始包名为 `telegram_ai_staging_real.zip`，但当前仓库、服务器本地文件系统和挂载文件系统中都没有该文件；只剩 `support_cases.sql` 等派生导出，也没有原始媒体目录。派生 SQL 不能作为原始聊天归档使用。

建议将原始归档放入受控对象存储，保存 SHA-256、导入批次、解析器版本和只读保留策略。聊天记录可能包含姓名、账号、序列号、现场信息和外部链接，不应长期依赖公共临时网盘保存。

## 5. 已由真实请求确认的问题

### P0：导入器丢弃原提问者的后续成功反馈（已修复）

`import_case_memory.message_texts()` 把第一条消息作为问题，之后只收集作者不是 `root_author` 的消息作为答案。这个规则会丢掉客户的补充信息、测试结果和“已解决”反馈。

Hik-Connect case `481` 是已确认实例：原线程最后由提问者反馈，把服务器改为 `dev.hik-connectru.com` 后恢复正常。但生成的 case memory 只保留工程师发送的说明链接，没有保存成功服务器地址。请求 `28`、`29` 虽然召回了 case `481`，模型实际没有得到可直接支持答案的文本，因此返回 unsupported。

风险不只限于这一条。所有由 root author 确认的结果都可能从 `answer_text`、`claims`、`searchable_text` 和 embedding 中消失。

对当前 `support_cases.sql` 的只读统计显示，602 个 case 中有 222 个包含 root author 的后续非空文本；这批记录都需要重新判断消息角色，而不能继续整体排除。

本轮已按消息元数据、回复关系、时间顺序和内容重建全部 602 条 case memory；原提问者的后续消息会保留，且会被标记为观察结果或确认结果。以下方案中的“建议”是后续抽样验证项：

1. 不再用“是否为 root author”判断问题或答案角色。
2. 根据 `message_id`、`reply_to_message_id`、时间顺序和内容给每条消息标记 `user_report`、`engineer_instruction`、`engineer_hypothesis`、`observed_result`、`confirmed_resolution`。
3. 保留完整线程用于检索，但只让经过角色和置信度约束的 claim 成为回答证据。
4. 把用户明确反馈“работает / помогло / решено”等结果存入 `observed_result`；只有条件、设备 scope 和操作均可追溯时才升级为 confirmed resolution。
5. 重建 602 条 case memory 的 searchable text 和 embedding，并生成差异审计报告。

### P0：正确官方文档被 Top-K 截断

请求 `30`（如何进入呼叫面板管理员模式）的 trace 已经检索到正确的 `Video Intercom Villa Door Station` 手册 chunk `41`、`42`、`43`。手册说明应长按主界面进入认证页，再输入激活密码或使用管理员人脸/卡。

但正确内容的排名低于前 5。`retrieve()` 最终只向后续链路返回前 5 个 chunk，前几名反而是 `DS-K1T320` 人脸终端等其他设备内容。模型遵守 scope 约束，没有把人脸终端步骤冒充为呼叫面板步骤，因此拒绝回答。

建议方案：

1. 文档召回与最终证据选择分离：先取 20–50 个候选，再 rerank 到 5–8 个证据。
2. 增加标题、产品类别、章节名和短语匹配特征；`door station`、`Authentication via Admin` 应优先于只有语义相似的 terminal 内容。
3. 不把无 `product_model` 元数据简单视为通用知识；从文档标题和文档级 metadata 提取 `product_family=door_station`。
4. 保证来源多样性，避免同一错误文档占满全部槽位。
5. 对关键步骤允许相邻 chunk 扩展，避免章节标题和具体步骤被切开。

### P0：fallback 文案覆盖了真正的澄清问题（已修复）

`query()` 先把 `answer` 初始化为固定失败文本。无证据或模型返回 unsupported 时，代码只增加 `clarifying_question`，没有清除或替换 `answer`。Telegram 发送端又优先取 `answer`，因此用户只看到：

`Не удалось подтвердить по доступным документам.`

实际生成的“请提供型号和固件版本”等澄清问题没有发给用户。

现已使用互斥状态机：

- `answered`：发送 `answer`；
- `needs_clarification`：只发送 `clarifying_question`；
- `unsupported`：发送失败原因和最小补充信息；
- `service_error`：发送服务错误文案。

API schema 应避免同一响应同时包含有效 `answer` 和 `clarifying_question`。Telegram 测试必须覆盖每个状态。

### P1：低阈值和固定配额让无关知识占位

Verified Knowledge 和 case memory 默认向量阈值为 `0.20`。候选通过阈值后，系统按 scope 和 RRF 排序，固定选择最多 3 条 verified knowledge 和 3 条 case memory。当前已发布知识只有 5 条，因此 password reset、firmware、rack ears、intercom capacity、autotracking 等条目会出现在完全无关的请求中。

后果包括：

- 挤占上下文窗口；
- `knowledge_source` 被错误显示成 `verified_knowledge`；
- 不相关 scope 型号污染整体 scope；
- 模型需要在噪声中做最终拒绝判断。

建议方案：

1. 用离线标注集按数据源分别校准阈值，不共用一个 `0.20`。
2. 增加绝对相关性门槛和 top-1/top-2 margin；没有足够相关候选时返回空集。
3. 先按 route/knowledge key/product family 过滤，再进行向量和 lexical 融合。
4. `knowledge_source` 只能根据 `final_evidence` 计算，不能根据预选候选计算。
5. 记录候选被选中和被 LLM 使用的两个阶段，避免 trace 含义混淆。

### P1：意图和知识 key 会被品牌词误导

路由规则中的英文 `connect` 会匹配品牌名 `Hik-Connect`，所以“服务器地址是什么”被标成 compatibility，而不是 parameter/configuration。case `481` 还被归类成 `firmware.check_latest`，尽管其中包含一个明确的 cloud server resolution。

建议把品牌实体识别与动作意图识别分开，并支持一个线程生成多个原子知识项。case `481` 至少应拆成：

- firmware freshness question；
- Hik-Connect server configuration；
- cloud registration troubleshooting；
- user-confirmed observed result。

### P1：以整个线程拼成一个 `answer_text` 会混合问题、假设和结论

当前 importer 把所有非 root author 文本用换行拼接为一个 answer，再同时复制到 `claims` 和 `procedure_steps`。工程师的反问、诊断问题、玩笑、互相冲突的建议和未验证假设可能全部被当成事实。

现已增加 `message_relations` 和 `knowledge_evidence`：每条消息保留角色、来源和成功/失败/上下文状态；`case_knowledge_memory` 仍只作 recall/candidate evidence。`needs_context` 内容只能生成澄清问题，不能作为独立事实回答。

### P1：缺少面向真实失败请求的回归评测

已建立 `data/golden_set.json` v2 固定评测集，共 135 条真实线程引用；包含直接回答、
条件限制、型号混淆、证据不足和多知识命中。当前标签为工程师初标，仍需领域专家
复核答案内容后再作最终模型结论。

请求 `28`、`29`、`30` 已加入固定骨架，并逐步收集：

- 应答问题；
- 必须澄清的问题；
- 型号冲突问题；
- 只能用 Verified Knowledge 回答的问题；
- 必须禁止未审核 case memory 直接回答的问题；
- 必须拒答的库存、时效性或无证据问题。

评测应分别衡量 recall@K、MRR、scope accuracy、citation precision、answer support、clarification usefulness 和拒答正确率。

### P2：可观测性不足

当前 trace 已保存候选、最终证据和文档 reranker 分数，但仍缺少统一的拒绝原因码、各阶段耗时和候选过滤原因。建议增加可聚合字段：

- `retrieval_empty`
- `below_relevance_threshold`
- `scope_conflict`
- `requires_context`
- `llm_unsupported`
- `llm_error`
- `answer_schema_error`

运营看板应按 route、语言、品牌、产品族和知识源统计回答率、澄清率与 unsupported 率。

## 6. 推荐实施顺序

### 阶段 A：修复用户可见行为（已完成基础修复）

1. 修复 Telegram 对 `needs_clarification` 的发送优先级（已完成）。
2. 让 `knowledge_source` 和 citations 只来自 `final_evidence`（已完成基础约束）。
3. 添加请求 `28`、`29`、`30` 的固定评测骨架（已由 v2 golden set 覆盖，待答案复核）。

这一阶段不改变知识内容，风险最低，可以首先上线。

### 阶段 B：修复 Telegram memory 构建

1. 新增 `message_relations` 与 `knowledge_evidence`（已完成）。
2. 修复 root author 后续消息丢失并重建 602 条 case memory（已完成）。
3. 重新分析 case `481` 和一组人工抽样线程。
4. 验证差异后重建全部 case memory 与 embedding。

重建必须保留现有人工 `verified`、`rejected` 状态和 reviewer override，不能被批处理覆盖。

### 阶段 C：改造召回与 rerank（基础能力已实施）

1. 建立离线 golden set（135 条 v2 样本已完成，待领域专家复核答案标签）。
2. 已接入 OpenRouter Nemotron embedding、lexical/title/family 特征和 reranker；embedding 不再阻塞审批，后续仍需用 golden set 校准阈值。
3. 校准每种来源的相关性门槛。
4. 增加相邻 chunk 和来源多样性策略。
5. 通过离线评测后灰度发布。

### 阶段 D：扩大可信知识覆盖

1. 优先审核高频、低时效风险、已有明确解决结果的 Telegram case。
2. 为服务器地址、密码流程等可能随区域、固件或时间变化的知识增加 freshness 和 scope。
3. 补建文档上传/OCR 管道和产品结构化数据。
4. 最后接入实时库存，保持与历史知识严格隔离。

## 7. 验收标准

完成整改后至少满足：

1. case `481` 的完整线程保留成功反馈；未审核前只能作为审核/召回证据，不能直接生成客户答案，发布后也必须限定适用型号、区域和条件。
2. “Как зайти в режим админ на вызывной панели?” 能召回 Villa Door Station 手册具体步骤；没有型号时给条件式答案或有用的型号澄清，而不是固定失败文案。
3. 不相关 Verified Knowledge 不再进入最终证据，`knowledge_source` 与 citations 和 `final_evidence` 一致。
4. Telegram 在 `needs_clarification` 状态下发送澄清问题。
5. 任何型号冲突、无证据产品事实、库存和时效性信息继续 fail closed。
6. 全量单元测试和 golden-set retrieval/answer eval 通过，并保存上线前后的指标对比。

## 8. 数据安全与交付注意事项

Telegram 聊天记录属于潜在敏感数据。任何归档或外发前应：

1. 明确接收方和保留期限；
2. 检查是否包含姓名、手机号、账号、序列号、现场地址、图片和外部私有链接；
3. 排除 `tgtoken`、`/etc/aihelper.env`、数据库 dump、API key 和运行日志；
4. 生成并记录压缩包 SHA-256；
5. 优先使用受控、可撤销访问的存储。若使用公共临时文件服务，应把链接视为可转发的公开访问凭据。

当前服务器缺少 `telegram_ai_staging_real.zip` 原始包。因此在重新取得原始文件之前，只能交付本技术文档，不能宣称已交付 Telegram 原始聊天记录。
