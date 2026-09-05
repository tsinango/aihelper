# aihelper V2 Future Design Notes

本文件只记录未来可能出现的结构化需求，不代表当前实现计划。所有复杂性必须由真实失败、golden eval 和可重复证据获得存在资格。

## CURRENTLY FORBIDDEN

Phase 1–6 当前禁止实现或接入 V2 主流程：

- Dimension Registry
- Knowledge Gardening engine / bulk AI gardening
- Ontology
- Knowledge Graph
- Complex taxonomy
- Topic abstraction
- Knowledge intent V1 / V1.1
- Review groups
- Knowledge keys
- Complex scope hierarchy
- Candidate review workflow
- Publish workflow
- Multi-provider framework
- Local-model benchmark framework（本地模型只能保留为已有影子评测资产）
- Agent swarm
- Multi-stage router

V1 中已经存在的相关表、脚本和 UI 只做 legacy 保留、审计和安全迁移，不重新接入 V2 主路径。

当前已经提供的 Knowledge 页面搜索、人工编辑、软删除、恢复、来源/历史查看和
空 Entity 分支 prune 属于 deterministic maintenance lite，不是 Knowledge Gardening
engine，也不会自动重写、合并、拆分或全局整理 Knowledge。

## 可能的后续演化

只有真实数据证明需要时，才讨论：

- emergent dimensions：从大量 Knowledge 和检索失败中发现稳定的产品维度。
- alias normalization：多个真实用户叫法导致 exact/full-text 检索失败时，增加可审计别名。
- dimension registry：支持维度的合并、拆分和 generalization proposal，但必须由人确认。
- knowledge gardening engine：只在真实证据证明 maintenance lite 无法处理重复、冲突、过期、无法检索的 Knowledge 后再讨论。
- knowledge graph：只有实体关系确实是问答失败原因时才考虑。
- coverage visualization：展示已确认知识覆盖和 unresolved gaps，而不是提前建设 dashboard。

## 讨论门槛

### Dimension Registry

同时满足以下条件后才允许讨论设计：

```text
knowledge > 200
并且
至少存在 10 个真实案例，证明同一技术维度被不同叫法拆散，
已经导致 retrieval 或 answer 失败
```

### Knowledge Gardening Engine

必须先有可重复的真实失败样本，至少证明：简单的原子 Knowledge、provenance、trust filter、exact/full-text/embedding retrieval 和主动提问无法解决同一类重复/冲突问题。没有这种证据时，保持独立 Knowledge 和明确 source。

### 其他结构

任何新 abstraction 都需要：

1. golden-v2 或新增真实失败样本上的 baseline。
2. 复杂机制带来的可量化收益。
3. 对 trust、provenance、型号隔离和拒答行为没有回归。
4. 明确删除/回滚路径和迁移影响。

原则：

> Complexity must earn its existence.

先让“学 → 用 → 补 → 整”产生真实数据，再决定是否需要“整”。
