# Agent Runtime Guide

## 1. Scope

This document explains the real agent runtime of `gl-hnsw` as implemented today.
It focuses on the offline indexing phase, because the project keeps agents out of the online query path by default.

The main goals of this document are:

- explain how the main orchestrator and subagents collaborate
- explain how context is passed between phases
- explain how memory is updated and persisted
- explain the concrete loop structure used during offline indexing
- explain how capabilities, tools, and skills are attached to agents
- show the implementation with bilingual Mermaid diagrams

---

## 2. Design Positioning

The system is `agent-centric` in the offline indexing stage, not in online serving.

- Offline stage:
  - agents profile documents
  - agents scout candidate links
  - agents judge relations
  - agents review edge utility and risk
  - agents curate memory after each anchor pass
- Online stage:
  - no agent call by default
  - retrieval uses HNSW, sparse supplement, stored graph, and local scoring only

This split is intentional:

- offline agent execution can be slower but richer
- online retrieval must remain deterministic, cheap, and observable

```mermaid
flowchart TD
    A["原始文档 / Raw Documents"] --> B["离线 agent 建模 / Offline Agent Indexing"]
    B --> C["文档简档 / DocBrief Store"]
    B --> D["逻辑图 / Logic Overlay Graph"]
    B --> E["记忆层 / Memory Stores"]
    A --> F["向量编码 / Embedding Build"]
    F --> G["HNSW 索引 / HNSW Index"]
    Q["在线查询 / Online Query"] --> H["本地检索链 / Local Retrieval Stack"]
    G --> H
    C --> H
    D --> H
    E --> H
    H --> I["最终结果 / Final Search Response"]
```

---

## 3. Runtime Topology

The actual runtime assembly is created in [bootstrap.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/services/bootstrap.py).

The important objects are:

- `AgentFactory`
- `LogicOrchestrator`
- `LogicDiscoveryService`
- `BuildPipeline`
- `MemoryCuratorService`
- `HybridRetrievalService`

The runtime relationship is:

```mermaid
flowchart TD
    A["应用装配 / build_app()"] --> B["Provider / Stub or OpenAI-Compatible"]
    A --> C["AgentFactory / Agent 工厂"]
    C --> D["LogicOrchestrator / 主编排器"]
    C --> E["DocProfilerAgent / 建档代理"]
    C --> F["CorpusScoutAgent / 侦察代理"]
    C --> G["RelationJudgeAgent / 裁决代理"]
    C --> H["EdgeReviewerAgent / 复核代理"]
    C --> I["MemoryCuratorAgent / 记忆整理代理"]
    C --> J["DeepAgent Runtime / DeepAgents 运行时(可选)"]
    D --> K["LogicDiscoveryService / 发现服务"]
    K --> L["BuildPipeline / 构建流水线"]
    A --> M["HybridRetrievalService / 在线检索服务"]
```

### Important implementation note

The repository prepares a `deepagents` runtime through `AgentFactory.try_create_deep_agent()`, but the main production path does not depend on the deepagents loop for every step.

The dominant execution path is:

- `LogicOrchestrator`
- typed subagent wrappers
- provider methods such as:
  - `profile_docs`
  - `propose_candidates`
  - `judge_relations_with_signals`
  - `review_relations_with_signals`
  - `curate_memory`

So the runtime is best described as:

- `deepagents-capable`
- but currently implemented as an explicit orchestrator pipeline with typed agent roles

---

## 4. Agent Roles

The configured subagents live in [agents.yaml](/Users/armstrong/gl-hnsw/configs/agents.yaml).

### 4.1 Main agent

The main agent is `LogicOrchestrator`, implemented in [orchestrator.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/agents/orchestrator.py).

Responsibilities:

- coordinate the offline indexing phases
- compute local signals and heuristics
- call subagents in a fixed sequence
- rank anchors and candidates
- convert judged results into final `LogicEdge` objects
- decide whether discovery should be attempted for an anchor

### 4.2 Subagents

The system currently defines these subagents:

- `DocProfilerAgent`
  - wraps `provider.profile_doc()` and `provider.profile_docs()`
- `CorpusScoutAgent`
  - wraps `provider.propose_candidates()`
- `RelationJudgeAgent`
  - wraps relation judgment with or without local signals
- `EdgeReviewerAgent`
  - wraps second-pass review with utility and risk signals
- `MemoryCuratorAgent`
  - wraps `provider.curate_memory()`

There is also a `QueryStrategyAgent` implementation, but it is not part of the default online serving path now.

```mermaid
flowchart LR
    O["主编排器 / LogicOrchestrator"] --> P["建档代理 / DocProfiler"]
    O --> S["侦察代理 / CorpusScout"]
    O --> J["裁决代理 / RelationJudge"]
    O --> R["复核代理 / EdgeReviewer"]
    O --> C["记忆整理代理 / MemoryCurator"]
    P --> X["DocBrief / 文档简档"]
    S --> Y["CandidateProposal / 候选提议"]
    J --> Z["JudgeResult / 裁决结果"]
    R --> U["Reviewed JudgeResult / 复核后结果"]
    C --> V["Memory Payload / 记忆更新载荷"]
```

---

## 5. Context Model

The main context objects are defined in [models.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/core/models.py).

### 5.1 Primary payload types

- `DocRecord`
  - normalized full document
- `DocBrief`
  - condensed agent-facing representation
- `CandidateProposal`
  - scout output
- `JudgeSignals`
  - local grounded evidence bundle
- `JudgeResult`
  - model-side verdict and utility signal
- `LogicEdge`
  - persisted graph edge
- `AnchorMemory`
  - per-anchor memory
- `GlobalSemanticMemory`
  - corpus-level reusable memory

### 5.2 Why two context layers exist

The system deliberately separates:

- `semantic context`
  - summary, claims, entities, relation hints
- `grounded local signals`
  - dense score, sparse score, mention score, direction score, risk flags, utility score

This separation is what keeps the system agent-centric but not prompt-only.

The agent does not operate from raw text intuition alone.
It also receives structured local evidence computed by the orchestrator.

```mermaid
flowchart TD
    A["DocRecord / 全量文档"] --> B["DocBrief / 语义压缩上下文"]
    A --> C["Local Signals / 本地计算信号"]
    B --> D["Relation Judge / 关系裁决"]
    C --> D
    D --> E["JudgeResult / 初判结果"]
    C --> F["Edge Reviewer / 边复核"]
    E --> F
    F --> G["LogicEdge / 最终边"]
```

---

## 6. Offline Agent Loop

The true offline loop is split between:

- [pipeline.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/services/pipeline.py)
- [discovery.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/services/discovery.py)
- [orchestrator.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/agents/orchestrator.py)

### 6.1 Pipeline stages

`BuildPipeline` runs the offline path in this order:

1. `build_embeddings`
2. `build_hnsw`
3. `profile_docs`
4. `discover_edges`
5. `revalidate_edges`

Only stages `3` and `4` are agent-heavy.

### 6.2 Per-anchor discovery loop

For each selected anchor:

1. load anchor `DocBrief`
2. scout candidate documents
3. compute local metrics and signal bundles
4. judge candidate relations
5. review judged candidates
6. build final accepted edges
7. write graph edges
8. update anchor memory and semantic memory

```mermaid
sequenceDiagram
    participant BP as 构建流水线 / BuildPipeline
    participant DS as 发现服务 / LogicDiscoveryService
    participant O as 主编排器 / LogicOrchestrator
    participant S as 侦察代理 / CorpusScout
    participant J as 裁决代理 / RelationJudge
    participant R as 复核代理 / EdgeReviewer
    participant C as 记忆整理代理 / MemoryCurator
    participant GS as 图存储 / GraphStore
    participant MS as 记忆存储 / Memory Stores

    BP->>DS: discover_for_anchor(anchor_id)
    DS->>O: scout(anchor, briefs)
    O->>S: propose_candidates(anchor, corpus)
    S-->>O: CandidateProposal[]
    O->>O: compute JudgeSignals per candidate
    DS->>O: judge_many_with_diagnostics(anchor, candidates)
    O->>J: judge_relations_with_signals(...)
    J-->>O: JudgeResult[]
    O->>R: review_relations_with_signals(...)
    R-->>O: reviewed JudgeResult[]
    O-->>DS: CandidateAssessment[]
    DS->>GS: add_edges(accepted)
    DS->>C: curate(anchor, accepted, rejected)
    C-->>DS: provider_payload
    DS->>MS: merge + persist memory
```

### 6.3 Retry logic

There is a controlled retry path in [discovery.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/services/discovery.py):

- only for live providers
- only when no accepted edges were produced
- only for anchors above a priority threshold
- retry uses an expanded scout pass

This is not a free-running agent loop.
It is a bounded second pass under orchestrator control.

```mermaid
flowchart TD
    A["初次 discovery / Initial discovery"] --> B{"是否 live provider?\nIs live provider?"}
    B -- "否 / No" --> Z["结束 / Finish"]
    B -- "是 / Yes" --> C{"是否零接纳边?\nNo accepted edges?"}
    C -- "否 / No" --> Z
    C -- "是 / Yes" --> D{"anchor 优先级足够高?\nAnchor priority high enough?"}
    D -- "否 / No" --> Z
    D -- "是 / Yes" --> E["扩展 scout / Expanded scout"]
    E --> F["补充 judge + review / Retry judge + review"]
    F --> Z
```

---

## 7. Main Agent to Subagent Call Semantics

### 7.1 The orchestrator is not passive

The orchestrator is not just a router.
It does substantial pre- and post-processing:

- caching embeddings
- deriving surrogate query terms
- computing bridge information gain
- computing duplicate penalties
- assembling `JudgeSignals`
- ranking discovery anchors
- selecting which candidates are allowed into judge/reviewer

This means:

- subagents are specialized decision-makers
- the orchestrator is the global controller and evidence builder

### 7.2 Call graph

```mermaid
flowchart TD
    A["Anchor DocBrief / 锚点简档"] --> B["主编排器 / LogicOrchestrator"]
    B --> C["候选召回 / scout()"]
    C --> D["CorpusScoutAgent / 侦察代理"]
    D --> E["CandidateProposal[] / 候选提议"]
    E --> B
    B --> F["信号构造 / build JudgeSignals"]
    F --> G["RelationJudgeAgent / 裁决代理"]
    G --> H["JudgeResult[] / 裁决结果"]
    H --> I["EdgeReviewerAgent / 复核代理"]
    F --> I
    I --> J["CandidateAssessment[] / 候选评估"]
    J --> K["边构造 / LogicEdge build"]
    K --> L["MemoryCuratorAgent / 记忆整理代理"]
```

---

## 8. Context Passing Rules

The project does not pass one giant mutable conversation transcript.
Instead, it passes structured context between explicit stages.

### 8.1 Profile stage context

Input:

- `DocRecord`

Output:

- `DocBrief`

### 8.2 Scout stage context

Input:

- anchor `DocBrief`
- full corpus of `DocBrief`

Output:

- `CandidateProposal[]`

### 8.3 Judge stage context

Input:

- anchor `DocBrief`
- candidate `DocBrief`
- `JudgeSignals`

Output:

- `JudgeResult`

### 8.4 Review stage context

Input:

- anchor `DocBrief`
- candidate `DocBrief`
- `JudgeSignals`
- initial `JudgeResult`

Output:

- reviewed `JudgeResult`

### 8.5 Curate stage context

Input:

- anchor `DocBrief`
- accepted `LogicEdge[]`
- rejected doc ids

Output:

- provider payload for memory merge

```mermaid
flowchart LR
    A["DocRecord / 全量文档"] --> B["DocBrief / 简档"]
    B --> C["CandidateProposal / 候选提议"]
    B --> D["JudgeSignals / 信号包"]
    C --> D
    D --> E["JudgeResult / 裁决"]
    E --> F["Reviewed JudgeResult / 复核结果"]
    F --> G["LogicEdge / 最终边"]
    G --> H["Memory Payload / 记忆载荷"]
```

---

## 9. Memory Management

Memory is file-backed and explicitly merged, not agent-hidden.

The memory stores are:

- [anchor_memory.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/memory/anchor_memory.py)
- [semantic_memory.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/memory/semantic_memory.py)
- [graph_memory.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/memory/graph_memory.py)
- [curator.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/memory/curator.py)

### 9.1 Anchor memory

Anchor memory stores per-anchor operational state:

- explored docs
- rejected docs
- accepted edge ids
- active hypotheses
- successful and failed search patterns
- rejection reasons
- candidate scores
- accepted edge scores

### 9.2 Global semantic memory

Global semantic memory stores reusable corpus-level patterns:

- canonical entities
- aliases
- relation patterns
- rejection patterns

### 9.3 Graph memory

Graph memory stores graph-level stats:

- accepted edge count
- last revalidation time

```mermaid
flowchart TD
    A["Accepted / Rejected Results\n接纳与拒绝结果"] --> B["MemoryCuratorAgent / 记忆整理代理"]
    B --> C["provider_payload / 模型侧记忆载荷"]
    C --> D["MemoryCuratorService.merge / 本地合并器"]
    D --> E["AnchorMemoryStore / 锚点记忆"]
    D --> F["SemanticMemoryStore / 全局语义记忆"]
    D --> G["GraphMemoryStore / 图统计记忆"]
```

### 9.4 Why memory is split

The split avoids mixing different timescales:

- anchor memory:
  - local, per-document, operational
- semantic memory:
  - corpus-level reusable abstraction
- graph memory:
  - system-level monitoring state

This is a practical replacement for letting an opaque agent conversation history accumulate forever.

---

## 10. Agent Loop Implementation Logic

The agent loop is implemented as a bounded deterministic controller around model calls.

### 10.1 Loop structure

For each anchor:

1. compute whether the anchor is eligible for discovery
2. scout candidates
3. build feature bundles
4. judge candidates
5. review candidates
6. accept top utility edges
7. write graph
8. update memory
9. optionally retry once

### 10.2 Why this is still an agent loop

It is an agent loop because:

- specialized roles exist
- different roles operate on different subproblems
- role-specific skills guide their behavior
- the provider may use remote reasoning for each role
- outputs from one role become context for the next role

It is not an open-ended autonomous loop because:

- maximum passes are bounded
- graph writes happen through explicit gating
- memory writes happen through explicit merge logic
- the orchestrator owns final control

```mermaid
flowchart TD
    A["选择 anchor / Select anchor"] --> B["Scout / 侦察"]
    B --> C["Judge / 裁决"]
    C --> D["Review / 复核"]
    D --> E{"有高质量边?\nAny high-utility edge?"}
    E -- "是 / Yes" --> F["写图 / Write edges"]
    F --> G["更新记忆 / Update memory"]
    G --> H["下一 anchor / Next anchor"]
    E -- "否 / No" --> I{"允许重试?\nRetry allowed?"}
    I -- "是 / Yes" --> J["扩展候选 / Expanded scout"]
    J --> C
    I -- "否 / No" --> H
```

---

## 11. Capability Management

Capabilities are distributed across three layers:

- provider capabilities
- tool capabilities
- skill capabilities

### 11.1 Provider capabilities

The provider is the true execution backend for subagent reasoning.

Examples:

- `profile_doc`
- `propose_candidates`
- `judge_relation_with_signals`
- `review_relation_with_signals`
- `curate_memory`

So the provider owns:

- remote reasoning access
- structured output generation
- embedding calls
- live reasoning toggles

### 11.2 Tool capabilities

The tool registry is defined in [registry.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/agents/tools/registry.py).

Available tools:

- `search_summaries`
- `lookup_entities`
- `get_hnsw_neighbors`
- `read_doc_brief`
- `read_doc_full`
- `commit_logic_edge`
- `load_anchor_memory`
- `update_global_memory`

These are the environment-facing abilities available to a deepagent runtime.

### 11.3 Skill capabilities

Skills are stored under [agents/skills](/Users/armstrong/gl-hnsw/src/hnsw_logic/agents/skills).

Configured skill mapping:

- `doc_profiler`
  - `doc_briefing`
  - `entity_canonicalization`
- `corpus_scout`
  - `corpus_navigation`
- `relation_judge`
  - `evidence_linking`
  - `relation_typing`
- `edge_reviewer`
  - `edge_utility`
  - `signal_fusion`
- `memory_curator`
  - `memory_update`

```mermaid
flowchart TD
    A["能力层 / Capability Layers"] --> B["Provider Methods / Provider 能力"]
    A --> C["Tools / 工具能力"]
    A --> D["Skills / 技能能力"]

    B --> B1["profile / scout / judge / review / curate"]
    C --> C1["search_summaries"]
    C --> C2["lookup_entities"]
    C --> C3["read_doc_full"]
    C --> C4["commit_logic_edge"]
    D --> D1["doc_briefing"]
    D --> D2["corpus_navigation"]
    D --> D3["relation_typing"]
    D --> D4["signal_fusion"]
    D --> D5["memory_update"]
```

---

## 12. Skills Mechanism

The `skills` mechanism is prompt-level capability shaping, not executable business logic.

### 12.1 What skills do

Skills tell each role:

- what output shape to prefer
- what evidence to prioritize
- when to abstain
- what kind of mistakes to avoid

Examples:

- `edge_utility`
  - prefer retrieval-useful edges over topical similarity
- `signal_fusion`
  - treat local signals as grounded evidence
- `relation_typing`
  - constrain canonical relation space

### 12.2 What skills do not do

Skills do not:

- persist data directly
- replace local scoring code
- bypass graph write gates
- override the orchestrator

### 12.3 Effective execution model

The effective execution model is:

`local code computes signals -> skill tells the role how to read them -> provider generates structured output -> orchestrator decides`

```mermaid
flowchart LR
    A["本地规则 / Local Code"] --> B["JudgeSignals / 信号包"]
    C["Skill Prompt / 技能提示"] --> D["Provider Reasoning / 模型推理"]
    B --> D
    D --> E["Structured Output / 结构化输出"]
    E --> F["Orchestrator Gate / 编排器门控"]
    F --> G["Persisted Edge or Memory / 持久化结果"]
```

---

## 13. DeepAgents Integration

`AgentFactory.try_create_deep_agent()` builds a deepagents runtime only when:

- `runtime_mode == "deepagents"`
- provider is `OpenAICompatibleProvider`
- API key exists
- deepagents and langchain bindings are importable

When enabled, the deepagent is created with:

- model: `ChatOpenAI`
- skills root: `src/hnsw_logic/agents/skills`
- tools from the registry
- subagent specs from config
- `FilesystemBackend`

```mermaid
flowchart TD
    A["AgentsConfig / 代理配置"] --> B{"runtime_mode == deepagents?"}
    B -- "否 / No" --> X["不创建 deepagent / No deepagent"]
    B -- "是 / Yes" --> C{"Provider 可用?\nOpenAI-compatible?"}
    C -- "否 / No" --> X
    C -- "是 / Yes" --> D{"API key + imports 就绪?\nCredentials and imports ready?"}
    D -- "否 / No" --> X
    D -- "是 / Yes" --> E["create_deep_agent(...)"]
    E --> F["FilesystemBackend / 文件系统后端"]
    E --> G["Tools / 工具"]
    E --> H["Skills Root / 技能目录"]
    E --> I["Subagent Specs / 子代理规格"]
```

### Important nuance

The project is not using the deepagents object as the only execution engine today.
It is better to think of it as:

- an integration layer
- a future richer runtime path
- a capability shell around the same subagent roles

The explicit typed orchestrator remains the authoritative control path.

---

## 14. Failure Handling and Guard Rails

The system contains several explicit guard rails:

- bounded retry only
- graph write happens after judge and review
- duplicate-edge suppression
- mirror-edge generation only for selected relation types
- anchor eligibility filtering before discovery
- memory merge through deterministic code, not free-form agent state

```mermaid
flowchart TD
    A["候选边 / Candidate Edge"] --> B["Judge / 裁决"]
    B --> C["Reviewer / 复核"]
    C --> D{"utility 足够?\nUtility high enough?"}
    D -- "否 / No" --> E["Reject / 拒绝"]
    D -- "是 / Yes" --> F{"重复或风险过高?\nDuplicate or risky?"}
    F -- "是 / Yes" --> E
    F -- "否 / No" --> G["Persist Edge / 写入边"]
    G --> H["Curate Memory / 更新记忆"]
```

---

## 15. Practical Reading of the System

If you want to understand the system in execution order, read it like this:

1. [bootstrap.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/services/bootstrap.py)
2. [pipeline.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/services/pipeline.py)
3. [discovery.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/services/discovery.py)
4. [orchestrator.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/agents/orchestrator.py)
5. [provider.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/embedding/provider.py)
6. [curator.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/memory/curator.py)
7. [registry.py](/Users/armstrong/gl-hnsw/src/hnsw_logic/agents/tools/registry.py)

This order mirrors the real implementation dependencies.

---

## 16. Summary

The agent runtime of `gl-hnsw` is best understood as:

- offline-first
- orchestrator-controlled
- multi-role
- structured-context-driven
- memory-explicit
- deepagents-compatible
- but not dependent on open-ended autonomous agent loops

That is the core reason the system can remain both:

- `agent-centric`
- and still operationally stable enough to build a retrieval index.
