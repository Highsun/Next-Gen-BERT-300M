# Next-Gen-BERT-300M

面向 **Ubuntu + CUDA、单张 RTX 4090 或同级 GPU** 的双语编码器研究代码。
基于 mmBERT，将部分全局层替换为局部／摘要交互，并研究块级检索和查询全局重排。

**当前交付状态：只完成代码与本地静态检查。未安装依赖、下载权重、导入模型、运行单元测试、训练或测量 CUDA 性能。** 可运行性、数值兼容和显存容量必须在 AutoDL 验证，不能把本仓库当作已复现实验结果。

## 结构与实验入口

| 变体 | 行为 |
| --- | --- |
| B0 | 原 mmBERT 的全局／局部周期，使用与候选统一的执行路径 |
| L3 | 仅保留 0／9／21 层为全局，原其他全局层改为局部 |
| M1 | L3 加单均值摘要，固定 0.5 混合 |
| M2 | 均值＋可学习池化摘要，固定混合 |
| G2 | 双摘要＋每 token／head 门控 |
| C2 | 双摘要＋可学习常数门控，对照 token 自适应是否必要 |

保留原 embeddings、归一化、QKV/O、GeGLU、预测头和 RoPE 模块。模型逐文档处理，隔离文档并去掉 padding 后计算；没有跨文档 packing。摘要由腐蚀后的当前层表示计算，不读取 MLM 原文。摘要位置为块中心，原 token 与字符 offsets 保留。

`tiled` 后端在不启用查询全局通道时，将所有局部窗口批量送入 SDPA；启用查询全局通道时保留逐 tile 的局部／query 去重路径。两者都不会生成完整局部 N×N mask。完整全局层使用 PyTorch SDPA，摘要使用短 K/V cross-attention。`reference` 是限长 1024 的稠密正确性参考。该实现**尚非定制融合 kernel，批量窗口也有复制与额外显存开销，不能预先保证更快**。

## 安装：只在租用的 CUDA 服务器执行

建议 Ubuntu 22.04、Python 3.11、支持 CUDA 12.8 wheel 的 NVIDIA 驱动。先在 AutoDL 确认 `nvidia-smi`，不要因安装脚本而修改系统驱动。

```bash
git clone https://github.com/Highsun/Next-Gen-BERT-300M.git
cd Next-Gen-BERT-300M
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

脚本仅创建 `.venv`，安装 PyTorch 2.7.1 cu128 与项目依赖，不使用 sudo。Transformers 固定 4.57.1，因为本项目显式依赖该版本 ModernBERT 内部模块结构。其他依赖范围见 `pyproject.toml`；安装后的精确环境写入 `requirements-resolved.txt`，正式实验应单独归档它。换 GPU/CUDA 时先核对官方 wheel 和驱动兼容，不盲目沿用。

配置中的 `revision=main` 在服务器首次构建时解析为不可变的 Hugging Face commit，随后存入配置和 provenance；所有正式消融配置应改成**同一解析 SHA**再开跑。依赖下载不计入本地作者阶段。

## 必须先做的 CUDA 检查

```bash
pytest -q tests/test_cuda_contracts.py
nextbert pilot --config configs/B0.json --output runs/pilot-B0.json --iterations 20 --backward
nextbert pilot --config configs/G2.json --output runs/pilot-G2.json --iterations 20 --backward
```

测试覆盖：B0 与原 HF hidden states 对齐；tiled／reference 的输出和梯度；零过渡时恢复原全局路径；各变体反向；padding 和文档隔离。测试采用随机小模型，之后还需对真实 mmBERT 检查点、BF16、8K 和任务数据做一致性验证。

Pilot 使用重复合成文本、3 次预热，记录 512/2K/4K/8K、显存、OOM、p50/p95；不是质量评测，也不是论文正式计时协议。正式速度评测应改到至少 20 次预热、100 次计时、三轮重复，覆盖实际长度分布。`--backward` 只测前反向，不包含 AdamW 状态与 optimizer.step 的峰值；完整训练是否放得下还需短程真实训练确认。

## 数据：显式 JSONL，不隐式下载基准

每行必须有唯一字符串 `id`，建议再有 `language`、`source`、`group_id`。同原始文档的窗口、译文、近重复簇共享 group_id。准备原始数据时遵守各数据集条款，切分和去污染在正式训练前冻结。

```json
{"id":"u1","group_id":"article1","language":"zh","text":"北京大学位于北京。"}
```

各任务附加字段：

| task | 字段 |
| --- | --- |
| mlm | `text` |
| classification | `text`, 整数 `label`，从 0 连续编号 |
| ner | `text`, `entities: [{"start":0,"end":4,"type":"ORG"}]`；Unicode 字符偏移，end 不包含 |
| qa | `query`, `text`, `answers: [{"start":0,"text":"北京大学"}]`；无答案用空列表 |
| retrieval | `query`, `positive`, `negatives`（字符串数组，默认预制 7 个） |
| rerank | `query`, `text`, 二值 `label`；评估另有 `query_id`, `doc_id`，可选整数 `relevance` |

NER 使用不重叠实体、BIO 标签；配置 label_names 必须与类型对应。QA 默认首个位置为 null，正样本被截断时**拒绝训练/评估**，先用答案覆盖窗口整理数据，不将截断误标为无答案。当前没有自动聚合多窗口答案：正式 SQuAD/CMRC 评估需在官方 adapter 中实现窗口分数合并。

冻结自建语料切分：

```bash
nextbert prepare --input raw/wiki.jsonl --output-dir data/unsupervised --task mlm --train-percent 99 --dev-percent 1
nextbert prepare --input raw/pairs.jsonl --output-dir data/retrieval --task retrieval --train-percent 90 --dev-percent 10
```

`prepare` 输出 SHA-256 清单、按 group hash 切分与文本精确去重；近重复聚类需上游提供 group_id。官方 train/dev/test 不要重新混合切分。命令失败留下的目录不是有效冻结集，只有完整 manifest 才能使用。这里提供统一训练契约，不冒充已完成 Wikipedia／mMARCO／MIRACL／MLDR 官方数据整理。

## 训练、迁移、恢复

```bash
nextbert train --config configs/B0.json
nextbert train --config configs/G2.json
nextbert train --config configs/G2-retrieval.json --init runs/G2/final
nextbert train --config configs/G2-rerank.json --init runs/G2/final
nextbert train --config configs/G2-classification.json --init runs/G2/final
nextbert train --config configs/G2-ner.json --init runs/G2/final
nextbert train --config configs/G2-qa.json --init runs/G2/final
```

配置是**模板**，运行前设置真实数据路径、模型 SHA、总预算和 labels。任务配置默认 G2，但 B0 对照必须复制成相同任务与预算、仅改变 variant/output_dir；不能只训练候选。`--init` 仅迁移同结构同 LoRA rank 的主干与摘要分支，任务 heads 独立初始化。B0 的目标配置应从 B0 的 MLM 检查点迁移。

```bash
nextbert train --config configs/G2.json --resume runs/G2/step-000100
```

`--resume` 要求配置、数据 SHA 完全相同，恢复模型、AdamW、样本 cursor、RNG、累计 tokens 和估算 FLOPs。LoRA 模式使用 trainable-v1 格式，仅保存 LoRA、新增模块和任务头及优化器／RNG；恢复时从不可变 SHA 加载共享底座。全参模式仍保存完整权重。默认 keep_checkpoints=2，成功写入新检查点后仅保留本运行最近两个 step 目录，另保留 final；不要把唯一检查点放在这些自动轮换目录中。改变任务或预算使用 `--init`，不要伪装 exact resume。只加载可信的自行生成 checkpoint；完整恢复文件包含 Python pickle 状态，不加载来源不明的 `.pt`。

默认：LoRA rank 16，QKV/O 与 FFN 线性层，microbatch=1、累积 8、BF16、gradient checkpointing。新增摘要与 heads 训练，底座冻结。rank=0 为全参。query-global 仅在 qa/rerank 允许；检索建库绝不读取 query。MLM 只对选中位置执行词表预测头，避免 8K×256K 的全量 logits。

检索训练使用一个 query、一个正例和固定负例，显式交叉熵；普通梯度累积**不扩大** in-batch negatives。当前不提供梯度缓存或分布式训练；长文多负例显存不够时，应所有组统一减少负例或另行实现已验证缓存，不能只给候选组省预算。

MLM 提供动态 token 腐蚀与可选片段腐蚀（span_batch_probability）。不默认加入 RTD 或 diffusion；它们是方案中的后续独立研究，不属于本次前三项结构实验的实现范围。

## 计算预算与日志的口径

输出 `provenance.json`、`train.jsonl`、阶段 checkpoint 和 final。首次记录 GPU/CUDA/PyTorch、解析模型 SHA、配置、数据 SHA、参数量与可训练量。

`max_steps`、`max_seconds`、`max_estimated_flops` 任一上限到达即在完整 optimizer step 后结束。解析估算覆盖主要投影、attention、MLM 词表与过渡双路径，并用系数估计反向／重计算；**不是实际校准 FLOPs**，不包含全部小算子和检索评分。正式等计算比较必须在 AutoDL 用 profiler 校准，冻结预算再确认。预算最多超出一个 optimizer step；失败及额外调参另计。

长度课程使用短文 warmup，之后按累计 token 缺额追踪 30/30/40 的 512/2K/8K 桶；裁剪与有限样本会有偏差，日志记录实际桶 tokens。中英采样比例、文档去重和官方任务数据均需在输入清单中控制；此处不把随机长度采样等同于完整的数据配方。

## 评估与检索

```bash
nextbert evaluate --checkpoint runs/G2-classification/final --data data/classification/test.jsonl --output outputs/classification.json
nextbert evaluate --checkpoint runs/G2-ner/final --data data/ner/test.jsonl --output outputs/ner.json
nextbert evaluate --checkpoint runs/G2-qa/final --data data/qa/test.jsonl --output outputs/qa.json
nextbert evaluate --checkpoint runs/G2-rerank/final --data data/rerank/test.jsonl --output outputs/rerank.json
```

分类输出 accuracy/macro-F1；NER 输出原文字符实体 F1 和 subtoken BIO F1；QA 输出通用中文字符／英文词归一化 EM/F1；重排输出候选内 nDCG/MRR。**通用评分器不能替代官方基准脚本**；最终报告须对导出的预测做官方核对。逐样本输出保存 base_id、condition、distance 等干预信息。

检索 corpus/queries 均为 `{"id":"...","text":"..."}`；qrels 每行 `{"id":"j1","query_id":"q1","doc_id":"d1","relevance":1}`；固定候选每行 `{"id":"q1","doc_ids":["d1","d2"]}`。

```bash
nextbert search --checkpoint runs/G2-retrieval/final --queries data/queries.jsonl --corpus data/corpus.jsonl --qrels data/qrels.jsonl --candidates data/fixed-candidates.jsonl --output outputs/retrieval.json
```

没有 `--candidates` 时以全局向量精确 CPU top-100 召回，再 CUDA 块打分；带候选时直接采用该列表，便于 E2 消融。该实现是透明的精确研究基线，不是生产 ANN 速度实现；大规模语料的 RAM 与建库成本要实际 profiling。导出 TREC run、每 query 指标、候选召回率与数据 SHA。

representation=`single` 用全局向量；`late` 全文编码后分块；`independent` 另外独立编码各块，保留全文全局向量以固定混合评分。后者多一次块编码，计费，不能声称与单向量同计算。不同方案保持块边界、128 维、候选与训练标签一致。原生 B0+late 是必备对照。

`compare_results.py` 可对齐逐样本 JSONL，按 base_id 做配对聚类 bootstrap；它只估计样本不确定性，不代替三个训练 seeds、非劣界或多重比较校正。长程数据需包含稀有事实密度、边界移动、反事实与缺证据条件，不能用重复 pilot 文本作为质量证据。`scripts/make_diagnostics.py` 提供可复现的单证据／反事实／缺证据起始探针，按实际 tokenizer 检查长度，并保留 base_id/group_id；它不是完整跨段组合诊断，需人工抽查和扩展。

```bash
python scripts/make_diagnostics.py --revision <固定模型SHA> --output raw/diagnostics.jsonl
nextbert prepare --input raw/diagnostics.jsonl --output-dir data/diagnostics --task classification
```

该探针有 0/1/2 三个标签，分类配置需设 num_labels=3。`raw/` 原始数据也不提交版本库。

## 本地静态检查与仓库边界

```bash
python3 scripts/static_check.py  # Python 3.11+；仅解析/编译，不导入项目
bash -n scripts/setup_ubuntu.sh scripts/run_matrix.sh
```

报告保存在 `docs/static-check.json`。静态通过不代表依赖接口、数值、梯度或 GPU 通过。本仓库不使用 GitHub Actions，也不包含 workflow；检查由用户在授权的环境中手动运行。

AutoDL 的分步启动流程见 [docs/AUTODL.md](docs/AUTODL.md)。

所有环境局限于 `.venv`；删除 `.venv` 可移除项目依赖。系统层不安装 CUDA/驱动。模型缓存遵循 Hugging Face 缓存目录，可在 AutoDL 显式设置 HF_HOME 指向数据盘。数据、权重、实验输出、密钥和环境文件均被 gitignore 排除。

研究设计与论文依据见 `docs/`。自有代码 MIT；权重、数据和依赖遵循各自许可证，见 NOTICE。
