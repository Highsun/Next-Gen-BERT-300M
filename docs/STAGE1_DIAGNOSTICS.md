# Stage 1 后续诊断：先评估，不追加训练

适用状态：B0、L3、G2 各 1000 步已完成，`outputs/stage1-s17` 和三组 final checkpoint 都还在。该脚本不创建优化器、不反向传播、不更新参数，不修改已有 stage1 文件。首次执行仍须在 Ubuntu + CUDA 服务器验证；本地仅完成静态检查。

## 要回答的问题

|诊断|比较方式|可支持的结论|
|---|---|---|
|原始底座参考|P0 原始 mmBERT，B0/L3/G2 适配后权重；共用编码实现|当前 Wikipedia 样本的 MLM 变化；不等于原有全部应用能力保留|
|长度相关损伤|按自然文本长度分层，分别编码至 512/2048/8192 上限|损失差距在哪些长度、哪些语言更明显|
|远端内容敏感性|同一长文、同一目标位置和标签，远端内容保留/打乱/替换为掩码|远端内容被改变时预测如何变化；不等于正确利用远端证据|
|摘要门控|逐层统计实际 gate 激活均值、标准差、低于 0.01/高于 0.1 的比例|排查 G2 是否几乎退化为局部注意力；不能直接归因训练失败|

长度探针沿用 15% 预测目标和 80/10/10 污染策略。不同长度前缀的目标集合不同，因此不能直接用它们之间的 loss 差值声称增加上下文带来的收益。实际 token 长度与文档数一起报告；不足上限的样本不会补造内容，相同实际长度的重复前缀会去重。`natural_length` 是源文档编码至 8192 的长度，不推断被截断部分长度。

上下文探针只在长文的前部（约 10%）、中部、后部（约 90%）分别选最多 16 个固定预测 token，将这些目标全部替换为 mask。保留目标所在区间及两端各 256 token 的邻域，其余正文 token 才进入干预。三种条件共享序列长度、位置、attention/content mask、目标标签与局部输入：

- `full`：远端内容保留。
- `far_shuffled`：只打乱远端 token，保留其 token 多重集合。
- `far_masked`：只将远端 token 替换为 mask；这些位置仍参与注意力，**不是删除 token 或阻断注意力**。

远端打乱和大量 mask 都可能造成分布外输入。若 `corrupted loss - full loss` 为正，只能说原始远端内容对这些预测有帮助，不能据此声称具备长程推理。若差值不明显，也不能断言模型没有远程能力：这批随机 MLM 目标未必需要远程证据。独立的真实任务和反事实证据探针仍需另做。

## 服务器执行

将更新包中的文件覆盖到原项目。无需安装新依赖、下载新数据或修改权重。打开新的终端后：

```bash
cd /root/autodl-tmp/Next-Gen-BERT-300M
source .venv/bin/activate
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 只准备冻结输入：CPU 分词，尚不加载模型。
python scripts/diagnose_stage1.py --action prepare
```

默认每语言每个自然长度桶至多 12 篇（总计至多 72 篇），长文探针每语言另外选 8 篇实际长度至少 4096 的文档。两种探针之间允许复用文档，在统计时按源文档聚类。小桶不够 12 篇时保留全部并报告数量；长文不足 8 篇则明确报错，不静默降低要求。若确需缩小长文数，在**新输出目录**显式设置 `--long-per-language`。

准备完成后查看打印的 `case_count` 和 `max_forward_calls_all_arms`。默认上限为 1152 次单文档前向（4 组，共享固定输入）；这不是 1152 个训练步骤。尚未实测所需时间，不能用旧训练耗时直接保证评估耗时。

启动正式诊断：

```bash
mkdir -p logs
nohup bash -c '
set -e
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u scripts/diagnose_stage1.py --action all
' > logs/stage1-diagnostics-s1701.log 2>&1 < /dev/null &

tail -n 30 -f logs/stage1-diagnostics-s1701.log
```

P0、B0、L3、G2 分别在独立进程中顺序评估，避免同时占用显存。P0 使用相同编码实现和固定 revision，关闭 LoRA；未训练的公共任务头不参与 MLM。其余三组加载已有 final，不重新训练。G2 gate 记录有额外开销，本轮不报告性能提升。

新终端查看同一个日志即可，不必恢复原 shell。进程中断后重新运行 `--action all`：输入已冻结，不会重新准备；结果身份与哈希一致的已完成组会跳过，未完成组从头评估。请勿同时启动两个诊断进程。输入准备中断时保留旧目录，用新 `--output` 重做，不将半成品当作冻结输入。

如使用自定义 `--output`，prepare、all、summary 三个阶段必须使用同一路径。准备后的 manifest 为实际输入配置；恢复 all 不会用命令行默认值重新抽样。更换种子或样本量必须换新目录重新 prepare。

## 输出和回传

`outputs/stage1-diagnostics-s1701/` 包含：

- `manifest.json`：源文件、原配置、模型 revision、抽样参数及输入哈希。
- `cases.jsonl`：冻结 token 输入、目标和干预条件。留在服务器，不必回传。
- 四组 `*-metrics.json`：模型/输入身份、预测文件哈希、G2 gate 激活统计。
- 四组 `*-predictions.jsonl`：逐案例实际长度、目标签名、目标数、NLL 总和、准确数和平均 loss，不含正文。
- `summary.json`：文档宏平均与 masked-token 加权 loss、模型差异、远端干预差异、按源文档聚类的 bootstrap 区间。

置信区间只反映该固定种子下的文档抽样差异；不是训练多种子结论，没有多重比较校正。抽样按自然长度分层，整体均值也不是原始验证集分布的无偏估计。长文只有每语言 8 篇，先用于定位问题。

汇总完成后可执行：

```bash
tar -czf stage1-diagnostics-review.tar.gz \
  outputs/stage1-diagnostics-s1701/manifest.json \
  outputs/stage1-diagnostics-s1701/summary.json \
  outputs/stage1-diagnostics-s1701/*-metrics.json \
  outputs/stage1-diagnostics-s1701/*-predictions.jsonl \
  logs/stage1-diagnostics-s1701.log

tar -tzf stage1-diagnostics-review.tar.gz
```

下载小包后可以关机，保留实例、输入与现有权重。无需上传基础权重和 cases.jsonl。不要仅为使 G2 的 MLM 数字好看而追加训练：先检查长度、底座变化和 gate，再决定是否进行任务验证或否定当前结构。
