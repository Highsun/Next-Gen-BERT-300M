# AutoDL：50GB 数据盘启动流程

以下命令全部在 **AutoDL 的 JupyterLab Terminal 或 SSH 终端**执行，不在 Mac 上执行。
镜像推荐 PyTorch 2.7.0 / Python 3.12 / CUDA 12.8，单张 4090 24GB。
本仓库不使用 GitHub Actions，不能跳过服务器数值验证。

## 1. 拉取代码与安排缓存

```bash
cd /root/autodl-tmp
# 若 GitHub/Hugging Face 直连较慢，AutoDL 官方提供以下可选加速：
source /etc/network_turbo
git clone https://github.com/Highsun/Next-Gen-BERT-300M.git
cd Next-Gen-BERT-300M
export HF_HOME=/root/autodl-tmp/hf-cache
mkdir -p "$HF_HOME" logs
nvidia-smi
df -h / /root/autodl-tmp
```

已克隆则在仓库内先 `git pull --ff-only`，不要重复克隆。私有仓库若要求认证，使用你自己的 GitHub 凭据；不要把 token 写进 Git remote、聊天或代码。没有 `/etc/network_turbo` 时跳过该行。代理用于 GitHub/HF，可能影响其他域名；安装 PyTorch 时可在当前终端 `unset http_proxy https_proxy`，准备 HF 数据时再 source。

## 2. 安装项目隔离环境

```bash
unset http_proxy https_proxy
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"
```

应显示 PyTorch 2.7.1+cu128、CUDA=True、4090、BF16=True。如果 venv/ensurepip 不存在，先检查 `which python3`，应使用镜像提供的 Python/Conda Python；不要直接把全部依赖装到系统 Python。安装失败先看日志，不用反复下载模型。

pip 不保留安装缓存；.venv 在数据盘项目目录。基础权重只缓存一份，LoRA 检查点不重复存储冻结底座。保留至少 10GB 空闲空间，数据子集准备完后确认 `du -sh . "$HF_HOME"`。

## 3. 先验证实现，再测 8K

```bash
pytest -q tests/test_cuda_contracts.py
nextbert pilot --config configs/B0.json --output outputs/pilot-B0.json --lengths 512,2048,8192 --iterations 10 --backward
nextbert pilot --config configs/G2.json --output outputs/pilot-G2.json --lengths 512,2048,8192 --iterations 10 --backward
```

任一测试失败则停止，不跑长训练。小随机模型测试不下载真实权重，pilot 会下载 mmBERT。pilot 仅测前反向，不包含 optimizer.step 的完整峰值，仍需下一步真实试训。

## 4. 准备有限的真实中英语料

```bash
source /etc/network_turbo
python scripts/prepare_wikipedia.py --output-dir data/wiki-starter --tokens-per-language 2000000
```

脚本流式读取 Wikipedia `20231101.zh/en`，不下载完整语料；每篇保留首个连续 <=8190-token 片段，跳过极短文本，精确去重，按文档 ID hash 切 train/dev=99/1，写入来源和文件 SHA。约 2M tokens/语言是启动子集，不是正式完整消融数据；不保证覆盖最终基准分布，也尚未去除与测试基准的近重复。

生成 manifest，以及每个变体的 `*-smoke.json` 和 `*-starter.json`。它们使用相同的不可变模型 SHA、数据与 seed。输出目录已存在时拒绝覆盖；失败后检查并保留/改名该目录，再用新路径重试，别把半成品当冻结数据。

## 5. 20 步真实训练验证

```bash
nextbert train --config data/wiki-starter/B0-smoke.json
nextbert train --config data/wiki-starter/G2-smoke.json
nextbert evaluate --checkpoint runs/B0-smoke/final --data data/wiki-starter/dev.jsonl --output outputs/B0-smoke-dev.json
nextbert evaluate --checkpoint runs/G2-smoke/final --data data/wiki-starter/dev.jsonl --output outputs/G2-smoke-dev.json
```

两个实验依次跑；每个 20 个 optimizer steps、microbatch=1、累积2次、长度512。确认 loss 有限、无 OOM，compact checkpoint 能恢复评估。MLM loss 下降不能作为下游收益证据。这一步只验证管线，绝不是论文结果。

## 6. 再运行 8K starter 配置

先查看 pilot 与 smoke 的耗时/显存并冻结预算，勿直接启动全部矩阵。确认后可先跑 B0：

```bash
mkdir -p logs
nohup nextbert train --config data/wiki-starter/B0-starter.json > logs/B0-starter.log 2>&1 &
echo $! > logs/B0-starter.pid
tail -f logs/B0-starter.log
```

Ctrl+C 只退出 tail，不会停止 nohup 训练。等待 B0 完成，再以同样方式跑 G2；不要在同一张卡并行启动两个训练。`starter` 含 8K 桶和 1000 步，仍只是初始配方，需要结合实际曝光与计算预算确定正式确认实验。先充分训练底座，再迁移检索等任务，不能拿 smoke 直接作正式主干。

默认最近两个 step 检查点自动轮换，final 独立保留。中断后选择确实存在的最新 step，使用同一个原配置：

```bash
nextbert train --config data/wiki-starter/B0-starter.json --resume runs/B0-starter/step-000100
```

上面 step-000100 只是示例，可能已被轮换；按 `ls runs/B0-starter/step-*/checkpoint.pt` 的实际结果填写。不要用 --init 冒充恢复；--init 是新阶段，会重置优化器和样本进度。

完成首次验证后应保存：pytest 输出、两份 pilot JSON、smoke train.jsonl/provenance、dev 评估和磁盘使用情况。再决定正式语料规模、种子、任务数据以及统一 FLOPs/GPU 时间预算。训练权重和语料不通过 GitHub 上传。

参考：AutoDL 学术加速 https://www.autodl.com/docs/network_turbo/ ；HF streaming https://huggingface.co/docs/datasets/stream 。

## 7. CUDA 测试通过后的真实近8K短训与耗时定位

若 pilot 成功而新模型更慢，不应直接开始完整消融。以下脚本只用现有缓存和冻结训练集，选择每语言4篇、实际编码长度8000–8192的真实文档，两组使用相同数据，依次训练20步、累积2次，保存完整优化器状态。小集合会重复曝光，只用于显存与开销验证，不作为质量结论。

```bash
export HF_HOME=/root/autodl-tmp/hf-cache
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/long_check.py --action train
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/long_check.py --action profile --variant G2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/long_check.py --action profile --variant B0
```

输出集中在 `data/wiki-long-check/`：`manifest.json`、`training-summary.json`、两份 `*-operators.txt`。训练检查点位于 `runs/*-long-check/`。profile先做两步预热，再收集两步，不替代正式性能测量；默认不保存大型 trace，必要时才显式加 --trace。profile可能受容器CUPTI权限限制，如失败保留报错和训练汇总，不重新训练。

如果训练已完成，不重复 --action train，使用 --action summary 重建汇总即可；若中断，在原配置上使用 nextbert train --resume 恢复完成后再汇总。不要删除旧的 smoke 或 pilot 结果。此阶段不改变模型计算定义，也不重跑已通过的正确性测试。

## 8. 第一轮profile后的实现优化复测

初始真实长文20步：B0约445.6秒，G2约470.6秒。算子报告中，两次被profile的更新分别出现约11.5万/16.9万次kernel启动，以及3584/4864次unique操作。这是初始实现的负面效率证据，不是新结构加速的证明。

实现优化版本将无query-global的局部窗口批量送入一次SDPA，避免逐tile unique与高级索引；摘要池化按block向量化；固定tokenizer词表大小与特殊token集合在collator初始化时缓存。参数名及模型边定义保持不变，但算子归并可能带来浮点舍入差异，必须重新验证输出和梯度。当前query-global路径仍使用旧的逐tile参考实现，不宣称其性能已经优化。

先运行CUDA测试，新增短序列、非整tile长度与摘要梯度对照；全部通过后使用新目录，保留旧实验：

```bash
pytest -q tests/test_cuda_contracts.py --tb=short -x
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/long_check.py --action train --output data/wiki-long-check-v2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/long_check.py --action profile --variant B0 --output data/wiki-long-check-v2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/long_check.py --action profile --variant G2 --output data/wiki-long-check-v2
```

新配置打开profile_pipeline，按阶段同步并记录collate、transfer、model前反向、梯度裁剪及optimizer耗时。同步本身会引入开销，此轮用于解释瓶颈；正式速度对比需两组采用一致计时设置，并另测关闭阶段同步的吞吐。新汇总的mean_phase_seconds_after_step5可以检查完整训练与预制batch profiler之间的差距是否来自数据处理，不预设原因已经证实。

本地仅做静态检查；本轮CUDA测试和性能改善需由服务器验证。勿把profiler表中父级范围的百分比相加，或将超过100%的嵌套范围统计解释为物理GPU利用率。

## 9. 实现开销解决后的第一轮质量筛选

服务器第二轮真实长文汇总：B0/G2均完成20步与327680 tokens，记录训练时间约10.73/10.75秒；旧实现约445.6/470.6秒。两组共同提速，不能算作G2相对B0的架构优势。两组显存仍约5.66GiB，G2的MLM loss仍更高。该8篇重复语料的短跑只用于验证工程可行性。

下一步停止反复profile，用B0/L3/G2单seed筛选：每语言10M tokens候选池、按文档hash 90/10切分，要求每语言开发文档至少100篇；保留相同模型/数据SHA，每组从共同预训练底座新建LoRA，不继续8篇短跑的权重。新增参数改在共同head/LoRA之后初始化，避免候选分支消耗随机数导致共同参数的初始化不同；CUDA测试加入相同seed下共同参数一致检查。旧检查点的参数名和加载方式不变。

从Mac上传stage1更新包并在仓库根目录解压后，执行：

```bash
pytest -q tests/test_cuda_contracts.py --tb=short -x
export HF_HOME=/root/autodl-tmp/hf-cache
source /etc/network_turbo
python scripts/stage1.py --action download
python scripts/stage1.py --action prepare
mkdir -p logs
HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nohup python -u scripts/stage1.py --action train > logs/stage1-s17.log 2>&1 < /dev/null &
tail -f logs/stage1-s17.log
```

数据下载需要联网，不能给download设置离线标志。默认reference读取旧的data/wiki-starter/manifest.json，复用原模型与Wikipedia revision；新语料路径data/wiki-phase1，不能覆盖原冻结数据。prepare只生成配置与中英文开发集文件，不会启动训练。下载按每语言10M原始token计数，实际训练集约90%，以manifest为准；数据只做精确去重，不能称为已经排除所有基准污染。

三组依次训练，各1000个optimizer steps、累积8次，长度课程一致；这轮是同数据曝光/同训练步数的单seed对照，**不是严格等校准FLOPs**。关闭profile_pipeline的阶段同步，再收集实际时间与tokens。输出runs/stage1-s17/{B0,L3,G2}，每组保留最近两个step检查点与final，底座权重共享缓存。

长文短跑的累积2次约0.48秒/step，线性外推累积8次、全长8K的1000步约32分钟/组，只是预算参考；混合长度、数据分布、GPU负载与检查点I/O会改变它。先从日志前50步确认速度，再决定是否继续。中断时用相同原参数加--resume；脚本会备份日志、对齐至最后完整检查点并恢复，已完成的组不会重复训练。

训练全部完成后：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/stage1.py --action evaluate
python scripts/stage1.py --action summary
```

输出outputs/stage1-s17/summary.json，包含逐语言MLM loss与训练日志末条统计。MLM只用于结构适配诊断，不能用它决定最终模型胜负；之后必须进行同预算分类、抽取、检索/重排及长程证据应用验证，再对有潜力的变化做三seed确认。尤其不能因为单seed MLM loss更低就保留G2，或仅因更高就宣称所有用途均失败。
