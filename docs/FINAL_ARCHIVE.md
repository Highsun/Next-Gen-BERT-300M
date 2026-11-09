# 释放 AutoDL 前的最终归档

本流程只归档已经产生的数据，不新增实验、不加载模型、不执行推理。打包、校验和以后重绘现有结果图均不需要 GPU。不要把关闭浏览器当作服务器关机，也不要在本地副本检查完成前释放实例。

## 保留范围

`scripts/final_archive.py` 生成一个完整归档：

- `project/`：当前源代码、配置、文档、已有 data/outputs/runs/logs/raw 内容；包括所有已有训练曲线原始点、逐篇预测、冻结诊断输入、主实验与早期试跑的权重/配置/分词器。
- `hf-cache/hub/models--jhu-clsp--mmBERT-base/`：这一模型的完整现有缓存，包括固定 SHA 的底座权重。当前 LoRA checkpoint 只有增量参数，单独保留它并不包含完整模型。
- `ENVIRONMENT.json`：Python、包版本、Git HEAD/状态与只读 GPU/驱动信息。不收集完整环境变量或 Git remote，避免包含凭据。
- `ARCHIVE-MANIFEST.json`：逐文件 SHA-256、大小、缓存符号链接目标、必需文件与可选历史文件清单。

排除 `.venv`、Git 历史、Python/pytest 缓存、Jupyter 检查点、常见密钥/认证文件和未完成下载。它不是操作系统/容器镜像；未来运行仍需要重装依赖及兼容 CUDA 环境。项目之外自行保存的文件不会被自动包含。

脚本拒绝覆盖已有备份，必需文件缺失会停止。早期 pilot/profile 文件缺失只记录缺失清单，不因此重新训练。归档结束后检查源文件大小、mtime、inode 未变，并重新读取压缩包核验所有成员的内容哈希。不要在打包期间训练、修改或删除项目文件。

## 服务器操作

上传更新包 `autodl-final-archive-v1.zip` 到项目根目录后，在 AutoDL 终端执行：

```bash
cd /root/autodl-tmp/Next-Gen-BERT-300M
source .venv/bin/activate
export HF_HOME=/root/autodl-tmp/hf-cache
python -m zipfile -e autodl-final-archive-v1.zip .

nohup python -u scripts/final_archive.py \
  > /root/autodl-tmp/final-archive.log 2>&1 < /dev/null &

tail -n 30 -f /root/autodl-tmp/final-archive.log
```

默认输出目录为 `/root/autodl-tmp/nextbert-final-backup/`。它在项目之外，归档日志也在项目之外，不会出现归档自己或不断变化的打包日志的情况。若已存在半成品目录，保留它，使用新的 `--output /root/autodl-tmp/nextbert-final-backup-v2` 重试；不要忽略报错而释放实例。

看到以下两项均出现后，表示服务器端备份和校验完成：

```text
ARCHIVE VERIFIED: ... all SHA-256 checks passed.
BACKUP COMPLETE: .../nextbert-final-backup.tar.gz
```

最终还会打印大小与压缩包 SHA-256。根据实际文件数和大小，打包与逐字节校验可能需要一定时间，不需占用 GPU；脚本本地只做静态检查，首次归档执行在服务器完成。

## 下载与本地复核

从 JupyterLab 下载以下两个文件到 Mac（不需要在服务器生成图片）：

```text
/root/autodl-tmp/nextbert-final-backup/nextbert-final-backup.tar.gz
/root/autodl-tmp/nextbert-final-backup/SHA256SUMS.txt
```

将两个本地文件的路径提供给助手。在允许释放实例前，检查本地压缩包整体 SHA-256 与 `SHA256SUMS.txt` 一致，重新读取并核验嵌入清单、成员哈希、链接以及关键主实验/诊断文件。检查不得导入模型或反序列化 `.pt`。

用户也可以先在 Mac 终端自行检查传输是否完整（两个文件同在 Downloads 时）：

```bash
cd /Users/highsun/Downloads
shasum -a 256 -c SHA256SUMS.txt
```

`OK` 只说明传输哈希一致；文件清单和逐文件核查通过后，再关机并释放实例。下载成功并不自动保证内容齐全。

## 后续可画与不可补造的图

|图或表|所需原始数据|
|---|---|
|训练 loss—step、loss—累计 token/时间|三组完整 train.jsonl|
|累计时间、阶段耗时/吞吐与显存比较|日志累计计时、tokens、max_memory_bytes；时间差中保留 checkpoint I/O 的解释|
|中英验证损失、逐篇差值及区间|验证集逐篇 predictions.jsonl 和冻结数据|
|实际长度/语言分布及长度分桶质量|诊断 cases.jsonl、manifest、逐案例 predictions.jsonl|
|远端干预配对图与置信区间|固定目标的三种干预条件逐案例记录|
|G2 各层 gate 激活与阈值比例|G2-metrics.json；只有聚合量，不能补出完整激活分布|
|早期实现与批量实现的已记录开销比较|各阶段原始日志/summary 和已有 operator 报告；缺失就标记缺失|

没有记录的每秒 GPU 利用率、功耗、温度、完整注意力矩阵和逐 token gate 原始激活，不能事后从 loss 推导出来。现有报告可以不包含这些图，也不需要为此重启实验。未来只基于已经保留的观测绘图；未做的分类、抽取、检索和重排成绩继续标记为未验证。
