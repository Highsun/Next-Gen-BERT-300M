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
