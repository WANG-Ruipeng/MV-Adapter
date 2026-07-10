# 在 VS Code 中用 Colab 运行 MV-Adapter

这套配置把三类文件明确分开：

- GitHub：代码、配置和 notebook。
- Google Drive：模型 checkpoint、Hugging Face 缓存和生成结果。
- Colab `/content`：每次会话临时克隆的代码和运行环境。

不要把 checkpoint 放进 Git 仓库。仓库根目录的 `.gitignore` 已排除常见权重格式与输出目录。

## 1. Git 远端

本仓库已经配置为：

```text
origin    https://github.com/WANG-Ruipeng/MV-Adapter.git
upstream  https://github.com/huanngzh/MV-Adapter.git
```

日常把自己的修改推到 `origin`：

```bash
git add .
git commit -m "Configure Colab workflow"
git push origin main
```

需要同步原作者更新时：

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

## 2. VS Code 扩展与 Colab 内核

打开仓库后，接受 VS Code 的推荐扩展，或者手动安装：

- `Google Colab`，扩展 ID：`google.colab`
- `Jupyter`，扩展 ID：`ms-toolsai.jupyter`
- `Python`，扩展 ID：`ms-python.python`

打开 `notebooks/mvadapter_colab.ipynb`，点击右上角 `Select Kernel`，选择 `Colab`，再选 `Auto Connect` 或新建指定 GPU 的 Colab Server。首次使用会在浏览器中完成 Google OAuth 登录。

本地 notebook 只负责发送 cell；实际 Python 进程、GPU 和 `/content` 文件系统都在 Colab。因此 notebook 会在远端重新 `git clone` 代码，不能假定 Colab 能直接看到本地仓库文件。

## 3. Drive 目录

首次运行 notebook 后会自动创建：

```text
MyDrive/
├── ModelWeights/MV-Adapter/
│   ├── huggingface/       # SD2.1/SDXL 等 Hugging Face 基础模型缓存
│   └── mv-adapter/        # MV-Adapter 的 .safetensors
└── Colab_Projects/MV-Adapter/
    └── outputs/           # 生成图片
```

`--adapter_path` 必须传 `mv-adapter` 目录，不能传某个 `.safetensors` 文件。项目 loader 会在该目录中按固定文件名寻找权重。

默认 notebook 使用较省显存的 SD2.1：

```text
stabilityai/stable-diffusion-2-1-base
mvadapter_t2mv_sd21.safetensors
```

第一次运行会下载模型，之后复用 Drive 中的缓存。Drive 上直接加载大量小文件会比 Colab 本地盘慢，但不会因运行时回收而丢失。生成结果也直接写入 Drive。

## 4. 运行顺序

从上到下运行 notebook：

1. 挂载 Google Drive。
2. 设置仓库、模型与输出路径。
3. 在 CPU runtime 下载/复用 adapter checkpoint；只下载时可在这里结束。
4. 切换到 GPU runtime 并检查 GPU。
5. 从 GitHub 克隆或更新远端临时代码。
6. 安装 Colab 精简依赖；`nvdiffrast` 会按 NVIDIA 官方要求使用 `--no-build-isolation` 单独安装。
7. 执行一次文生多视图推理并把结果写入 Drive。

从 CPU runtime 切换到 GPU 后，内存状态会重置。请重新运行挂载 Drive 和路径配置两节，再从 GPU 检查继续；已写入 Drive 的 checkpoint 不会丢失。

如果出现 CUDA OOM，先确认使用的是 SD2.1，并把 Colab Server 换成显存更大的 GPU。项目 README 说明图生多视图的最高显存需求约为 14 GB，SDXL 通常也比 SD2.1 更吃显存。

## 5. 私有仓库与密钥

当前 GitHub 仓库是公开地址，notebook 无需 GitHub token。若以后改为私有仓库，不要把 PAT 写入 notebook 或提交到 Git；应使用 Colab Secret 或临时交互式认证。

Hugging Face token 同理。只有遇到 gated/private model 时才在 Colab Secret 中添加 `HF_TOKEN`，不要把 token 硬编码进代码。
