# 🚀 基于语义相似度的大规模文本去重工具

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-orange.svg)](https://huggingface.co/transformers/)
[![Index](https://img.shields.io/badge/Index-FAISS-red.svg)](https://faiss.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于 **Transformer** 模型（如BERT）和 **FAISS** 索引的高性能文本去重工具，专为处理大规模语料库中的语义重复问题而设计。

---

## 🧐 问题背景

在训练大语言模型或进行NLP数据分析时，数据集中常包含大量语义相同但文本表达不同的“软”重复。传统的基于哈希或编辑距离的方法无法有效识别这类重复，导致模型训练效率低下或评估结果失真。

本工具通过将文本映射到高维向量空间，并利用高效的近似最近邻搜索库 **FAISS**，能够准确、快速地识别并过滤掉语义上重复的文本。

## ✨ 主要特性

- **🧠 语义级去重**：使用 **BERT** 等预训练模型生成文本的嵌入向量，精准捕捉深层语义，有效识别同义句、转述句等。
- **⚡️ 高性能索引**：集成 **FAISS** 库，能够处理百万甚至亿级规模的文本数据，将 $O(n^2)$ 的暴力比较变为高效的近似搜索。
- **🔌 设备自适应**：自动检测并使用可用的 **GPU (CUDA)** 进行加速，无GPU时则回退至CPU。
- **🔧 灵活可控**：支持自定义相似度阈值和选择不同的 **Transformer** 模型。
- **📋 格式兼容**：专为处理常见的 `JSONL` 数据格式而设计。

## 🔬 工作流程

本工具的去重流程分为三个核心步骤：

 1.  **文本嵌入 (Embedding)**
    * 使用指定的 Transformer 模型（如 `bert-base-chinese`）将每一条文本转换为一个固定维度的嵌入向量（例如768维）。
    * 为了使用内积计算等价于余弦相似度，所有嵌入向量在存入索引前都会进行 **L2范数归一化**。

2.  **索引与搜索 (Indexing & Searching)**
    * 初始化一个 **FAISS** 索引（本项目使用 `IndexFlatIP`，即基于内积的精确搜索索引）。
    * 对于每条新文本，将其嵌入向量与索引中已存在的所有向量进行相似度比较（查找最近邻）。
    * FAISS 将此过程极大地加速，避免了逐一对比。

3.  **去重决策 (Deduplication)**
    * 获取与当前文本最相似的向量的相似度分数。
    * 若该分数 **超过** 预设的阈值（如 `0.95`），则判定当前文本为重复项，予以丢弃。
    * 若未超过阈值，则将该文本视为唯一项，保留它，并将其嵌入向量添加到FAISS索引中，供后续文本比对。

---

## 🛠️ 安装指南

1.  **克隆项目**
    ```bash
    git clone [https://github.com/your-username/Semantic-Text-Deduplicator.git](https://github.com/your-username/Semantic-Text-Deduplicator.git)
    cd Semantic-Text-Deduplicator
    ```

2.  **安装依赖**
    建议在虚拟环境中安装。
    ```bash
    pip install -r requirements.txt
    ```
    **注意**: `requirements.txt` 默认安装 **CPU版本** 的 FAISS (`faiss-cpu`)。如果您有支持CUDA的NVIDIA GPU，可以获得巨大性能提升，请先卸载 `faiss-cpu` 并安装GPU版本：
    ```bash
    pip uninstall faiss-cpu
    # 访问 [https://github.com/facebookresearch/faiss/blob/main/INSTALL.md](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) 获取适合您CUDA版本的安装命令
    # 例如: pip install faiss-gpu-cu118
    ```

---

## 📖 使用方法

本工具提供了一个简单的命令行接口来处理 `JSONL` 文件。

### 输入数据格式

输入文件应为 **JSONL** 格式，即每行一个有效的JSON对象。请确保每个JSON对象中包含一个用于提取文本的键（默认为 `output`）。

**示例 (`data/sample_input.jsonl`):**
```jsonl
{"id": 1, "output": "什么是人工智能？人工智能是指让机器具备人类智能的技术。"}
{"id": 2, "output": "人工智能的定义是什么？人工智能是赋予机器类似人类智能的能力。"}
{"id": 3, "output": "今天天气真不错。"}
```

### 命令行用法

```bash
python main.py --input-file <输入文件> --output-file <输出文件> [其他可选参数]
```

**参数说明:**

* `--input-file` (必须): 输入的 `.jsonl` 文件路径。
* `--output-file` (必须): 用于保存去重后唯一文本的 `.jsonl` 文件路径。
* `--text-key` (可选): JSON对象中包含待处理文本的键名。默认为 `output`。
* `--model-name` (可选): Hugging Face上的预训练模型名称。默认为 `bert-base-chinese`。
* `--threshold` (可选): 相似度阈值，介于 0 和 1 之间。值越高，去重越宽松。默认为 `0.95`。
* `--device` (可选): 指定设备 (`cuda` 或 `cpu`)。默认为自动检测。

**示例:**
```bash
# 使用默认参数处理示例文件
python main.py --input-file data/sample_input.jsonl --output-file output_unique.jsonl

# 使用更高的阈值和不同的文本键
python main.py \
    --input-file my_data.jsonl \
    --output-file my_data_unique.jsonl \
    --text-key "content" \
    --threshold 0.98
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。
