import json
import torch
import faiss
from tqdm import tqdm
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModel

class TextDeduplicatorWithFAISS:
    """
    使用 FAISS 索引实现的文本去重类（基于余弦相似度）。
    """
    def __init__(self, model_name: str = 'bert-base-chinese', device: str = None) -> None:
        """
        初始化文本去重类。
        :param model_name: 使用的预训练模型名称。
        :param device: 指定运行设备（'cpu' 或 'cuda'），默认为自动检测。
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        
        # 初始化 FAISS 索引
        self.embedding_dim = self.model.config.hidden_size
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        # 如果使用GPU
        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
        self.index_ids = []

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        计算文本的嵌入表示，并进行归一化。
        :param texts: 要计算嵌入的一组文本列表。
        :return: 归一化后的文本嵌入张量，形状为 (batch_size, hidden_size)。
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用 [CLS] token 的 embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # L2 归一化 (等价于计算余弦相似度)
        faiss.normalize_L2(embeddings.cpu().numpy())
        return embeddings.cpu()

    def is_duplicate(self, embedding: torch.Tensor, threshold: float) -> bool:
        """
        检查一个嵌入是否与 FAISS 索引中的嵌入重复。
        :param embedding: 待检查的嵌入向量。
        :param threshold: 相似度的阈值。
        :return: 是否为重复项（True / False）。
        """
        if self.index.ntotal == 0:
            return False

        embedding_np = embedding.numpy()
        # 查找最近的 1 个向量
        distances, _ = self.index.search(embedding_np, k=1)
        
        # 检查最近向量的相似度是否高于阈值
        max_similarity = distances[0][0]
        return max_similarity >= threshold

    def add_to_index(self, embedding: torch.Tensor, doc_id: int) -> None:
        """
        将新的嵌入添加到 FAISS 索引中。
        :param embedding: 要添加的嵌入向量。
        :param doc_id: 该嵌入对应的文档 ID。
        """
        self.index.add(embedding.numpy())
        self.index_ids.append(doc_id)

    def process_and_save(self, input_path: str, output_path: str, text_key: str, threshold: float) -> None:
        """
        处理输入文件，去除相似文本并保存到输出文件。
        :param input_path: 输入 JSONL 文件路径。
        :param output_path: 输出 JSONL 文件路径。
        :param text_key: JSON中包含文本的键。
        :param threshold: 去重的相似度阈值。
        """
        doc_id = 0
        unique_count = 0
        total_count = 0

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in tqdm(infile, desc="Processing lines"):
                total_count += 1
                try:
                    item: Dict[str, Union[str, int, float]] = json.loads(line)
                    text_to_check: str = item.get(text_key)
                    if not text_to_check or not isinstance(text_to_check, str):
                        continue
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                    continue

                current_embedding = self.get_embeddings([text_to_check])

                if not self.is_duplicate(current_embedding, threshold):
                    outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                    self.add_to_index(current_embedding, doc_id)
                    unique_count += 1
                
                doc_id += 1
        
        print("\n--- Deduplication Complete ---")
        print(f"Total lines processed: {total_count}")
        print(f"Unique lines kept: {unique_count}")
        print(f"Duplicate lines removed: {total_count - unique_count}")
        print(f"Results saved to: {output_path}")
