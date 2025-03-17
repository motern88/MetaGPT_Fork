import asyncio
from typing import List, Tuple, Union

import evaluate
import jieba
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel

from metagpt.const import EXAMPLE_BENCHMARK_PATH
from metagpt.logs import logger
from metagpt.rag.factories import get_rag_embedding
from metagpt.utils.common import read_json_file


class DatasetInfo(BaseModel):
    """数据集信息模型"""
    name: str  # 数据集名称
    document_files: List[str]  # 文档文件列表
    gt_info: List[dict]  # 真实信息（ground truth）列表


class DatasetConfig(BaseModel):
    """数据集配置模型"""
    datasets: List[DatasetInfo]  # 数据集信息列表


class RAGBenchmark:
    """RAG（Retrieval-Augmented Generation）基准测试类"""

    def __init__(
        self,
        embed_model: BaseEmbedding = None,  # 嵌入模型
    ):
        self.evaluator = SemanticSimilarityEvaluator(
            embed_model=embed_model or get_rag_embedding(),  # 语义相似度评估器
        )

    def set_metrics(
        self,
        bleu_avg: float = 0.0,  # BLEU平均分数
        bleu_1: float = 0.0,  # BLEU-1分数
        bleu_2: float = 0.0,  # BLEU-2分数
        bleu_3: float = 0.0,  # BLEU-3分数
        bleu_4: float = 0.0,  # BLEU-4分数
        rouge_l: float = 0.0,  # ROUGE-L分数
        semantic_similarity: float = 0.0,  # 语义相似度分数
        recall: float = 0.0,  # 召回率
        hit_rate: float = 0.0,  # 命中率
        mrr: float = 0.0,  # 平均倒数排名
        length: float = 0.0,  # 生成文本长度
        generated_text: str = None,  # 生成的文本
        ground_truth_text: str = None,  # 真实文本
        question: str = None,  # 问题
    ):
        """设置评估指标"""
        metrics = {
            "bleu-avg": bleu_avg,  # BLEU平均分数
            "bleu-1": bleu_1,  # BLEU-1分数
            "bleu-2": bleu_2,  # BLEU-2分数
            "bleu-3": bleu_3,  # BLEU-3分数
            "bleu-4": bleu_4,  # BLEU-4分数
            "rouge-L": rouge_l,  # ROUGE-L分数
            "semantic similarity": semantic_similarity,  # 语义相似度分数
            "recall": recall,  # 召回率
            "hit_rate": hit_rate,  # 命中率
            "mrr": mrr,  # 平均倒数排名
            "length": length,  # 生成文本长度
        }

        log = {
            "generated_text": generated_text,  # 生成的文本
            "ground_truth_text": ground_truth_text,  # 真实文本
            "question": question,  # 问题
        }

        return {"metrics": metrics, "log": log}  # 返回指标和日志

    def bleu_score(self, response: str, reference: str, with_penalty=False) -> Union[float, Tuple[float]]:
        """计算BLEU分数"""
        f = lambda text: list(jieba.cut(text))  # 使用jieba分词
        bleu = evaluate.load(path="bleu")  # 加载BLEU评估器
        results = bleu.compute(predictions=[response], references=[[reference]], tokenizer=f)  # 计算BLEU分数

        bleu_avg = results["bleu"]  # BLEU平均分数
        bleu1 = results["precisions"][0]  # BLEU-1分数
        bleu2 = results["precisions"][1]  # BLEU-2分数
        bleu3 = results["precisions"][2]  # BLEU-3分数
        bleu4 = results["precisions"][3]  # BLEU-4分数
        brevity_penalty = results["brevity_penalty"]  # 简短惩罚

        if with_penalty:
            return bleu_avg, bleu1, bleu2, bleu3, bleu4  # 返回带惩罚的BLEU分数
        else:
            return 0.0 if brevity_penalty == 0 else bleu_avg / brevity_penalty, bleu1, bleu2, bleu3, bleu4  # 返回不带惩罚的BLEU分数

    def rougel_score(self, response: str, reference: str) -> float:
        """计算ROUGE-L分数"""
        # pip install rouge_score
        f = lambda text: list(jieba.cut(text))  # 使用jieba分词
        rouge = evaluate.load(path="rouge")  # 加载ROUGE评估器

        results = rouge.compute(predictions=[response], references=[[reference]], tokenizer=f, rouge_types=["rougeL"])  # 计算ROUGE-L分数
        score = results["rougeL"]  # 获取ROUGE-L分数
        return score

    def recall(self, nodes: list[NodeWithScore], reference_docs: list[str]) -> float:
        """计算召回率"""
        if nodes:
            total_recall = sum(any(node.text in doc for node in nodes) for doc in reference_docs)  # 计算总召回率
            return total_recall / len(reference_docs)  # 返回召回率
        else:
            return 0.0  # 如果没有节点，返回0

    def hit_rate(self, nodes: list[NodeWithScore], reference_docs: list[str]) -> float:
        """计算命中率"""
        if nodes:
            return 1.0 if any(node.text in doc for doc in reference_docs for node in nodes) else 0.0  # 计算命中率
        else:
            return 0.0  # 如果没有节点，返回0

    def mean_reciprocal_rank(self, nodes: list[NodeWithScore], reference_docs: list[str]) -> float:
        """计算平均倒数排名（MRR）"""
        mrr_sum = 0.0

        for i, node in enumerate(nodes, start=1):  # 遍历节点
            for doc in reference_docs:  # 遍历参考文档
                if text in doc:  # 如果节点文本在文档中
                    mrr_sum += 1.0 / i  # 更新MRR
                    return mrr_sum  # 返回MRR

        return mrr_sum  # 返回MRR

    async def semantic_similarity(self, response: str, reference: str) -> float:
        """计算语义相似度"""
        result = await self.evaluator.aevaluate(
            response=response,
            reference=reference,
        )  # 异步计算语义相似度

        return result.score  # 返回语义相似度分数

    async def compute_metric(
        self,
        response: str = None,  # 生成的文本
        reference: str = None,  # 真实文本
        nodes: list[NodeWithScore] = None,  # 节点列表
        reference_doc: list[str] = None,  # 参考文档列表
        question: str = None,  # 问题
    ):
        """计算所有指标"""
        recall = self.recall(nodes, reference_doc)  # 计算召回率
        bleu_avg, bleu1, bleu2, bleu3, bleu4 = self.bleu_score(response, reference)  # 计算BLEU分数
        rouge_l = self.rougel_score(response, reference)  # 计算ROUGE-L分数
        hit_rate = self.hit_rate(nodes, reference_doc)  # 计算命中率
        mrr = self.mean_reciprocal_rank(nodes, reference_doc)  # 计算平均倒数排名

        similarity = await self.semantic_similarity(response, reference)  # 计算语义相似度

        result = self.set_metrics(
            bleu_avg,
            bleu1,
            bleu2,
            bleu3,
            bleu4,
            rouge_l,
            similarity,
            recall,
            hit_rate,
            mrr,
            len(response),
            response,
            reference,
            question,
        )  # 设置指标

        return result  # 返回结果

    @staticmethod
    def load_dataset(ds_names: list[str] = ["all"]):
        """加载数据集"""
        infos = read_json_file((EXAMPLE_BENCHMARK_PATH / "dataset_info.json").as_posix())  # 读取数据集信息
        dataset_config = DatasetConfig(
            datasets=[
                DatasetInfo(
                    name=name,
                    document_files=[
                        (EXAMPLE_BENCHMARK_PATH / name / file).as_posix() for file in info["document_file"]
                    ],  # 文档文件路径
                    gt_info=read_json_file((EXAMPLE_BENCHMARK_PATH / name / info["gt_file"]).as_posix()),  # 真实信息
                )
                for dataset_info in infos
                for name, info in dataset_info.items()
                if name in ds_names or "all" in ds_names  # 过滤数据集
            ]
        )

        return dataset_config  # 返回数据集配置


if __name__ == "__main__":
    benchmark = RAGBenchmark()  # 初始化RAG基准测试
    answer = "是的，根据提供的信息，2023年7月20日，应急管理部和财政部确实联合发布了《因灾倒塌、损坏住房恢复重建救助工作规范》的通知。这份《规范》旨在进一步规范因灾倒塌、损坏住房的恢复重建救助相关工作。它明确了地方各级政府负责实施救助工作，应急管理部和财政部则负责统筹指导。地方财政应安排足够的资金，中央财政也会提供适当的补助。救助资金将通过专账管理，并采取特定的管理方式。救助对象是那些因自然灾害导致住房倒塌或损坏，并向政府提出申请且符合条件的受灾家庭。相关部门将组织调查统计救助对象信息，并建立档案。此外，《规范》还强调了资金发放的具体方式和公开透明的要求。"
    ground_truth = "“启明行动”是为了防控儿童青少年的近视问题，并发布了《防控儿童青少年近视核心知识十条》。"
    bleu_avg, bleu1, bleu2, bleu3, bleu4 = benchmark.bleu_score(answer, ground_truth)  # 计算BLEU分数
    rougeL_score = benchmark.rougel_score(answer, ground_truth)  # 计算ROUGE-L分数
    similarity = asyncio.run(benchmark.SemanticSimilarity(answer, ground_truth))  # 计算语义相似度

    logger.info(
        f"BLEU Scores: bleu_avg = {bleu_avg}, bleu1 = {bleu1}, bleu2 = {bleu2}, bleu3 = {bleu3}, bleu4 = {bleu4}, "
        f"RougeL Score: {rougeL_score}, "
        f"Semantic Similarity: {similarity}"
    )  # 记录指标