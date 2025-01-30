
from dataclasses import dataclass
from typing import Optional

@dataclass
class BertConfig:
    """BERT模型的配置类
    
    Attributes:
        vocab_size: 词表大小
        hidden_size: 隐藏层维度
        num_hidden_layers: Transformer编码器层数
        num_attention_heads: 注意力头数
        intermediate_size: 前馈网络中间层维度
        hidden_dropout_prob: 隐藏层dropout概率
        attention_probs_dropout_prob: 注意力dropout概率
        max_position_embeddings: 最大位置编码长度
        type_vocab_size: token类型词表大小(通常为2，表示句子A/B)
        initializer_range: 权重初始化范围
        layer_norm_eps: LayerNorm的epsilon值
        pad_token_id: padding token的ID
        position_embedding_type: 位置编码类型
        use_cache: 是否使用past key/values缓存
        classifier_dropout: 分类器dropout概率
    """
    vocab_size: int = 30522  # 原始BERT词表大小
    hidden_size: int = 768   # BERT-base的隐藏层维度
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072  # 通常是hidden_size的4倍
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: Optional[float] = None
    
    def to_dict(self):
        """将配置转换为字典格式"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建配置实例"""
        return cls(**config_dict)

