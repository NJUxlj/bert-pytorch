
import torch
import torch.nn as nn
from typing import Optional, Tuple

class BertConfig:
    def __init__(self):
        self.vocab_size = 30522  # 词表大小
        self.hidden_size = 768   # 隐藏层维度
        self.num_hidden_layers = 12  # Transformer块的数量
        self.num_attention_heads = 12  # 注意力头数
        self.intermediate_size = 3072  # FFN中间层维度
        self.hidden_dropout_prob = 0.1  # 隐藏层dropout概率
        self.attention_probs_dropout_prob = 0.1  # 注意力dropout概率
        self.max_position_embeddings = 512  # 最大序列长度
        self.type_vocab_size = 2  # segment类型数量
        self.layer_norm_eps = 1e-12  # Layer Norm epsilon值
        self.pad_token_id = 0  # padding token的ID

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embeddings = BertEmbeddings(config)
        
        # Transformer编码器层
        self.encoder = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 初始化模型权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 注意力mask处理
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 获取嵌入
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # 通过Transformer层
        hidden_states = embedding_output
        for layer in self.encoder:
            hidden_states = layer(hidden_states, extended_attention_mask)

        return hidden_states

class BertForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        masked_lm_labels=None,
        next_sentence_label=None,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        prediction_scores, seq_relationship_score = self.cls(sequence_output)
        
        outputs = (prediction_scores, seq_relationship_score)
        
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs
            
        return outputs

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(sequence_output[:, 0])
        return prediction_scores, seq_relationship_score

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

