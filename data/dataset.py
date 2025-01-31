
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import random
from transformers import PreTrainedTokenizer

class BertPreTrainDataset(Dataset):
    """
    Dataset for BERT pre-training, supporting MLM and NSP tasks
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self) -> int:
        return len(self.texts)

    def _get_random_text(self) -> str:
        """Get a random text that's different from the current one"""
        random_idx = random.randint(0, len(self.texts) - 1)
        return self.texts[random_idx]

    def create_mlm_inputs(self, inputs: torch.Tensor) -> tuple:
        """Create inputs and labels for masked language modeling"""
        
        # 创建输入的深拷贝作为标签。这是因为我们需要保留原始token用作训练目标
        labels = inputs.clone()
        
        # Create a mask array for 15% of the token positions
        # 创建一个与输入形状相同的矩阵，填充值为self.mlm_probability（通常是0.15，即15%）。这个矩阵用于决定哪些位置需要被mask。
        probability_matrix = torch.full(labels.shape, self.mlm_probability)


        # 这段代码确保特殊token（如[CLS], [SEP]等）不会被mask。通过将这些位置的概率设为0，确保它们在随后的mask过程中被跳过。
        special_tokens_mask:List[int] = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        
        '''
        上面这行代码的目的是识别序列中的特殊token。让我们逐个分析其组成部分：

            a) self.tokenizer是分词器实例，通常是HuggingFace的tokenizer（如BertTokenizer）。

            b) get_special_tokens_mask()是tokenizer的一个方法，用于标识序列中的特殊token。它返回一个由0和1组成的列表：

            1 表示该位置是特殊token
            0 表示该位置是普通token
            c) labels.tolist()将PyTorch张量转换为Python列表，因为get_special_tokens_mask()期望接收列表作为输入。

            d) already_has_special_tokens=True参数表明输入序列已经包含了特殊token（如[CLS], [SEP]等）。
        
        '''
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # 使用伯努利分布随机采样来决定哪些位置需要被mask。每个位置被选中的概率是15%（由self.mlm_probability指定）。
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        # 使用与运算(&)确保只处理已被选中要mask的token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random word
        # 使用50%的伯努利分布（相对于剩余20%的一半，即10%）
        # ~indices_replaced确保不处理已经被[MASK]替换的token
        # 随机生成词表范围内的token ID进行替换
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest 10% of the time, keep the masked input tokens unchanged
        return inputs, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Select current text and decide whether to use next text or random text for NSP
        current_text = self.texts[idx]
        is_next = random.random() >= 0.5
        second_text = self.texts[idx + 1] if idx < len(self.texts) - 1 and is_next else self._get_random_text()

        # Tokenize and truncate
        tokenized = self.tokenizer(
            current_text,
            second_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        inputs = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        token_type_ids = tokenized['token_type_ids'].squeeze(0)

        # Create MLM inputs and labels
        mlm_inputs, mlm_labels = self.create_mlm_inputs(inputs.clone())
        
        return {
            'input_ids': mlm_inputs,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'mlm_labels': mlm_labels,
            'nsp_labels': torch.tensor(1 if is_next else 0, dtype=torch.long)
        }

class BertFineTuneDataset(Dataset):
    """
    Dataset for BERT fine-tuning on classification tasks
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

