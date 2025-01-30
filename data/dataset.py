
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
        labels = inputs.clone()
        # Create a mask array for 15% of the token positions
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random word
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

