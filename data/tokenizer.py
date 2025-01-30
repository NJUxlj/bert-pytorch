
from typing import List, Dict, Optional
import torch
from transformers import PreTrainedTokenizer
from collections import OrderedDict
import regex as re

class BertTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        self.do_lower_case = do_lower_case
        
        # 基本分词器用于处理标点符号和空格
        if do_basic_tokenize:
            never_split = never_split if never_split is not None else []
            never_split = set(never_split).union(self.all_special_tokens)
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case, never_split=never_split
            )
        
        # WordPiece分词器
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """从词汇表文件加载词汇"""
        vocab = OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """将文本转换为token序列"""
        split_tokens = []
        
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
            
        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """将token转换为ID"""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """将ID转换为token"""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """将token序列转换回文本"""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

class BasicTokenizer:
    """处理基本的分词操作，如标点符号分割和空格处理"""
    
    def __init__(self, do_lower_case=True, never_split=None):
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []

    def tokenize(self, text: str) -> List[str]:
        """基本分词处理"""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        
        orig_tokens = text.strip().split()
        split_tokens = []
        
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
            split_tokens.extend(self._run_split_on_punc(token))
            
        return [t for t in split_tokens if t]

    def _clean_text(self, text: str) -> str:
        """清理文本中的无效字符"""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _tokenize_chinese_chars(self, text: str) -> str:
        """在中文字符周围添加空格"""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.extend([" ", char, " "])
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _is_chinese_char(cp: int) -> bool:
        """判断是否为中文字符"""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
        return False

class WordPieceTokenizer:
    """WordPiece分词器"""
    
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> List[str]:
        """使用WordPiece算法进行分词"""
        output_tokens = []
        
        for token in text.strip().split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_substr = None
                
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                    
                if cur_substr is None:
                    is_bad = True
                    break
                    
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
                
        return output_tokens

