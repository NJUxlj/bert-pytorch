# bert-pytorch-reproduce

## Abstract


## Core Innovation
- 使用[MASK] token迫使模型学习上下文语义
- 随机替换帮助模型建立更强的语言理解能力
- 保持一部分token不变可以减少预训练和微调之间的差异
    - 如果只使用[MASK]，模型在微调时会面临**预训练-微调不一致**的问题，因为微调时不会出现[MASK]标记
    - 随机替换帮助模型建立更强大的上下文理解能力
    - 保持一部分原始token不变可以帮助模型更好地保持原始语言信息


## Project Structure
```Plain Text

bert-pytorch/
├── config/
│   └── bert_config.py
├── data/
│   ├── dataset.py
│   └── tokenizer.py
├── model/
│   ├── attention.py
│   ├── embeddings.py
│   ├── encoder.py
│   └── bert.py
├── training/
│   ├── pretrain.py
│   └── finetune.py
├── utils/
│   └── utils.py
└── requirements.txt



```





## Citation
```bibtex
@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

```