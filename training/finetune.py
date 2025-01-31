
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..model.bert import BertModel
from ..data.dataset import BertFineTuneDataset
from ..config.bert_config import BertConfig
from ..utils.utils import set_seed, setup_logging

class BERTFineTuner:
    """BERT模型微调类
    
    用于对预训练好的BERT模型进行下游任务的微调训练
    支持分类任务的微调，可扩展支持其他任务类型
    """
    
    def __init__(
        self,
        model: BertModel,
        config: BertConfig,
        train_dataset: BertFineTuneDataset,
        val_dataset: Optional[BertFineTuneDataset] = None,
        num_labels: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.config = config
        self.device = device
        self.num_labels = num_labels
        
        # 初始化模型
        self.model = model
        # 添加分类头
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.model.to(device)
        self.classifier.to(device)
        
        # 准备数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        
        if val_dataset:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers
            )
        else:
            self.val_dataloader = None
            
        # 设置优化器
        # 对不同层使用不同的学习率
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            },
            {
                'params': self.classifier.parameters(),
                'weight_decay': config.weight_decay
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            eps=config.adam_epsilon
        )
        
        # 设置学习率调度器
        total_steps = len(self.train_dataloader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_steps,
            last_epoch=-1
        )
        
        self.loss_fn = CrossEntropyLoss()
        
    def train(self) -> Dict[str, List[float]]:
        """执行完整的训练循环"""
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            logging.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # 训练阶段
            train_loss, train_acc = self._train_epoch()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # 验证阶段
            if self.val_dataloader:
                val_loss, val_acc = self._validate()
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
                    
                logging.info(
                    f"Epoch {epoch+1} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch+1} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}"
                )
                
        return history
    
    def _train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        self.classifier.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # 准备数据
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 获取[CLS]标记的输出
            pooled_output = outputs[:, 0, :]
            logits = self.classifier(pooled_output)
            
            # 计算损失
            loss = self.loss_fn(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/labels.size(0):.4f}"
            })
            
        avg_loss = total_loss / len(self.train_dataloader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """验证模型性能"""
        self.model.eval()
        self.classifier.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_dataloader, desc="Validating"):
            # 准备数据
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            pooled_output = outputs[:, 0, :]
            logits = self.classifier(pooled_output)
            
            # 计算损失
            loss = self.loss_fn(logits, labels)
            
            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            
        avg_loss = total_loss / len(self.val_dataloader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _save_checkpoint(self, filename: str):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        save_path = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, save_path)
        logging.info(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logging.info(f"Loaded checkpoint from {checkpoint_path}")

def main():
    """主函数，用于演示如何使用BERTFineTuner类"""
    # 设置随机种子
    set_seed(42)
    
    # 设置日志
    setup_logging()
    
    # 加载配置
    config = BertConfig()
    
    # 加载数据集
    train_dataset = BertFineTuneDataset(...)  # 需要实现
    val_dataset = BertFineTuneDataset(...)    # 需要实现
    
    # 初始化模型
    model = BertModel(config)  # 需要实现
    
    # 初始化训练器
    trainer = BERTFineTuner(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_labels=2  # 二分类任务
    )
    
    # 开始训练
    history = trainer.train()
    
    # 可以根据需要处理训练历史记录
    
if __name__ == "__main__":
    main()

