
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import math
from typing import Tuple, Dict

class BertPreTrainer:
    """BERT Pre-training class implementing MLM and NSP tasks"""
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config['weight_decay']
        )
        
        # Initialize learning rate scheduler
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=config['num_epochs'] * len(train_dataloader)
        )
        
        # Initialize tensorboard writer
        self.writer = TensorboardWriter(config['log_dir'])
        
        # Initialize loss function
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # -100 is padding token
        self.nsp_loss_fn = nn.CrossEntropyLoss()
        
        # Initialize metrics
        self.mlm_meter = AverageMeter('MLM_Loss')
        self.nsp_meter = AverageMeter('NSP_Loss')
        self.mlm_acc_meter = AverageMeter('MLM_Accuracy')
        self.nsp_acc_meter = AverageMeter('NSP_Accuracy')

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        self.mlm_meter.reset()
        self.nsp_meter.reset()
        self.mlm_acc_meter.reset()
        self.nsp_acc_meter.reset()
        
        pbar = tqdm(self.train_dataloader, desc=f'Epoch {epoch}')
        for step, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            mlm_labels = batch['mlm_labels'].to(self.device)
            nsp_labels = batch['nsp_labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=mlm_labels,
                next_sentence_label=nsp_labels
            )
            
            mlm_logits = outputs.logits
            nsp_logits = outputs.seq_relationship_logits
            
            # Calculate losses
            mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, self.config['vocab_size']), 
                                       mlm_labels.view(-1))
            nsp_loss = self.nsp_loss_fn(nsp_logits, nsp_labels)
            
            # Combined loss
            loss = mlm_loss + nsp_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Calculate accuracies
            mlm_accuracy = self._compute_mlm_accuracy(mlm_logits, mlm_labels)
            nsp_accuracy = compute_accuracy(nsp_logits, nsp_labels)
            
            # Update meters
            self.mlm_meter.update(mlm_loss.item())
            self.nsp_meter.update(nsp_loss.item())
            self.mlm_acc_meter.update(mlm_accuracy)
            self.nsp_acc_meter.update(nsp_accuracy)
            
            # Update progress bar
            pbar.set_postfix({
                'MLM_Loss': f'{self.mlm_meter.avg:.4f}',
                'NSP_Loss': f'{self.nsp_meter.avg:.4f}',
                'MLM_Acc': f'{self.mlm_acc_meter.avg:.4f}',
                'NSP_Acc': f'{self.nsp_acc_meter.avg:.4f}',
                'LR': f'{get_lr(self.optimizer):.2e}'
            })
            
            # Log to tensorboard
            if step % self.config['log_step'] == 0:
                global_step = epoch * len(self.train_dataloader) + step
                self.writer.log_scalars('Training', {
                    'MLM_Loss': self.mlm_meter.avg,
                    'NSP_Loss': self.nsp_meter.avg,
                    'MLM_Accuracy': self.mlm_acc_meter.avg,
                    'NSP_Accuracy': self.nsp_acc_meter.avg,
                    'Learning_Rate': get_lr(self.optimizer)
                }, global_step)
        
        return self.mlm_meter.avg, self.nsp_meter.avg

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        self.mlm_meter.reset()
        self.nsp_meter.reset()
        self.mlm_acc_meter.reset()
        self.nsp_acc_meter.reset()
        
        pbar = tqdm(self.val_dataloader, desc='Validation')
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            mlm_labels = batch['mlm_labels'].to(self.device)
            nsp_labels = batch['nsp_labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=mlm_labels,
                next_sentence_label=nsp_labels
            )
            
            mlm_logits = outputs.logits
            nsp_logits = outputs.seq_relationship_logits
            
            # Calculate losses
            mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, self.config['vocab_size']), 
                                       mlm_labels.view(-1))
            nsp_loss = self.nsp_loss_fn(nsp_logits, nsp_labels)
            
            # Calculate accuracies
            mlm_accuracy = self._compute_mlm_accuracy(mlm_logits, mlm_labels)
            nsp_accuracy = compute_accuracy(nsp_logits, nsp_labels)
            
            # Update meters
            self.mlm_meter.update(mlm_loss.item())
            self.nsp_meter.update(nsp_loss.item())
            self.mlm_acc_meter.update(mlm_accuracy)
            self.nsp_acc_meter.update(nsp_accuracy)
            
            # Update progress bar
            pbar.set_postfix({
                'MLM_Loss': f'{self.mlm_meter.avg:.4f}',
                'NSP_Loss': f'{self.nsp_meter.avg:.4f}',
                'MLM_Acc': f'{self.mlm_acc_meter.avg:.4f}',
                'NSP_Acc': f'{self.nsp_acc_meter.avg:.4f}'
            })
        
        # Log validation metrics
        self.writer.log_scalars('Validation', {
            'MLM_Loss': self.mlm_meter.avg,
            'NSP_Loss': self.nsp_meter.avg,
            'MLM_Accuracy': self.mlm_acc_meter.avg,
            'NSP_Accuracy': self.nsp_acc_meter.avg
        }, epoch)
        
        return self.mlm_meter.avg, self.nsp_meter.avg

    def _compute_mlm_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute MLM accuracy excluding padding tokens"""
        predictions = torch.argmax(logits, dim=-1)
        mask = (labels != -100)  # Exclude padding tokens
        correct = (predictions == labels) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        return accuracy.item()

    def train(self):
        """Main training loop"""
        logging.info("Starting training...")
        best_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_mlm_loss, train_nsp_loss = self.train_epoch(epoch)
            logging.info(f'Epoch {epoch}: Train MLM Loss = {train_mlm_loss:.4f}, '
                        f'Train NSP Loss = {train_nsp_loss:.4f}')
            
            # Validation phase
            val_mlm_loss, val_nsp_loss = self.validate(epoch)
            val_loss = val_mlm_loss + val_nsp_loss
            logging.info(f'Epoch {epoch}: Val MLM Loss = {val_mlm_loss:.4f}, '
                        f'Val NSP Loss = {val_nsp_loss:.4f}')
            
            # Save checkpoint
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_loss': best_loss,
                'config': self.config
            }, is_best, self.config['checkpoint_dir'])
        
        self.writer.close()
        logging.info("Training finished!")

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load configuration
    with open('config/bert_config.json', 'r') as f:
        config = json.load(f)
    
    # Setup logging
    setup_logging(config['save_dir'])
    
    # Initialize model, dataloaders, and trainer
    model = BertForPreTraining(config)  # You need to implement this
    train_dataloader = get_train_dataloader(config)  # You need to implement this
    val_dataloader = get_val_dataloader(config)  # You need to implement this
    
    trainer = BertPreTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()

