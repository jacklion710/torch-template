import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
import yaml
from typing import Dict, Any

from models import create_model_instance
from dataloader import CustomDataset

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Generic trainer initialization
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._setup_data()
        self._setup_models()
        self._setup_optimizers()
        self._setup_criterions()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Initialize metrics tracking
        self.metrics = {}
    
    def _setup_logging(self):
        """Configure logging and experiment tracking"""
        # File logging
        log_dir = Path(self.config['training']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
        # WandB initialization (optional)
        if self.config['logging'].get('use_wandb', False):
            wandb.init(
                project=self.config['logging']['project_name'],
                config=self.config
            )
    
    def _setup_data(self):
        """Initialize datasets and dataloaders"""
        # Create datasets
        self.train_dataset = CustomDataset(
            self.config['data']['train_dir'],
            **self.config['data'].get('dataset_params', {})
        )
        
        self.val_dataset = CustomDataset(
            self.config['data']['val_dir'],
            **self.config['data'].get('dataset_params', {})
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            **self.config['data'].get('train_dataloader_params', {})
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            **self.config['data'].get('val_dataloader_params', {})
        )
    
    def _setup_models(self):
        """Initialize models"""
        self.models = {}
        for model_name, model_config in self.config['models'].items():
            self.models[model_name] = create_model_instance(
                model_config['type'],
                model_config['params']
            ).to(self.device)
    
    def _setup_optimizers(self):
        """Initialize optimizers"""
        self.optimizers = {}
        for model_name, model in self.models.items():
            optimizer_config = self.config['training']['optimizers'][model_name]
            optimizer_class = getattr(torch.optim, optimizer_config['type'])
            self.optimizers[model_name] = optimizer_class(
                model.parameters(),
                **optimizer_config['params']
            )
    
    def _setup_criterions(self):
        """Initialize loss functions"""
        self.criterions = {}
        for loss_name, loss_config in self.config['training']['losses'].items():
            loss_class = getattr(nn, loss_config['type'])
            self.criterions[loss_name] = loss_class(**loss_config.get('params', {}))
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Implement single training step logic
        Args:
            batch: Dictionary containing the current batch data
        Returns:
            Dictionary containing computed metrics
        """
        pass
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Implement single validation step logic
        Args:
            batch: Dictionary containing the current batch data
        Returns:
            Dictionary containing computed metrics
        """
        pass
    
    def train_epoch(self):
        """Run one epoch of training"""
        for model in self.models.values():
            model.train()
        
        epoch_metrics = {}
        pbar = tqdm(self.train_loader, desc=f'Training epoch {self.current_epoch}')
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Update progress bar
            pbar.set_postfix({k: f'{v:.4f}' for k, v in step_metrics.items()})
            
            # Log metrics
            if self.config['logging'].get('use_wandb', False):
                wandb.log(step_metrics, step=self.global_step)
            
            # Update global step
            self.global_step += 1
            
            # Accumulate epoch metrics
            for k, v in step_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
        
        # Compute epoch averages
        epoch_metrics = {k: v / len(self.train_loader) for k, v in epoch_metrics.items()}
        return epoch_metrics
    
    def validate(self):
        """Run validation"""
        for model in self.models.values():
            model.eval()
        
        val_metrics = {}
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Validation step
                step_metrics = self.validation_step(batch)
                
                # Accumulate metrics
                for k, v in step_metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v
        
        # Compute validation averages
        val_metrics = {f'val_{k}': v / len(self.val_loader) 
                      for k, v in val_metrics.items()}
        return val_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'metrics': metrics,
            'models': {name: model.state_dict() 
                      for name, model in self.models.items()},
            'optimizers': {name: opt.state_dict() 
                          for name, opt in self.optimizers.items()}
        }
        
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        for name, state_dict in checkpoint['models'].items():
            self.models[name].load_state_dict(state_dict)
        
        for name, state_dict in checkpoint['optimizers'].items():
            self.optimizers[name].load_state_dict(state_dict)
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            logging.info(f"\nEpoch {epoch}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            logging.info(f"Training metrics: {train_metrics}")
            
            # Validation
            val_metrics = self.validate()
            logging.info(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint
            if epoch % self.config['training']['checkpoint_frequency'] == 0:
                self.save_checkpoint({**train_metrics, **val_metrics})

if __name__ == "__main__":
    # Base configuration - can be used as fallback or default values
    with open('./config/config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # WandB Sweep configuration
    use_wandb_sweep = base_config.get('wandb', {}).get('use_sweep', False)
    
    if use_wandb_sweep:
        # Load sweep configuration
        with open('./config/sweep_config.yaml', 'r') as f:
            sweep_config = yaml.safe_load(f)
            
        # Initialize wandb with sweep configuration
        wandb.init(
            project=base_config['wandb']['project_name'],
            config=sweep_config
        )
        
        # Merge wandb config with base config
        # WandB sweep params will override base config values
        config = base_config.copy()
        
        # Update nested dictionaries with sweep parameters
        for key, value in wandb.config.items():
            # Handle nested keys like "model.learning_rate"
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            
    else:
        # Use base config as is
        config = base_config
    
    # Initialize trainer with the final config
    trainer = Trainer(config)
    
    # Load checkpoint if specified
    if config['training'].get('resume_from'):
        trainer.load_checkpoint(config['training']['resume_from'])
    
    # Start training
    trainer.train()
