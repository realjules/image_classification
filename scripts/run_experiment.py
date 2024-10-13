import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os

from config import config, DEVICE
from dataset import AlbumentationsDataset, ImagePairDataset, TestImagePairDataset, train_transforms, val_transforms
from model import CustomResNet50
from train import train_epoch
from validate import valid_epoch_cls, valid_epoch_ver
from test import test_epoch_ver, generate_submission
from utils import save_model, load_model

def main():
    # Initialize wandb
    wandb.init(project="project-name", config=config)

    # Create model
    model = CustomResNet50(num_classes=config['num_classes']).to(DEVICE)

    # Create datasets and dataloaders
    train_dataset = AlbumentationsDataset(os.path.join(config['data_dir'], 'train'), transform=train_transforms)
    val_dataset = AlbumentationsDataset(os.path.join(config['data_dir'], 'dev'), transform=val_transforms)
    pair_dataset = ImagePairDataset(config['data_ver_dir'], csv_file='val_pairs.csv', transform=val_transforms)
    test_pair_dataset = TestImagePairDataset(config['data_ver_dir'], csv_file='test_pairs.csv', transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    pair_dataloader = DataLoader(pair_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    test_pair_dataloader = DataLoader(test_pair_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    # Initialize mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    best_valid_cls_acc = 0.0
    best_valid_ret_acc = 0.0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        # Train
        train_cls_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE, criterion)
        print(f"Train Cls. Acc {train_cls_acc:.04f}%\t Train Cls. Loss {train_loss:.04f}")

        # Validate
        valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, DEVICE, criterion)
        print(f"Val Cls. Acc {valid_cls_acc:.04f}%\t Val Cls. Loss {valid_loss:.04f}")

        valid_ret_acc = valid_epoch_ver(model, pair_dataloader, DEVICE)
        print(f"Val Ret. Acc {valid_ret_acc:.04f}%")

        # Log metrics
        wandb.log({
            'train_cls_acc': train_cls_acc,
            'train_loss': train_loss,
            'valid_cls_acc': valid_cls_acc,
            'valid_loss': valid_loss,
            'valid_ret_acc': valid_ret_acc
        })

        # Save model
        save_model(model, optimizer, scheduler, 
                   {'cls_acc': valid_cls_acc, 'ret_acc': valid_ret_acc}, 
                   epoch, os.path.join(config['checkpoint_dir'], 'last.pth'))

        # Save best models
        if valid_cls_acc > best_valid_cls_acc:
            best_valid_cls_acc = valid_cls_acc
            save_model(model, optimizer, scheduler, 
                       {'cls_acc': valid_cls_acc, 'ret_acc': valid_ret_acc}, 
                       epoch, os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
            wandb.save(os.path.join(config['checkpoint_dir'], 'best_cls.pth'))

        if valid_ret_acc > best_valid_ret_acc:
            best_valid_ret_acc = valid_ret_acc
            save_model(model, optimizer, scheduler, 
                       {'cls_acc': valid_cls_acc, 'ret_acc': valid_ret_acc}, 
                       epoch, os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
            wandb.save(os.path.join(config['checkpoint_dir'], 'best_ret.pth'))

    # Test
    model, _, _, _, _ = load_model(model, path=os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
    scores = test_epoch_ver(model, test_pair_dataloader, DEVICE)
    generate_submission(scores, "verification_submission.csv")

if __name__ == "__main__":
    main()