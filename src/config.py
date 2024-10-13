import torch

config = {
    'batch_size': 32,
    'lr': 0.0058,
    'epochs': 30,
    'data_dir': "/path/to/data/",
    'data_ver_dir': "/path/to/verification/data/",
    'checkpoint_dir': "/path/to/checkpoints/",
    'weight_decay': 1e-2,
    'momentum': 0.9,
    'num_workers': 8,
    'pin_memory': True,
    'scheduler_step_size': 5,
    'scheduler_gamma': 0.1,
    'num_classes': 8631,
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'