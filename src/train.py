import torch
from tqdm import tqdm
from utils import AverageMeter, accuracy

def train_epoch(model, dataloader, optimizer, lr_scheduler, scaler, device, criterion):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        images = images.to(device, non_blocking=True)
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs['out'], labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_m.update(loss.item())
        acc = accuracy(outputs['out'], labels)[0].item()
        acc_m.update(acc)

        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        batch_bar.update()

    if lr_scheduler is not None:
        lr_scheduler.step()

    batch_bar.close()
    return acc_m.avg, loss_m.avg