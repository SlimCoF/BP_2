import torch
from tqdm import tqdm
from auxiliary_functions import perform_validation

VAL_RATIO = 20 # after 20% perform validation

def train(loader, validation_loader, model, optimizer, loss_fn, scaler, DEVICE):
    loop = tqdm(loader)
    val = int(len(loop)/4);

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        if batch_idx%val == 0:
            perform_validation(model, optimizer, validation_loader, DEVICE)
    
