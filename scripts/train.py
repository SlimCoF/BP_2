import torch
from tqdm import tqdm
from auxiliary_functions import perform_validation
from auxiliary_functions import check_accuracy
import wandb

VAL_RATIO = 20 # after 20% perform validation

def train(loader, validation_loader, model, optimizer, loss_fn, scaler, DEVICE, epoch):
    loop = tqdm(loader)
    val = int(len(loop)/4);

    # data_valid, targets_valid = next(iter(validation_loader))
    # data_valid = data_valid.to(device=DEVICE)
    validation_number = 1;

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss_train = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss_train).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss_train.item())

        wandb.log({
            "Train Loss": loss_train,
        })

        if batch_idx%val == 0:
            
            # calculate train data acc and dice
            acc_train , dice_train = check_accuracy(loader, model, DEVICE) 

            # calculate validation data loss
            # with torch.cuda.amp.autocast():
            #     predictions_valid = model(data_valid)
            #     targets_valid = targets_valid.float().unsqueeze(1).to(device=DEVICE)
            #     loss_valid = loss_fn(predictions_valid, targets_valid)

            #calculate validation data acc and dice
            acc_valid, dice_valid = perform_validation(model, optimizer, validation_loader, DEVICE)

            wandb.log({
                "Epoch": epoch,
                "Val": validation_number,

                "Train Loss": loss_train,
                "Train Acc": acc_train,
                "Train Dice": dice_train,

                # "Valid Loss": loss_valid,
                "Valid Acc": acc_valid,
                "Valid Dice": dice_valid
            })
            validation_number += 1
    
