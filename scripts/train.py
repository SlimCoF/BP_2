import torch
from tqdm import tqdm
import wandb
import numpy as np

from scripts.auxiliary_functions import calculate_metrics

import time

def train(loader, validation_loader, model, optimizer, loss_fn, scaler, DEVICE, epoch):
    loop = tqdm(loader)
    val = int(len(loop)/5);

    validation_number = 1;
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            # CrossEntropyLoss - Requirements:
            #   model output shape: [batch_size, class_n, height, width]
            #   target shape: [batch_size, height, width] - elements [0 - (class_n - 1)]
            # (Currently in use) DiceLoss - Requirements:
            #   model outpus shape: [batch_size, class_n, height, width]
            #   target shape: [batch_size, height, width] or [batch_size, class_n, height, width]
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
            # START TIME !!
            start = time.time()
            acc_train, dice_train  = calculate_metrics(loader, model, DEVICE) 
            # END TIME !!
            end = time.time()
            print("\nACC/DICE TRAIN calculation time: ")
            print(end - start)

            #calculate validation data acc and dice
            # START TIME !!
            start = time.time()
            acc_valid, dice_valid = calculate_metrics(validation_loader, model, DEVICE)
            # END TIME !!
            end = time.time()
            print("\ACC/DICE VALIDATION calculation time: ")
            print(end - start)

            wandb.log({
                "Epoch": epoch,
                "Val": validation_number,

                "Train Acc": acc_train,
                "Train Dice": dice_train,

                # "Valid Loss": loss_valid,
                "Valid Acc": acc_valid,
                "Valid Dice": dice_valid
            })
            validation_number += 1
    
