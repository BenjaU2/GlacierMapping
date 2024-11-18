import os
import csv
import sys
import click
import traceback
import albumentations as A
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
from tqdm import tqdm
from torch.utils.data import DataLoader
from my_dataset import My_dataloader
#from dataset import CustomDataset
from modules_DeepLabV3 import DeepLabV3Plus
from modules_Unet import Unet
from modules_ResUnet import DeepResUNet

from metrics import calculate_metrics, calculate_metrics2, DICE_BCE_Loss, DICE_BCE_Loss2

import matplotlib.pyplot as plt

# image shape
INPUT = (256, 256)
CLASSES = 1
@click.command()
@click.option("-T", "--data-dir-train", type = str, required = True, help = "Path for data train directory")
@click.option("-V", "--data-dir-val", type = str, required = True, help = "Path for data val directory")
@click.option(
    "-E",
    "--num-epochs",
    type=int,
    default=30,
    help="Number of epochs to train the model for. Default - 25",
)
@click.option(
    "-L",
    "--learning-rate",
    type=float,
    default=1e-4,
    help="Learning Rate for model. Default - 1e-4",
)
@click.option(
    "-B",
    "--batch-size",
    type=int,
    default=4,
    help="Batch size of data for training. Default - 4",
)
@click.option(
    "-S",
    "--early-stop",
    type=bool,
    default=True,
    help="Stop training if val_loss hasn't improved for a certain no. of epochs. Default - True",
)
def main(data_dir_train, 
         data_dir_val,
         num_epochs, 
         learning_rate,
         batch_size,
         early_stop:bool,
         ):
    click.secho(message="🚀 Training...", fg="blue", nl=True)
    os.makedirs("output", exist_ok=True)

    train_transform = T.Compose([T.Resize([INPUT[0], INPUT[1]])])
    val_transform = T.Compose([T.Resize([INPUT[0], INPUT[1]])])

    Training_dataset=My_dataloader(data_dir_train, transform=train_transform)
    Validating_dataset=My_dataloader(data_dir_val, transform=val_transform)
   

    train_dataloader = DataLoader(Training_dataset, batch_size=batch_size,num_workers=8, shuffle=True)
    val_dataloader=DataLoader(Validating_dataset, batch_size=batch_size, num_workers=8, shuffle=True)

    # Defining model
    # DEFINING MODEL

    model = DeepLabV3Plus(num_classes=CLASSES)
    #model = Unet(6,1)
    #model = DeepResUNet(6,1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #criterion = DiceLoss()
    criterion = DICE_BCE_Loss2()
    #weight_decay=1e-6
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6, betas=(0.9, 0.999 ))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.1
    )

    # For Early-Stopping
    patience_epochs = 29
    no_improvement_epochs = 0

    # Logging
    csv_file = os.path.abspath("output/training_logs_UNET3.csv")
    csv_header = [
        "Epoch",
        "Avg Train Loss",
        "Avg Val Loss",
        "Avg mIoU Train",
        "Avg mIoU Val",
        "Avg Pix Acc Train",
        "Avg Pix Acc Val",
        "Avg Dice Coeff Train",
        "Avg Dice Coeff Val",
        "Learning Rate",
    ]

    # For saving best model 
    best_val_loss = float("inf")
    click.echo(
        f"\n{click.style(text=f'Train Size: ', fg='blue')}{Training_dataset.__len__()}\t{click.style(text=f'Test Size: ', fg='blue')}{Validating_dataset.__len__()}\n"
    )
    # Main loop
    # Main loop
    with open(csv_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

        ##### Parametros para actualizar lr ####


        #########################################
        # Crear listas para almacenar las perdidas d entrenamiento y validación en cada epoca
        training_losses, validating_losses = [], []
        training_dices, validating_dices = [], []
        training_ious, validating_ious = [], []
        training_pixes, validating_pixes = [], []

        for epoch in range(num_epochs):
            # TRAINING
            model.train()
            # Inicializar metricas
            train_loss = 0.0
            total_iou_train = 0.0
            total_pixel_accuracy_train = 0.0
            total_dice_coefficient_train = 0.0

            train_dataloader = tqdm(
                train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
            )

            current_lr = optimizer.param_groups[0]["lr"]

            for images, masks in train_dataloader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                # Realizar la retropropagación hacia adelante y el calculo de loss
                outputs = model(images)
                t_loss = criterion(outputs, masks)
                # Realizar la retropropagación y actualización de los parametros
                t_loss.backward()
                optimizer.step()

                train_loss += t_loss.item()

                # Calculating metrics for training
                with torch.no_grad():
                    pred_masks = outputs > 0.5
                    iou_train, dice_coefficient_train, pixel_accuracy_train = calculate_metrics(
                        pred_masks, masks
                    )

                    total_iou_train += iou_train
                    total_dice_coefficient_train += dice_coefficient_train
                    total_pixel_accuracy_train += pixel_accuracy_train

                # Displaying metrics in the progress bar description
                train_dataloader.set_postfix(
                    loss=t_loss.item(),
                    train_iou=iou_train,
                    train_pix_acc=pixel_accuracy_train,
                    train_dice_coef=dice_coefficient_train,
                    lr=current_lr,
                )

            train_loss /= len(train_dataloader)
            avg_iou_train = total_iou_train / len(train_dataloader)
            avg_pixel_accuracy_train = total_pixel_accuracy_train / len(train_dataloader)
            avg_dice_coefficient_train = total_dice_coefficient_train / len(train_dataloader)

            # Guardar las métricas de entrenamiento en las listas
            training_losses.append(train_loss)
            training_dices.append(avg_dice_coefficient_train)
            training_ious.append(avg_iou_train)
            training_pixes.append(avg_pixel_accuracy_train)
            print("Training Losses: ", training_losses)

            # VALIDATION
            model.eval()
            val_loss = 0.0
            total_iou_val = 0.0
            total_pixel_accuracy_val = 0.0
            total_dice_coefficient_val = 0.0

            val_dataloader = tqdm(val_dataloader, desc=f"Validation", unit="batch")

            with torch.no_grad():
                for images, masks in val_dataloader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)

                    v_loss = criterion(outputs, masks)
                    val_loss += v_loss.item()

                    # Calculating metrics for Validation
                    pred_masks = outputs > 0.5
                    iou_val, dice_coefficient_val, pixel_accuracy_val = calculate_metrics(
                        pred_masks, masks
                    )

                    total_iou_val += iou_val
                    total_pixel_accuracy_val += pixel_accuracy_val
                    total_dice_coefficient_val += dice_coefficient_val

                    # Displaying metrics in progress bar description
                    val_dataloader.set_postfix(
                        val_loss=v_loss.item(),
                        val_iou=iou_val,
                        val_pix_acc=pixel_accuracy_val,
                        val_dice_coef=dice_coefficient_val,
                        lr=current_lr,
                    )

            val_loss /= len(val_dataloader)
            avg_iou_val = total_iou_val / len(val_dataloader)
            avg_pixel_accuracy_val = total_pixel_accuracy_val / len(val_dataloader)
            avg_dice_coefficient_val = total_dice_coefficient_val / len(val_dataloader)

            validating_losses.append(val_loss)
            validating_ious.append(avg_iou_val)
            validating_pixes.append(avg_pixel_accuracy_val)
            validating_dices.append(avg_dice_coefficient_val)
            print("Validation Losses: ", validating_losses)

            scheduler.step(val_loss)

            print(
                f"\nEpoch {epoch + 1}/{num_epochs}\n"
                f"Avg Train Loss: {train_loss:.4f}\n"
                f"Avg Validation Loss: {val_loss:.4f}\n"
                f"Avg mIoU Train: {avg_iou_train:.4f}\n"
                f"Avg mIoU Val: {avg_iou_val:.4f}\n"
                f"Avg Pix Acc Train: {avg_dice_coefficient_train:.4f}\n"
                f"Avg Pix Acc Val: {avg_pixel_accuracy_val:.4f}\n"
                f"Avg Dice Coeff Train: {avg_dice_coefficient_train:.4f}\n"
                f"Avg Dice Coeff Val: {avg_dice_coefficient_val:.4f}\n"
                f"Current LR: {current_lr}\n"
                f"{'-'*50}"
            )

            # Saving best model
            if val_loss < best_val_loss:
                no_improvement_epochs = 0
                click.secho(
                    message=f"\n👀 val_loss improved from {best_val_loss:.4f} to {val_loss:.4f}\n",
                    fg="green",
                )
                best_val_loss = val_loss
                torch.save(model.state_dict(), "./output/best_model_UNET3.pth")
                click.secho(message="Saved Best Model! 🙌\n", fg="green")
                print(f"{'-'*50}")
            else:
                no_improvement_epochs += 1
                click.secho(
                    message=f"\nval_loss did not improve from {best_val_loss:.4f}\n", fg="yellow"
                )
                print(f"{'-'*50}")

            # Append the training and validation logs to the CSV file
            csv_writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    val_loss,
                    avg_iou_train,
                    avg_iou_val,
                    avg_pixel_accuracy_train,
                    avg_pixel_accuracy_val,
                    avg_dice_coefficient_train,
                    avg_dice_coefficient_val,
                    current_lr,
                ]
            )

            # Early-Stopping
            if early_stop:
                if no_improvement_epochs >= patience_epochs:
                    click.secho(
                        message=f"\nEarly Stopping: val_loss did not improve for {patience_epochs} epochs.\n",
                        fg="red",
                    )
                    break
    
    #print("Training Losses: ", training_losses, "Validation Losses: ", validating_losses)
    #print("Validation Losses: ", validating_losses)


    click.secho(message="🎉 Training Done!", fg="blue", nl=True)

    ######################### Guardar Graficos de metricas #############################
    
    
    #epochs = range(1, num_epochs + 1)
    
    """
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_losses, 'r', label='Train Loss')
    plt.plot(epochs, validating_losses, 'b', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, training_ious, 'r', label='Train IoU')
    plt.plot(epochs, validating_ious, 'b', label='Validation IoU')
    plt.title('IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, training_pixes, 'r', label='Train Pixel Accuracy')
    plt.plot(epochs, validating_pixes, 'b', label='Validation Pixel Accuracy')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Pixel Accuracy')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, training_dices, 'r', label='Train Dice Coefficient')
    plt.plot(epochs, validating_dices, 'b', label='Validation Dice Coefficient')
    plt.title('Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./output/training_metrics_unet1.png')
    """
    epochs = list(range(1, len(training_losses) + 1))

    plt.figure(figsize=(15, 7))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_losses, label='Train Loss')
    plt.plot(epochs, validating_losses, label='Validation Loss')
    plt.xticks(ticks=list(range(1, len(training_losses) + 1))) 
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.legend()
    #plt.text(1, min(training_losses), 'a', fontsize=12, verticalalignment='bottom', horizontalalignment='right')  # Añadir letra 'q' en la parte inferior
    #plt.text(len(epochs) / 2, min(training_losses) - 0.01, 'a', fontsize=12, ha='center')  # Colocar la letra 'q' debajo de la gráfica
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, training_ious, label='Train mIoU')
    plt.plot(epochs, validating_ious, label='Validation mIoU')
    plt.xticks(ticks=list(range(1, len(training_losses) + 1))) 
    plt.title('Training and Validation mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.grid()
    plt.tight_layout()
    plt.legend()
    #plt.text(1, min(training_ious), 'b', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    #plt.text(len(epochs) / 2, min(training_ious) - 0.01, 'b', fontsize=12, ha='center')  # Colocar la letra 'b' debajo de la gráfica

    plt.subplot(2, 2, 3)
    plt.plot(epochs, training_pixes, label='Train Pixel Accuracy')
    plt.plot(epochs, validating_pixes, label='Validation Pixel Accuracy')
    plt.xticks(ticks=list(range(1, len(training_losses) + 1))) 
    plt.title('Training and Validation PA')
    plt.xlabel('Epochs')
    plt.ylabel('PA')
    plt.grid()
    plt.tight_layout()
    plt.legend()
    #plt.text(1, min(training_pixes), 'c', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    #plt.text(len(epochs) / 2, min(training_pixes) - 0.01, 'c', fontsize=12, ha='center')  # Colocar la letra 'c' debajo de la gráfica

    plt.subplot(2, 2, 4)
    plt.plot(epochs, training_dices, label='Train Dice Coefficient')
    plt.plot(epochs, validating_dices, label='Validation Dice Coefficient')
    plt.xticks(ticks=list(range(1, len(training_losses) + 1))) 
    plt.title('Training and Validation Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.grid()
    plt.legend()
    #plt.text(1, min(training_dices), 'd', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    #plt.text(len(epochs) / 2, min(training_dices) - 0.01, 'd', fontsize=12, ha='center')  # Colocar la letra 'd' debajo de la gráfica
    # Añadir título general
    plt.suptitle('DeepLabV3Plus', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar layout para no solapar con el título

    plt.tight_layout()
    plt.savefig('./output/training_metric_UNET3.png')

    ####################################################################################

    return

if __name__ == "__main__":
    main()