# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory

# Prints out training loss, validation loss, and validation accuracy 
# as the network trains

# Options: 

#          * Set directory to save checkpoints: 
#            python train.py data_dir --save_dir save_directory 

#          * Choose architecture: 
#            python train.py data_dir --arch "vgg13" 

#          * Set hyperparameters: 
#            python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 

#          * Use GPU for training: 
#            python train.py data_dir --gpu


import argparse
import os
import time

import torch
from torch import nn 
from torch import optim
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler

def get_input_args():

    parser = argparse.ArgumentParser(description="Train a Neural Network on a Flower Image Dataset.")

    parser.add_argument("data_dir", 
                        help="Path to dataset directory (with train and valid folders)")

    parser.add_argument("--save_dir", default=".", 
                        help="Directory to save checkpoints")

    parser.add_argument("--arch", default="resnet18", 
                        help="Model architecture: choose: resnet18, vgg16, or densenet121")
    
    parser.add_argument("--learning_rate", type=float, default=0.003)

    parser.add_argument("--hidden_units", type=int, default=512, 
                        help="Hidden layer size (only used for some models)")
    
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    return parser.parse_args()


def load_data(data_dir):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {dataset: datasets.ImageFolder(os.path.join(data_dir, dataset),
                                                    data_transforms[dataset])
                    for dataset in ['train', 'valid']}

    image_dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'],
                                            batch_size=64, shuffle=True, 
                                            num_workers=0),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'],
                                            batch_size=64, shuffle=False, 
                                            num_workers=0)
    }

    return image_dataloaders, image_datasets


def build_model(arch, hidden_units):

    # load pretrained architecture for each option and add classifier

    if arch == "resnet18":
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        input_size = model.fc.in_features
        
        model.fc = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102)
        )
        classifier = model.fc

    elif arch == "vgg16":
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
        input_size = model.classifier[0].in_features

        model.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102)
        )
        classifier = model.classifier

    elif arch == "densenet121":
        model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        input_size = model.classifier.in_features

        model.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102)
        )
        classifier = model.classifier

    else:
        raise ValueError(f"Unsupported architecture '{arch}'")
    
    # freeze pretrained architecture
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze model classifier for training
    for param in classifier.parameters():
        param.requires_grad = True

    return model, classifier.parameters()

# pytorch documentation used for reference:
# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def train_model(model, criterion, optimizer, dataloaders, datasets,
                device, scheduler, num_epochs=5):

    # track the amount of time training
    start = time.time()

    best_accuracy = 0.0
    best_model_weights = model.state_dict()

    print()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}\n")

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    values, prediction = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # back propagation if training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(prediction == labels.data)

            if phase == 'train':
                scheduler.step()

            # epoch metrics
            epoch_loss = running_loss / len(datasets[phase])
            epoch_accuracy = running_correct.double() / len(datasets[phase])
            print(f"{phase} loss: {epoch_loss:.4f}  accuracy: {epoch_accuracy:.4f}")

            # save best model weights
            if phase == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = model.state_dict()

        print()

    # load best weights before returning
    model.load_state_dict(best_model_weights)

    time_elapsed = time.time() - start
    
    print(f'training complete in {time_elapsed // 60:.0f} minutes and {time_elapsed % 60:.0f} seconds')
    print(f'best accuracy: {best_accuracy:.4f}')

    return model


def save_checkpoint(model, image_datasets, optimizer, 
                    save_dir, model_arch, num_epochs):
    
    if model_arch == "resnet18":
        classifier = model.fc
    else:
        classifier = model.classifier

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': num_epochs,
        'classifier': classifier,
        'arch': model_arch
    }

    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)


if __name__ == "__main__":

    args = get_input_args()

    # set device
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    
    if args.gpu and device != "cuda":
        print("GPU requested but CUDA not available. Using CPU.")

    image_dataloaders, image_datasets = load_data(args.data_dir)

    model, classifier_params = build_model(args.arch, args.hidden_units)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_params, lr=0.003)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    model = train_model(model, criterion, optimizer, image_dataloaders, 
                        image_datasets, device, scheduler, args.epochs)

    save_checkpoint(model, image_datasets, optimizer, 
                    args.save_dir, args.arch, args.epochs)

