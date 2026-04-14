import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import os

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset

def get_args():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 Training Script")
    parser.add_argument("--task", type=int, default=4, choices=[1, 2, 3, 4], 
                        help="Task to run (1: Classify, 2: Localize, 3: Seg, 4: Multi)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability (Task 1)")
    parser.add_argument("--freeze_mode", type=str, default="full", choices=["strict", "partial", "full"], 
                        help="Transfer learning mode for Task 3 (strict, partial, full)")
    parser.add_argument("--run_name", type=str, default="baseline-run", help="Wandb run name")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Initialize Weights & Biases
    wandb.init(project="pet-multitask", name=args.run_name, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize as required by Assignment 2 extra instructions
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    dataset = OxfordIIITPetDataset(root_dir="dataset", split="train", transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    
    # TASK 1: Classification
    
    if args.task == 1:
        print(f"Starting Task 1 (Classification) with dropout={args.dropout}")
        model = VGG11Classifier(num_classes=37, dropout_p=args.dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for imgs, labels, _, _ in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                wandb.log({"train_loss": loss.item()})
                
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), "classifier.pth")
        print("Saved classifier.pth")

    
    # TASK 2: Localization
    
    elif args.task == 2:
        print("Starting Task 2 (Localization)")
        model = VGG11Localizer(dropout_p=args.dropout).to(device)
        iou_loss_fn = IoULoss()
        mse_loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for imgs, _, bboxes, _ in loader:
                imgs, bboxes = imgs.to(device), bboxes.to(device)
                outputs = model(imgs)
                
                # The assignment requires MSE + custom_IOU_loss
                loss_iou = iou_loss_fn(outputs, bboxes)
                loss_mse = mse_loss_fn(outputs, bboxes)
                loss = loss_iou + loss_mse
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                wandb.log({"train_loss": loss.item(), "iou_loss": loss_iou.item(), "mse_loss": loss_mse.item()})
                
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), "localizer.pth")
        print("Saved localizer.pth")

    
    # TASK 3: Segmentation (With Transfer Learning Freezing)
    
    elif args.task == 3:
        print(f"Starting Task 3 (Segmentation) with freeze_mode={args.freeze_mode}")
        model = VGG11UNet(num_classes=3).to(device)
        
        # Freezing Logic for Transfer Learning Showdown
        if args.freeze_mode == "strict":
            for param in model.encoder.parameters():
                param.requires_grad = False
        elif args.freeze_mode == "partial":
            # Freeze early blocks, leave later blocks unfrozen
            for param in model.encoder.block1.parameters(): param.requires_grad = False
            for param in model.encoder.block2.parameters(): param.requires_grad = False
            for param in model.encoder.block3.parameters(): param.requires_grad = False
            
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for imgs, _, _, masks in loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks.long())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                wandb.log({"train_loss": loss.item()})
                
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), "unet.pth")
        print("Saved unet.pth")

    
    # TASK 4: Unified Multi-Task Pipeline
    
    elif args.task == 4:
        print("Starting Task 4 (Unified Multi-Task Pipeline)")
        model = MultiTaskPerceptionModel().to(device)
        
        ce = nn.CrossEntropyLoss()
        iou = IoULoss()
        mse = nn.MSELoss()
        seg_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for imgs, labels, bboxes, masks in loader:
                imgs, labels, bboxes, masks = imgs.to(device), labels.to(device), bboxes.to(device), masks.to(device)

                outputs = model(imgs)

                loss_cls = ce(outputs["classification"], labels)
                loss_loc = iou(outputs["localization"], bboxes) + mse(outputs["localization"], bboxes)
                loss_seg = seg_loss(outputs["segmentation"], masks.long())

                loss = loss_cls + loss_loc + loss_seg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                wandb.log({
                    "total_loss": loss.item(),
                    "loss_cls": loss_cls.item(),
                    "loss_loc": loss_loc.item(),
                    "loss_seg": loss_seg.item()
                })

            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(loader):.4f}")
        
        # Save final unified model
        torch.save(model.state_dict(), "multitask.pth")
        print("Saved multitask.pth")

    wandb.finish()

if __name__ == "__main__":
    main()