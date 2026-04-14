import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset


wandb.init(project="pet-multitask", name="baseline-run")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = OxfordIIITPetDataset(
    root_dir="dataset",  
    split="train",
    transform=transform
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = MultiTaskPerceptionModel().to(device)

ce = nn.CrossEntropyLoss()
iou = IoULoss()
seg_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for imgs, labels, bboxes, masks in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        masks = masks.to(device)

        outputs = model(imgs)

        loss_cls = ce(outputs["classification"], labels)
        loss_loc = iou(outputs["localization"], bboxes)
        loss_seg = seg_loss(outputs["segmentation"], masks.long())

        loss = loss_cls + loss_loc + loss_seg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        wandb.log({
            "loss": loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_loc": loss_loc.item(),
            "loss_seg": loss_seg.item()
        })

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "model.pth")

wandb.finish()