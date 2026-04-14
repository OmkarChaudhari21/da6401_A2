import torch
from models.multitask import MultiTaskPerceptionModel

model = MultiTaskPerceptionModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

x = torch.randn(1, 3, 224, 224)

out = model(x)

print(out["classification"].shape)
print(out["localization"])
print(out["segmentation"].shape)