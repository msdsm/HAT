import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from hat import PartialHAT
from dataset import HAT_Dataset
from tqdm import tqdm

data_path = "/home/yuya/m1/hat/HAT/datasets/My_horse"
lowimg_files = os.listdir(data_path + "/horse_mosaic")
maskimg_files = os.listdir(data_path + "/horse_mask")
gtimg_files = os.listdir(data_path + "/horse_gt")

epochs = 50
batch_size = 32
learning_rate = 1e-4

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = HAT_Dataset(
    data_path=data_path,
    lowimg_files=lowimg_files,
    maskimg_files=maskimg_files,
    gtimg_files=gtimg_files,
    transform=transform
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = PartialHAT(
    img_size=256,
    upscale=2 # 2にしてnetworkの最後の層でプーリングつけても良い
)
if torch.cuda.device_count() > 1:
    model.DataParallel(model, device_ids=[0, 1, 2, 3])
model.to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
for epoch in range(epochs):
    print("epoch {}:".format(str(epoch+1)))
    model.to(device)
    model.train()
    with tqdm(train_dataloader, total=len(train_dataloader)) as pbar:
        for i, (lowimgs, maskimgs, gtimgs) in enumerate(pbar):
            lowimgs = lowimgs.to(device)
            maskimgs = maskimgs.to(device)
            gtimgs = gtimgs.to(device)

            output = model.forward(lowimgs, maskimgs)

            loss = criterion(output, gtimgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())
    
    print("mean_train_loss:{}".format(sum(train_losses)/len(train_losses)))
    if (epoch+1)%10 == 0:
        model.to("cpu")
        torch.save(model.module.state_dict(), "./checkpoint/DDPM_SimpleUnet.pth")
        print("save model")