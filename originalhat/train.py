import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from hat import HAT
from dataset import HAT_Dataset
from tqdm import tqdm

data_path = "~/m1/myhat/dataset/horse_train"
data_path = os.path.expanduser(data_path)
lowimg_path = data_path + "/horse_mosaic/"
gtimg_path = data_path + "/horse_gt/"
lowimg_files = os.listdir(lowimg_path)
gtimg_files = os.listdir(gtimg_path)

epochs = 50
batch_size = 32
learning_rate = 1e-4

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = HAT_Dataset(
    lowimg_path = lowimg_path,
    gtimg_path = gtimg_path,
    lowimg_files=lowimg_files,
    gtimg_files=gtimg_files,
    transform=transform
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = HAT(
    img_size=256,
    upscale=1,  # 2にしてnetworkの最後の層でプーリングつけても良い
    upsampler='pixelshuffle'
)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

criterion = nn.L1Loss()
# criterion = nn.MSELoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
'''
#########################超重要#######################
parameters_load_path = "./experiments/pretrained_models/HAT-L_SRx2_ImageNet-pretrain.pth"
model.load_state_dict(torch.load(parameters_load_path))
######################################################
'''


'''
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
エラー解消として下の文を追加するということがweb検索でヒット
'''
# torch.set_grad_enabled(True)  # Context-manager 
# 解決しなかった

"""
for p in model.parameters():
    print(p)
    if p.requires_grad == False:
        print(p)
"""
"""
for name, param in model.named_parameters():
    param.requires_grad = True
"""
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    print("epoch {}:".format(str(epoch+1)))
    model.to(device)
    model.train()
    with tqdm(train_dataloader, total=len(train_dataloader)) as pbar:
        for i, (lowimgs, gtimgs) in enumerate(pbar):
            lowimgs = lowimgs.to(device)
            gtimgs = gtimgs.to(device)

            optimizer.zero_grad()
            output = model(lowimgs)
            #print(output.requires_grad)
            loss = criterion(output, gtimgs)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())
    
    print("mean_train_loss:{}".format(sum(train_losses)/len(train_losses)))
    if (epoch+1)%10 == 0:
        model.to("cpu")
        torch.save(model.module.state_dict(), "./checkpoint/hat20240310.pth")
        print("save model")