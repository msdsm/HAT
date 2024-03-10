import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from hat import HAT
from dataset import HAT_Dataset
from tqdm import tqdm
from PIL import Image
import numpy as np


data_path = "~/m1/myhat/dataset/horse_train"
data_path = os.path.expanduser(data_path)
lowimg_path = data_path + "/horse_mosaic/"
gtimg_path = data_path + "/horse_gt/"
lowimg_files = os.listdir(lowimg_path)
gtimg_files = os.listdir(gtimg_path)


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_dataset = HAT_Dataset(
    lowimg_path = lowimg_path,
    gtimg_path = gtimg_path,
    lowimg_files = lowimg_files,
    gtimg_files=gtimg_files,
    transform=transform
)
test_dataloader = DataLoader(test_dataset, batch_size=1)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = HAT(
    img_size=256,
    upscale=1 # 2にしてnetworkの最後の層でプーリングつけても良い
)
if torch.cuda.device_count() > 1:
    print("並列にした")
    model = nn.DataParallel(model)
    
model.to(device)

#########################超重要#######################
parameters_load_path = "./experiments/pretrained_models/HAT-L_SRx2_ImageNet-pretrain.pth"
model.load_state_dict(torch.load(parameters_load_path))
######################################################

model.eval()


print("epoch {}:".format(str(epoch+1)))
model.to(device)
model.train()
with tqdm(test_dataloader, total=len(test_dataloader)) as pbar:
    for i, (lowimgs, gtimgs) in enumerate(pbar):
        name = lowimgs[i].split("/")[-1]
        lowimgs = lowimgs.to(device)
        gtimgs = gtimgs.to(device) # 使わない、テスト時の損失計算するなら使う
        lowimgs = lowimgs.unsqueeze(0)
        output = model.forward(lowimgs)

        output_img = output.to("cpu").numpy().copy().transpose(1, 2, 0).astype(np.float32)

        output_img = Image.fromarray(np.clip(output_img*255.0, a_min=0, a_max=255).astype(np.uint8))

        output_img.save("./output/horse_test/20240310/"+name)