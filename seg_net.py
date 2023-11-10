import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from CSwin import cswin_small


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels + mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x
class cswinunet(nn.Module):
    def __init__(self):
        super(cswinunet, self).__init__()

        # 使用resnet50作为主干网络
        self.backbone = cswin_small()
        path = r'C:\Users\lee\Desktop\cswin\breast_multitask_baseline\cswin_small_224.pth'
        save_model = torch.load(path)
        # print(save_model['state_dict_ema'].keys())
        model_dict = self.backbone.state_dict()
        # print(model_dict.keys())
        state_dict = {k: v for k, v in save_model['state_dict_ema'].items() if k in model_dict.keys()}
        # print(state_dict)
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.up_channel = nn.Conv2d(1,3,1)

        self.up3 = DecoderBlock(512, 256, 256)
        self.up2 = DecoderBlock(256, 128, 128)
        self.up1 = DecoderBlock(128, 64, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.up_channel(x)
        cswin = self.backbone(x)
        layer1 = cswin[0]
        layer2 = cswin[1]
        layer3 = cswin[2]
        layer4 = cswin[3]
        #layer 4 512  3 256  2 128 1 64

        up3 = self.up3(layer4, layer3)
        #print(up3.shape)
        up2 = self.up2(up3, layer2)
        #print(up2.shape)
        up1 = self.up1(up2, layer1)
        #print(up1.shape)
        logits = self.outc(up1)
        logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=True)
#64  128  256  512
        return logits


if __name__ == "__main__":


    image = torch.randn(8,1,512,512)




    model_seg = cswinunet()

    mask = model_seg(image)
    print(mask.shape)
