import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *
from model.flow_warp import flow_warp
from model.VGGPerceptualLoss import *

class FILM(nn.Module):
    def __init__(self):
        super(FILM, self).__init__()
        '''self.nf = 64
        self.in_c = [64, 192, 448, 960]
        self.fusion_in_c = [128, 256, 512, 1926]'''

        self.nf = 32
        self.in_c = [32, 96, 224, 480]
        self.fusion_in_c = [64, 128, 256, 966]

        self.num_scale_down = 5
        self.fusion_level = 5
        self.num_extracts = 4

        self.feature_extracts = [
            nn.Sequential(
                nn.Conv2d(1, self.nf, 3, 1, 1, bias=True),
                nn.ReLU(),
                nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True),
                nn.ReLU(),
            )
        ]
        for i in range(0,3):
            self.feature_extracts.append(
                nn.Sequential(
                    nn.Conv2d(self.nf * (2**i), self.nf* (2**(i+1)), 3, 1, 1, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(self.nf* (2**(i+1)), self.nf* (2**(i+1)), 3, 1, 1, bias=True),
                    nn.ReLU(),
                )
            )
        self.feature_extracts = nn.ModuleList(self.feature_extracts)
        self.flow_estimations = []

        for i in range(4):
            in_c = self.in_c[i]
            mid_c = (self.nf//2) * (2**i)
            self.flow_estimations.append(nn.Sequential(
                nn.Conv2d(in_c * 2, mid_c, 3, 1, 1, bias=True),
                nn.ReLU(),
                nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=True),
                nn.ReLU(),
                nn.Conv2d(mid_c, mid_c, 3, 1, 1, bias=True),
                nn.ReLU(),
                nn.Conv2d(mid_c, mid_c//2, 3, 1, 1, bias=True),
                nn.ReLU(),
                nn.Conv2d(mid_c//2, 2, 3, 1, 1, bias=True),
            ))
        self.flow_estimations = nn.ModuleList(self.flow_estimations)

        self.fusion_first = []
        self.fusion = []
        for i in range(4):
            out_c = self.nf << i
            self.fusion_first.append(nn.Sequential(
                nn.Conv2d(self.fusion_in_c[i], out_c, 2, 1, 1, dilation=2, bias=True),
                nn.ReLU(),
            ))
            self.fusion.append(nn.Sequential(
                nn.Conv2d(out_c + (self.in_c[i] + 1 + 2) * 2, out_c, 3, 1, 1, bias=True),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, 1, 1, bias=True),
                nn.ReLU(),
            ))
        self.fusion_first = nn.ModuleList(self.fusion_first)
        self.fusion = nn.ModuleList(self.fusion)
        self.final_conv = nn.Conv2d(self.nf, 1, 1, 1, bias=True)

        self.AvgMaxPool = nn.AvgPool2d((2, 2), stride=2)
        self.BIUp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.vgg19perceptual = VGGPerceptualLoss_FILM()

    def getPerceptualLoss(self, input, target):
        return self.vgg19perceptual(input, target)

    def extract_features_and_img(self, img0, img1 ):
        B, C, H, W = img0.size()
        multi_scale_img = [torch.cat([img0, img1], dim=0)] # 2*B , C, H, W

        for i in range(self.num_scale_down - 1):
            multi_scale_img.append(self.AvgMaxPool(multi_scale_img[i]))

        output = []
        next_scale = []
        for i in range(self.num_extracts):

            if i == 0:
                for j in range(self.num_scale_down):
                    output.append(self.feature_extracts[i](multi_scale_img[j]))
                    next_scale.append(output[j])

            else:
                for j in range(self.num_scale_down-1, i-1, -1):
                    next_scale[j] = self.feature_extracts[i](self.AvgMaxPool(next_scale[j-1]))
                    output[j] = torch.cat([output[j], next_scale[j]], dim=1)

        for i in range(self.num_scale_down):
            output[i] = output[i].view(B, 2, -1, H//(2**i), W//(2**i))
            multi_scale_img[i] = multi_scale_img[i].view(B, 2, -1, H//(2**i), W//(2**i))


        '''for i in output:
            print(i.shape)
        print()'''

        return multi_scale_img, output

    def fusion_and_output(self, multi_scale_img, multi_res_fea,multi_flowt0, multi_flowt1 ):
        B, N, C, H, W = multi_scale_img[self.fusion_level - 1].size()
        imgs = multi_scale_img[self.fusion_level - 1].view(B, -1, H, W)

        ## feature warping
        w0_fea = flow_warp(multi_res_fea[self.fusion_level - 1][:, 0, :, :, :], multi_flowt0[self.fusion_level - 1].permute(0, 2, 3, 1))
        w1_fea = flow_warp(multi_res_fea[self.fusion_level - 1][:, 1, :, :, :], multi_flowt1[self.fusion_level - 1].permute(0, 2, 3, 1))

        #feas = multi_res_fea[self.fusion_level - 1].view(B, -1, H, W)
        feas = torch.cat([w0_fea, w1_fea], dim=1)
        flows = torch.cat([multi_flowt0[self.fusion_level - 1], multi_flowt1[self.fusion_level - 1]], dim=1)
        prev_fea = torch.cat([imgs, feas, flows], dim=1)

        for i in range(self.fusion_level - 2, -1, -1):
            if i >= self.num_extracts - 1:
                fusion_conv1 = self.fusion_first[self.num_extracts - 1]
                fusion_conv2 = self.fusion[self.num_extracts - 1]

            else:
                fusion_conv1 = self.fusion_first[i]
                fusion_conv2 = self.fusion[i]

            B, N, C, H, W = multi_scale_img[i].size()
            imgs = multi_scale_img[i].view(B, -1, H, W)

            ## feature warping
            w0_fea = flow_warp(multi_res_fea[i][:, 0, :, :, :],
                               multi_flowt0[i].permute(0, 2, 3, 1))
            w1_fea = flow_warp(multi_res_fea[i][:, 1, :, :, :],
                               multi_flowt1[i].permute(0, 2, 3, 1))
            feas = torch.cat([w0_fea, w1_fea], dim=1)
            flows = torch.cat([multi_flowt0[i], multi_flowt1[i]], dim=1)
            input = torch.cat([imgs, feas, flows],dim=1)


            out_fea = fusion_conv1(self.BIUp(prev_fea))
            out_fea = fusion_conv2(torch.cat([out_fea, input],dim=1))
            prev_fea = out_fea

        return self.final_conv(out_fea)

    def forward(self, x, scale=[4,2,1], timestep=0.5):
        img0 = x[:, :1]
        img1 = x[:, 1:2]
        gt = x[:, 2:] # In inference time, gt is None


        B, C, H, W = img0.size()

        multi_scale_img, multi_res_fea = self.extract_features_and_img(img0, img1)
        multi_flowt0 = []
        multi_flowt1 = []

        for i in range(self.num_scale_down-1, -1, -1):
            if i >= self.num_extracts - 1:
                flow_estimation = self.flow_estimations[self.num_extracts - 1]
            else:
                flow_estimation = self.flow_estimations[i]

            fea_0, fea_1 = multi_res_fea[i][:, 0, :, :, :], multi_res_fea[i][:, 1, :, :, :]

            B, C, H, W = fea_0.size()

            input = torch.cat([ torch.cat([fea_0,fea_1],dim=1), torch.cat([fea_1,fea_0], dim=1) ], dim=0)# 2B, 2C, H, W
            out = flow_estimation(input)

            if i == self.num_scale_down-1:
                flowt0, flowt1 = out[:B, :, :, :] * 2 , out[B:, :, :, :] * 2
                prev_flowt0, prev_flowt1 = flowt0, flowt1
            else:
                flowt0, flowt1 = self.BIUp(prev_flowt0) + out[:B, :, :, :], self.BIUp(prev_flowt1) + out[B:, :, :, :]
                prev_flowt0, prev_flowt1 = flowt0, flowt1

            multi_flowt0.append(flowt0)
            multi_flowt1.append(flowt1)

        multi_flowt0 = multi_flowt0[::-1]
        multi_flowt1 = multi_flowt1[::-1]

        result = self.fusion_and_output(multi_scale_img, multi_res_fea,multi_flowt0, multi_flowt1)

        return result

