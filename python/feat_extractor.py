import torchvision.models as models
import torch, sys
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.transforms import transforms
import torch.nn as nn
import time
from matplotlib import pyplot as plt



def stereo_match(left_img, right_img, layer):
    tensor_trans = transforms.ToTensor()
    norm_trans   = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    left  = Image.open(left_img).convert('RGB')
    right = Image.open(right_img).convert('RGB')
    print(f"{left.size}")
    print(f"{right.size}")
    left  = norm_trans(tensor_trans(left)).cuda()
    right = norm_trans(tensor_trans(right)).cuda()
    h, w = left.shape[1:]  # assume that both images are same size   



    #print(left.shape)
    #print(right.shape)
    print(f"w: {w}")
    print(f"h: {h}")
    # Depth (or disparity) map
    #depth = np.zeros((h, w), p.uint8)

    if layer >= 4:
        w = w//2
        h = h//2

    disparity = np.zeros((h,w),np.uint16)
    

    #offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range


    #vgg19_model = models.vgg19(pretrained=True)
    vgg_model = models.vgg19(pretrained=True)
    #print(vgg16_model)
    #print(vgg16_model.features)

    #img = torch.rand(1,3,3,3).cuda()
    #print(vgg19_model.features(img).shape)
    vgg_model.cuda()
    vgg_model.eval()
    #feat_extractor = vgg19_model.features[3]
    feat_extractor = nn.Sequential(vgg_model.features[:layer+1])
    print(feat_extractor)
    #print(feat_extractor(img).shape)

    

    left  = feat_extractor(torch.unsqueeze(left,0))[0]
    right = feat_extractor(torch.unsqueeze(right,0))[0]

    print(f"{left.shape}")
    print(f"{right.shape}")

    left  = torch.transpose(left,1,2)
    left  = torch.flip(left, [2])
    right = torch.transpose(right,1,2)
    right = torch.flip(right, [2])




    np.save("left.npy",np.array(left.cpu().detach().numpy(),dtype='f8'))
    np.save("right.npy",np.array(right.cpu().detach().numpy(),dtype='f8'))


    print(f"left:  {left.shape}")
    print(f"right: {right.shape}")



if __name__ == "__main__":

    #print(sys.argv[0])
    #for i in range(4,8):
    i=3
    print(f"Generating the result of layer {i}...")
    stereo_match(sys.argv[1],sys.argv[2],i)
