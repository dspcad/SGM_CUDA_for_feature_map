import torchvision.models as models
import torch, sys
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.transforms import transforms
import torch.nn as nn
import time
from matplotlib import pyplot as plt



def stereo_match(left_img, right_img, kernel, max_offset, layer):
    tensor_trans = transforms.ToTensor()
    norm_trans   = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    left  = norm_trans(tensor_trans(Image.open(left_img))).cuda()
    right = norm_trans(tensor_trans(Image.open(right_img))).cuda()
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
    

    kernel_half = kernel // 2
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

    #left  = rearrange(left, "c h w-> h w c")
    #right = rearrange(right, "c h w-> h w c")


    np.save("left.npy",np.array(left.cpu().detach().numpy(),dtype='f8'))
    np.save("right.npy",np.array(right.cpu().detach().numpy(),dtype='f8'))

    print(f"left:  {left.shape}")
    print(f"right: {right.shape}")
    for y in range(kernel_half, h - kernel_half):
        print("\rProcessing.. %d%% complete"%(y / (h - kernel_half) * 100), end="", flush=True)
        #print(f"\rProcessing y: {y} ", flush=True)

        for x in range(kernel_half, w - kernel_half):
            if x-kernel_half-max_offset < 0 or x+kernel_half+1>=w:
                continue


            start_time = time.time()
            left_img_patch = left[:,y-kernel_half:y+kernel_half+1,x-kernel_half:x+kernel_half+1]
            #left_img_patch = feat_extractor(torch.unsqueeze(left[:,y-kernel_half:y+kernel_half,x-kernel_half:x+kernel_half],0))
            #print(f"left:    x:{x}   y:{y}   patch: {left[:,y-kernel_half:y+kernel_half,x-kernel_half:x+kernel_half].shape}")
            #print(f"left:    x:{x}   y:{y}   patch: {left_img_patch.shape}")
            #left_img_patch = torch.flatten(left_img_patch)
            left_img_patch = rearrange(left_img_patch, "c h w-> (c h w)")
            right_img_patch_batch = torch.zeros(max_offset,64,kernel,kernel).cuda()
            for offset in range(max_offset):
                if x-kernel_half-offset < 0 or x+kernel_half-offset+1>=w:
                    continue
            
                #print(f"    x: {x}")
                #print(f"    y: {y}")
                #right_img_patch_batch[offset] = right[:,y-kernel_half:y+kernel_half,x-kernel_half-offset:x+kernel_half-offset]
                right_img_patch_batch[offset] = right[:,y-kernel_half:y+kernel_half+1,x-kernel_half-offset:x+kernel_half+1-offset]


            #print(f"Before: right_img_patch_batch: {right_img_patch_batch.shape}")
            right_img_patch_batch = rearrange(right_img_patch_batch, "b c h w-> b (c h w)")
         

            #print(f"left:  {left_img_patch.shape}")
            #print(f"right: {right_img_patch_batch.shape}")
            #print(right_img_patch_batch[idx])
            #print(right_img_patch_batch)
            diff_batch = left_img_patch-right_img_patch_batch

            #print(f"diff_batch: {diff_batch.shape}")
            #tmp = diff_batch*diff_batch
            #print(f"    tmp: {tmp.shape}")
            cur_ssd = torch.sum(diff_batch*diff_batch,1)
            #print(f"    cur_ssd: {torch.argmax(cur_ssd)}")
            disparity[y, x] = torch.argmin(cur_ssd)
            #depth[y, x] = torch.argmin(cur_ssd) * offset_adjust
            #print("--- %s seconds ---" % (time.time() - start_time))
            
    # Convert to PIL and save it
    #Image.fromarray(depth).save(f"depth_{layer}.png")
   
    #plt.imshow(disparity,'viridis')
    plt.imshow(disparity,'gray')
    #plt.imshow(disparity,cmap='gray',vmin=0, vmax=60)
    plt.savefig(f"disparity_{layer}.png")
    #plt.show()



if __name__ == "__main__":

    #print(sys.argv[0])
    #for i in range(4,8):
    i=3
    print(f"Generating the result of layer {i}...")
    stereo_match(sys.argv[1],sys.argv[2],3,64,i)
