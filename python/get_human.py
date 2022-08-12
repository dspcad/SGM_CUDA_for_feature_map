import cv2
import numpy as np
import sys



if __name__ == "__main__":
    disp_img = cv2.imread(sys.argv[1])
    mask_img = cv2.imread(sys.argv[2])

    print(f"{disp_img.shape}")
    print(f"{mask_img.shape}")
 
    out_img = np.zeros(disp_img.shape)
    h, w, c = mask_img.shape

    for r in range(0,h):
        for c in range(0,w):
            if any(mask_img[r][c]>20):
                #print(mask_img[r][c])
                out_img[r][c] = disp_img[r][c]


    cv2.imwrite(sys.argv[3], out_img)

