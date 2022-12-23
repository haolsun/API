import numpy as np
import cv2

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)/255
    heatmap[:,:,0] = (heatmap[:,:,0] - np.min(heatmap[:,:,0]))/ np.max(heatmap[:,:,0])
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam