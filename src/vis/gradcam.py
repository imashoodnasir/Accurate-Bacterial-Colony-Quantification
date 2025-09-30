
import torch, cv2
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

def gradcam_overlay(model, target_layer, rgb_img_np: np.ndarray, device="cuda"):
    model.eval()
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=(device=="cuda"))
    x = rgb_img_np.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    inp = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float()
    if device=="cuda": inp = inp.cuda()
    grayscale = cam(input_tensor=inp)[0]
    return show_cam_on_image(rgb_img_np.astype(np.float32)/255.0, grayscale, use_rgb=True)
