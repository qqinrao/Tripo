#提供图像处理功能
"""
TripoSG 系统中的关键预处理组件，负责准备输入图像以便进行高质量的 3D 形状合成。该模块实现了一系列图像处理功能，确保输入到 TripoSG 模型的图像具有最佳质量和格式。
实现 prepare_image 函数
负责图像的预处理工作
"""
# -*- coding: utf-8 -*-
import os
from skimage.morphology import remove_small_objects
from skimage.measure import label
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def find_bounding_box(gray_image):
    #cv2.threshold :用于图像阈值处理的一个重要函数。阈值处理是一种简单而有效的图像分割方法，它可以将图像中的像素根据其灰度值与设定的阈值进行比较，从而将像素分为不同的类别，常用于图像二值化等场景
    #cv2.THRESH_BINARY：二值化阈值处理。    
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    #cv2.findContours:在二值图像中找出对象的轮廓。它会返回两个值：轮廓信息和轮廓的层次结构信息
    #cv2.RETR_EXTERNAL：轮廓检索模式，它决定了函数查找轮廓的方式
    #cv2.CHAIN_APPROX_SIMPLE：轮廓逼近方法，它决定了如何存储轮廓的点。
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.contourArea:用于计算轮廓的面积
    max_contour = max(contours, key=cv2.contourArea)
    #cv2.boundingRect():用于计算轮廓的边界矩形。
    x, y, w, h = cv2.boundingRect(max_contour)
    return x, y, w, h

#图像加载和处理图像
def load_image(img_path, bg_color=None, rmbg_net=None, padding_ratio=0.1):
    #图像加载与初始检查
    #图像加载与调整尺寸
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return f"invalid image path {img_path}"

    #Alpha 通道有效性检查
    def is_valid_alpha(alpha, min_ratio = 0.01):
        bins = 20 #设置直方图的区间数量为 20
        if isinstance(alpha, np.ndarray): #判断 alpha 是否为 NumPy 数组。
            #使用cv2.calcHist函数计算直方图，[alpha]:输入的图像数组，[0]:使用第一个通道（这里只有一个通道），None:不使用掩码，[bins]:直方图的区间数量，[0, 256]:像素值的范围。
            hist = cv2.calcHist([alpha], [0], None, [bins], [0, 256])
        else:
            #使用torch.histc函数计算直方图，bins=bins:直方图的区间数量，min=0和max=1表示像素值的范围
            hist = torch.histc(alpha, bins=bins, min=0, max=1) 
        #计算直方图首尾区间内像素数量的最小阈值
        #alpha.shape[0] * alpha.shape[1] 表示 alpha 通道数据的总像素数量。
        min_hist_val = alpha.shape[0] * alpha.shape[1] * min_ratio
        #判断直方图的第一个区间（hist[0])和最后一个区间（hist[-1]）内的像素数量是否都大于等于最小阈值
        return hist[0] >= min_hist_val and hist[-1] >= min_hist_val
    
    #背景去除函数：这个函数调用外部提供的背景去除网络（rmbg_net），预处理图像并获取前景掩码
    def rmbg(image: torch.Tensor) -> torch.Tensor:
        image = TF.normalize(image, [0.5,0.5,0.5], [1.0,1.0,1.0]).unsqueeze(0)
        result=rmbg_net(image)
        return result[0][0]

    if len(img.shape) == 2:
        num_channels = 1
    else:
        num_channels = img.shape[2]

    # check if too large
    #检查并限制图像尺寸
    #加载图像，保留所有通道（包括alpha 通道）。检查图像是否过大，如果是，则等比例缩小到最大边长为 2000
    height, width = img.shape[:2]
    if height > width:
        scale = 2000 / height
    else:
        scale = 2000 / width
    if scale < 1:
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    if img.dtype != 'uint8':
        img = (img * (255. / np.iinfo(img.dtype).max)).astype(np.uint8)

    rgb_image = None
    alpha = None

    #图像格式处理
    #将所有格式转换为RGB
    #对于所有 RGBA图像，检查 alpha通道是否有效，如果有效则保留并将其转为 gpu 张量
    if num_channels == 1:  #灰度图
        rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif num_channels == 3:  #RGB图
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif num_channels == 4:  #RGBA图
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        #检查alpha通道是否有效
        b, g, r, alpha = cv2.split(img)
        if not is_valid_alpha(alpha):
            alpha = None
        else:
            alpha_gpu = torch.from_numpy(alpha).unsqueeze(0).cuda().float() / 255.
    else:
        return f"invalid image: channels {num_channels}"
    
    rgb_image_gpu = torch.from_numpy(rgb_image).cuda().float().permute(2, 0, 1) / 255.
    #自动背景去除
    if alpha is None:
        #调整大小并准备输入
        resize_transform = transforms.Resize((384, 384), antialias=True)
        rgb_image_resized = resize_transform(rgb_image_gpu)
        normalize_image = rgb_image_resized * 2 - 1

        mean_color = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
        resize_transform = transforms.Resize((1024, 1024), antialias=True)
        rgb_image_resized = resize_transform(rgb_image_gpu)
        #归一化处理
        max_value = rgb_image_resized.flatten().max()
        if max_value < 1e-3:
            return "invalid image: pure black image"
        normalize_image = rgb_image_resized / max_value - mean_color
        normalize_image = normalize_image.unsqueeze(0)
        resize_transform = transforms.Resize((rgb_image_gpu.shape[1], rgb_image_gpu.shape[2]), antialias=True)

        # seg from rmbg
        #运行背景去除网络
        alpha_gpu_rmbg = rmbg(rgb_image_resized)
        alpha_gpu_rmbg = alpha_gpu_rmbg.squeeze(0)
        #调整蒙版大小
        alpha_gpu_rmbg = resize_transform(alpha_gpu_rmbg)
        ma, mi = alpha_gpu_rmbg.max(), alpha_gpu_rmbg.min()
        alpha_gpu_rmbg = (alpha_gpu_rmbg - mi) / (ma - mi)

        alpha_gpu = alpha_gpu_rmbg
        
        alpha_gpu_tmp = alpha_gpu * 255
        alpha = alpha_gpu_tmp.to(torch.uint8).squeeze().cpu().numpy()
        #后处理蒙版
        _, alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        labeled_alpha = label(alpha)
        cleaned_alpha = remove_small_objects(labeled_alpha, min_size=200)
        cleaned_alpha = (cleaned_alpha > 0).astype(np.uint8)
        alpha = cleaned_alpha * 255
        alpha_gpu = torch.from_numpy(cleaned_alpha).cuda().float().unsqueeze(0)
        x, y, w, h = find_bounding_box(alpha)

    # If alpha is provided, the bounds of all foreground are used
    else: 
        #如果提供了alpha 通道，使用其边界
        #根据alpha通道或生成的蒙版计算前景对象的边界框
        rows, cols = np.where(alpha > 0)
        if rows.size > 0 and cols.size > 0:
            x_min = np.min(cols)
            y_min = np.min(rows)
            x_max = np.max(cols)
            y_max = np.max(rows)

            width = x_max - x_min + 1
            height = y_max - y_min + 1
        x, y, w, h = x_min, y_min, width, height

    if np.all(alpha==0):
        raise ValueError(f"input image too small")
    
    #背景替换和填充
    bg_gray = bg_color[0]
    bg_color = torch.from_numpy(bg_color).float().cuda().repeat(alpha_gpu.shape[1], alpha_gpu.shape[2], 1).permute(2, 0, 1)
    rgb_image_gpu = rgb_image_gpu * alpha_gpu + bg_color * (1 - alpha_gpu)
    #计算填充大小
    padding_size = [0] * 6
    if w > h:
        padding_size[0] = int(w * padding_ratio)
        padding_size[2] = int(padding_size[0] + (w - h) / 2)
    else:
        padding_size[2] = int(h * padding_ratio)
        padding_size[0] = int(padding_size[2] + (h - w) / 2)
    padding_size[1] = padding_size[0]
    padding_size[3] = padding_size[2]
    #应用填充
    padded_tensor = F.pad(rgb_image_gpu[:, y:(y+h), x:(x+w)], pad=tuple(padding_size), mode='constant', value=bg_gray)

    return padded_tensor


#对外接口函数
def prepare_image(image_path, bg_color, rmbg_net=None):
    if os.path.isfile(image_path):
        img_tensor = load_image(image_path, bg_color=bg_color, rmbg_net=rmbg_net)
        img_np = img_tensor.permute(1,2,0).cpu().numpy()
        img_pil = Image.fromarray((img_np*255).astype(np.uint8))
        
        return img_pil