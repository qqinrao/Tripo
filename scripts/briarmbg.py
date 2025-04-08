#背景去除模型
"""
定义 BriaRMBG 类
用于从输入图像中分离前景对象
"""
"""
Source and Copyright Notice:
This code is from briaai/RMBG-1.4
Original repository: https://huggingface.co/briaai/RMBG-1.4
Copyright belongs to briaai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

class REBNCONV(nn.Module):
    #dirate=1：扩张率默认为 1。扩张卷积用于增加卷积核的感受而不增加计算复杂度
    #stride=1：卷积的步长，默认值为 1
    def __init__(self,in_ch=3,out_ch=3,dirate=1,stride=1):
        #调用了 REBNCONV 类的父类的构造函数。确保父类被正确初始化。
        super(REBNCONV,self).__init__()

        #dilation=1*dirate：设置了扩张卷积的扩张率。扩张率为 1 时，表示标准卷积；大于 1 时，表示扩张卷积
        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate,stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        #inplace=True 表示直接在原地修改数据，节省内存消耗。
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
#将src（源张量）的空间维度（通常是图像的高度和宽度）调整到与tar（目标张量）相同的空间维度。
def _upsample_like(src,tar):

    #F.interpolate:主要用于对输入的张量进行插值操作。'bilinear'（双线性插值）
    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU7,self).__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1) ## 1 -> 1/2

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        b, c, h, w = x.shape

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        #编码器路径（下采样）
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        #中间层
        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        #解码器路径（上采样+跳跃连接）
        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin  #残差连接
"""
通过残差连接将初始特征与解码器输出相加，这有助于:
缓解梯度消失问题
使网络能学习残差映射而非完整映射
增强特征传播和模型表达能力
"""
### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()
        #初始转换层
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        #编码器路径（下采样）-3 层池化
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        #瓶颈层
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        #解码器路径（上采样）
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)  #初始特征转换

        # 编码路径（逐步降采样）
        hx1 = self.rebnconv1(hxin) #第一次下采样
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)  #第二次下采样
        hx = self.pool2(hx2)
 
        hx3 = self.rebnconv3(hx)  #第三次下采样
        hx = self.pool3(hx3)

        # 瓶颈处理
        hx4 = self.rebnconv4(hx)
        # 解码路径（逐步上采样 + 跳跃连接
        hx5 = self.rebnconv5(hx4)  #扩张卷积增大感受野

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1)) # 连接特征
        hx4dup = _upsample_like(hx4d,hx3)  # 上采样匹配尺寸

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin
"""
关键步骤分析：
特征提取与下采样：

逐层提取特征并降低分辨率
每次下采样后分辨率减半，通道数保持不变（都是mid_ch）
下采样用最大池化实现（nn.MaxPool2d）
特征融合：

使用torch.cat沿通道维度连接特征
每个解码器层接收两个输入：上一层上采样后的特征和编码器同级特征
通道数加倍（变为mid_ch*2）后再通过卷积降回mid_ch
上采样技术：

使用_upsample_like函数，基于双线性插值
精确匹配目标特征图尺寸
残差连接：

最终输出hx1d + hxin，建立短路连接
实现梯度畅通，促进信息流动，减轻梯度消失问题
"""
### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        #在最深层使用dirate=2的扩张卷积，在保持参数量不变的情况下扩大感受野
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)
        # 编码器路径
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)  # 第一次下采样

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)  # 第二次下采样

        hx3 = self.rebnconv3(hx) # 最底层特征处理

        hx4 = self.rebnconv4(hx3)  # 使用扩张卷积进一步处理

        # 解码器路径
        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1)) # 特征融合
        hx3dup = _upsample_like(hx3d,hx2)  # 上采样

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin  # 残差连接

### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        # 编码器部分 - 没有池化，使用递增的扩张率
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)
        # 瓶颈层 - 最大扩张率
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)
        # 解码器部分 - 对应递减的扩张率
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)  # 初始特征转换

        # 编码器路径 - 递增扩张率，无池化
        hx1 = self.rebnconv1(hxin)   # dirate=1
        hx2 = self.rebnconv2(hx1)   # dirate=2 
        hx3 = self.rebnconv3(hx2)    # dirate=4

        # 瓶颈层
        hx4 = self.rebnconv4(hx3)  # dirate=8，最大感受野

        # 解码器路径 - 特征融合，递减扩张率
        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))  # 连接特征，dirate=4
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))   # dirate=2
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))   # dirate=1

        return hx1d + hxin  # 残差连接


class myrebnconv(nn.Module):
    def __init__(self, in_ch=3,
                       out_ch=1,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       dilation=1,  #扩张率，默认为1（标准卷积）
                       groups=1):  #分组卷积的组数，默认为1（标准卷积）
        super(myrebnconv,self).__init__()

        #卷积层: 提取特征
        self.conv = nn.Conv2d(in_ch,
                              out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_ch) #批归一化: 规范化特征
        self.rl = nn.ReLU(inplace=True)  #ReLU激活: 引入非线性

    def forward(self,x):
        return self.rl(self.bn(self.conv(x)))



"""
整体结构包含三个主要部分：
1、编码器路径（下采样提取特征）
2、解码器路径（上采样恢复细节）
3、侧输出模块（实现深度监督）
"""
class BriaRMBG(nn.Module, PyTorchModelHubMixin):

    def __init__(self,config:dict={"in_ch":3,"out_ch":1}):
        super(BriaRMBG,self).__init__()
        in_ch=config["in_ch"]
        out_ch=config["out_ch"]
        ## 初始下采样
        self.conv_in = nn.Conv2d(in_ch,64,3,stride=2,padding=1)
        self.pool_in = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        
        # 编码器阶段 - 逐渐减少深度的RSU模块
        self.stage1 = RSU7(64,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # 解码器阶段 - 对称的RSU模块结构
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        # 侧输出卷积层
        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        # self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x
        
        #初始化下采样
        hxin = self.conv_in(hx)
        #hx = self.pool_in(hxin)

        # 编码器部分 - 逐步下采样
        #stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # 最深层特征
        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        # 解码器部分 - 逐步上采样并融合特征
        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output 侧输出 - 所有预测上采样到原始尺寸
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1,x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,x)

        # 返回所有侧输出(经过sigmoid激活)和中间特征
        return [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)],[hx1d,hx2d,hx3d,hx4d,hx5d,hx6]

"""
关键步骤：
编码阶段：逐步降低分辨率，增加通道数，提取深层特征
特征连接：使用torch.cat将解码器特征与对应编码器特征连接
解码阶段：逐步恢复分辨率，融合多级特征
多输出：生成6个侧输出，每个都上采样到原始尺寸
激活处理：使用sigmoid函数将输出映射到[0,1]范围，适合二分类蒙版
"""