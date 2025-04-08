#主要用于3D几何数据处理和推理相关任务，包含从生成密集网格点，寻找网格坐标，候选点选取，扩展边缘区域，数据缩放，到分层抽取几何形状的一系列功能。
import numpy as np
import torch
import scipy.ndimage
from skimage import measure

from triposg.utils.typing import *

#在给定边界框和八叉树深度下，生成密集的网格点集，返回网格点、网格大小和边界框长度，在GPU上运行。
def generate_dense_grid_points_gpu(bbox_min: torch.Tensor,  #一个torch.Tensor类型的张量，代表三维空间中边界框的最小坐标。
                                   bbox_max: torch.Tensor,  #一个torch.Tensor类型的张量，代表三维空间中边界框的最大坐标。
                                   octree_depth: int,  #一个整数，代表八叉树的深度，它会对网格点的数量产生影响。
                                   indexing: str = "ij"):  #一个字符串，默认值为"ij"，用于指定torch.meshgrid函数的索引方式。
    length = bbox_max - bbox_min #边界框在每个维度上的长度，也就是最大坐标与最小坐标的差值。
    #依据八叉树的深度计算每个维度上网格单元的数量。八叉树深度每增加 1，网格单元数量就会翻倍。
    num_cells = 2 ** octree_depth
    device = bbox_min.device
    
    #分别在 x、y、z 三个维度上生成等间距的坐标点。
    #torch.linspace函数会在指定的区间内生成指定数量的等间距点。
    #这里的数据类型为torch.float16，目的是减少内存占用。
    x = torch.linspace(bbox_min[0], bbox_max[0], int(num_cells), dtype=torch.float16, device=device)
    y = torch.linspace(bbox_min[1], bbox_max[1], int(num_cells), dtype=torch.float16, device=device)
    z = torch.linspace(bbox_min[2], bbox_max[2], int(num_cells), dtype=torch.float16, device=device)
    
    #利用torch.meshgrid函数生成三维网格。x、y、z分别作为三个维度上的坐标点，indexing参数用于指定索引方式。
    xs, ys, zs = torch.meshgrid(x, y, z, indexing=indexing)
    #把xs、ys、zs沿着最后一个维度进行堆叠，形成一个三维张量，其中每个元素代表一个三维坐标点。
    xyz = torch.stack((xs, ys, zs), dim=-1)
    xyz = xyz.view(-1, 3) #把三维张量xyz重塑成一个二维张量，每一行代表一个三维坐标点。
    grid_size = [int(num_cells), int(num_cells), int(num_cells)] #定义网格的大小，也就是每个维度上网格单元的数量。

    return xyz, grid_size, length

#在占用网格中快速定位满足邻域未占用条件的占用体素坐标，并可限制返回坐标数量。
#n_limits：一个整数，默认值为 -1。如果它不为 -1，且找到的符合条件的网格坐标数量超过n_limits，则会随机选取n_limits个坐标。
def find_mesh_grid_coordinates_fast_gpu(occupancy_grid, n_limits=-1):
    #从occupancy_grid中提取核心部分，去掉最外层的网格单元，这样做是为了后续方便检查邻居单元。
    core_grid = occupancy_grid[1:-1, 1:-1, 1:-1]
    #创建一个布尔型张量occupied，其中True表示该核心网格单元被占用（值大于 0）。
    occupied = core_grid > 0

    #
    neighbors_unoccupied = (
        (occupancy_grid[:-2, :-2, :-2] < 0)
        | (occupancy_grid[:-2, :-2, 1:-1] < 0)
        | (occupancy_grid[:-2, :-2, 2:] < 0)  # x-1, y-1, z-1/0/1
        | (occupancy_grid[:-2, 1:-1, :-2] < 0)
        | (occupancy_grid[:-2, 1:-1, 1:-1] < 0)
        | (occupancy_grid[:-2, 1:-1, 2:] < 0)  # x-1, y0, z-1/0/1
        | (occupancy_grid[:-2, 2:, :-2] < 0)
        | (occupancy_grid[:-2, 2:, 1:-1] < 0)
        | (occupancy_grid[:-2, 2:, 2:] < 0)  # x-1, y+1, z-1/0/1
        | (occupancy_grid[1:-1, :-2, :-2] < 0)
        | (occupancy_grid[1:-1, :-2, 1:-1] < 0)
        | (occupancy_grid[1:-1, :-2, 2:] < 0)  # x0, y-1, z-1/0/1
        | (occupancy_grid[1:-1, 1:-1, :-2] < 0)
        | (occupancy_grid[1:-1, 1:-1, 2:] < 0)  # x0, y0, z-1/1(occupancy_grid[:-2, :-2, :-2] < 0)
        | (occupancy_grid[:-2, :-2, 1:-1] < 0)
        | (occupancy_grid[:-2, :-2, 2:] < 0)  # x-1, y-1, z-1/0/1
        | (occupancy_grid[1:-1, 2:, :-2] < 0)
        | (occupancy_grid[1:-1, 2:, 1:-1] < 0)
        | (occupancy_grid[1:-1, 2:, 2:] < 0)  # x0, y+1, z-1/0/1
        | (occupancy_grid[2:, :-2, :-2] < 0)
        | (occupancy_grid[2:, :-2, 1:-1] < 0)
        | (occupancy_grid[2:, :-2, 2:] < 0)  # x+1, y-1, z-1/0/1
        | (occupancy_grid[2:, 1:-1, :-2] < 0)
        | (occupancy_grid[2:, 1:-1, 1:-1] < 0)
        | (occupancy_grid[2:, 1:-1, 2:] < 0)  # x+1, y0, z-1/0/1
        | (occupancy_grid[2:, 2:, :-2] < 0)
        | (occupancy_grid[2:, 2:, 1:-1] < 0)
        | (occupancy_grid[2:, 2:, 2:] < 0)  # x+1, y+1, z-1/0/1
    )
    #torch.nonzero()用于返回输入张量中非零元素的索引。
    #由于之前提取核心网格时去掉了最外层，所以这里要加 1 来恢复到原始网格的坐标。
    core_mesh_coords = torch.nonzero(occupied & neighbors_unoccupied, as_tuple=False) + 1

    if n_limits != -1 and core_mesh_coords.shape[0] > n_limits:
        print(f"core mesh coords {core_mesh_coords.shape[0]} is too large, limited to {n_limits}")
        ind = np.random.choice(core_mesh_coords.shape[0], n_limits, True) #使用np.random.choice函数从所有坐标中随机选取n_limits个坐标。
        core_mesh_coords = core_mesh_coords[ind]

    return core_mesh_coords  #返回符合条件的核心网格坐标。

#返回占用网格中SDF值绝对值小于给定阈值的体素坐标，可控制返回点数量。
#occupancy_grid (torch.Tensor)：一个三维的张量，其中的元素是 SDF 值，代表了空间中每个体素到最近表面的有符号距离。
#band_threshold (float)：一个浮点数，作为阈值。只有当体素的 SDF 值的绝对值小于该阈值时，该体素才会被考虑。
#n_limits (int)：一个整数，用于限制返回的坐标数量。如果设置为 -1，则不进行数量限制；如果设置为其他正整数，则最多返回该数量的坐标。
def find_candidates_band(occupancy_grid: torch.Tensor, band_threshold: float, n_limits: int = -1) -> torch.Tensor:
    """
    Returns the coordinates of all voxels in the occupancy_grid where |value| < band_threshold.

    Args:
        occupancy_grid (torch.Tensor): A 3D tensor of SDF values.
        band_threshold (float): The threshold below which |SDF| must be to include the voxel.
        n_limits (int): Maximum number of points to return (-1 for no limit)

    Returns:
        torch.Tensor: A 2D tensor of coordinates (N x 3) where each row is [x, y, z].
    """
    #从输入的 occupancy_grid 中提取核心部分，去掉了最外层的体素。目的：为了避免边界体素的干扰，或者是因为后续处理只需要关注内部的体素。
    core_grid = occupancy_grid[1:-1, 1:-1, 1:-1]  
    #将logits转换为SDF
    #将core_grid通过torch.sigmoid函数进行转换将其映射到[0, 1]区间，然后乘以2再减去1，将其进一步映射到 [-1, 1] 区间，从而得到 SDF 值。
    core_grid = torch.sigmoid(core_grid) * 2 - 1  
    #创建布尔掩码，用于标记哪些体素的 SDF 值的绝对值小于 band_threshold
    #torch.abs()用于计算输入张量中每个元素的绝对值。
    in_band = torch.abs(core_grid) < band_threshold

    #获取符合条件的体素坐标
    core_mesh_coords = torch.nonzero(in_band, as_tuple=False) + 1

    if n_limits != -1 and core_mesh_coords.shape[0] > n_limits:
        print(f"core mesh coords {core_mesh_coords.shape[0]} is too large, limited to {n_limits}")
        ind = np.random.choice(core_mesh_coords.shape[0], n_limits, True)
        core_mesh_coords = core_mesh_coords[ind]

    return core_mesh_coords 

#对边缘坐标进行扩展，先从低分辨率到高分辨率进行扩展计算。
#接收边缘坐标 edge_coords 和网格大小 grid_size 作为输入
def expand_edge_region_fast(edge_coords, grid_size):
    expanded_tensor = torch.zeros(grid_size, grid_size, grid_size, device='cuda', dtype=torch.float16, requires_grad=False)
    #将edge_coords所指定的位置在expanded_tensor中设置为1，这样就将边缘坐标信息映射到了张量中。
    expanded_tensor[edge_coords[:, 0], edge_coords[:, 1], edge_coords[:, 2]] = 1
    if grid_size < 512:
        kernel_size = 5
        #unsqueeze(0) 是为了在输入张量的第 0 维添加一个维度，以满足 max_pool3d 函数对输入维度的要求
        #squeeze() 用于移除张量中维度大小为 1 的维度
        pooled_tensor = torch.nn.functional.max_pool3d(expanded_tensor.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=2).squeeze()
    else:
        kernel_size = 3
        pooled_tensor = torch.nn.functional.max_pool3d(expanded_tensor.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=1).squeeze()
    expanded_coords_low_res = torch.nonzero(pooled_tensor, as_tuple=False).to(torch.int16)

    #torch.cat 用于将多个张量在指定维度上拼接起来。
    #torch.stack 用于在新的维度上堆叠多个张量。
    expanded_coords_high_res = torch.stack([
        torch.cat((expanded_coords_low_res[:, 0] * 2, expanded_coords_low_res[:, 0] * 2, expanded_coords_low_res[:, 0] * 2, expanded_coords_low_res[:, 0] * 2, expanded_coords_low_res[:, 0] * 2 + 1, expanded_coords_low_res[:, 0] * 2 + 1, expanded_coords_low_res[:, 0] * 2 + 1, expanded_coords_low_res[:, 0] * 2 + 1)),
        torch.cat((expanded_coords_low_res[:, 1] * 2, expanded_coords_low_res[:, 1] * 2, expanded_coords_low_res[:, 1] * 2+1, expanded_coords_low_res[:, 1] * 2 + 1, expanded_coords_low_res[:, 1] * 2, expanded_coords_low_res[:, 1] * 2, expanded_coords_low_res[:, 1] * 2 + 1, expanded_coords_low_res[:, 1] * 2 + 1)),
        torch.cat((expanded_coords_low_res[:, 2] * 2, expanded_coords_low_res[:, 2] * 2+1, expanded_coords_low_res[:, 2] * 2, expanded_coords_low_res[:, 2] * 2 + 1, expanded_coords_low_res[:, 2] * 2, expanded_coords_low_res[:, 2] * 2+1, expanded_coords_low_res[:, 2] * 2, expanded_coords_low_res[:, 2] * 2 + 1))
    ], dim=1)

    return expanded_coords_high_res  #返回扩展后的高分辨率坐标

#使用`scipy.ndimage.zoom`对数据块按比例缩放。
#block：输入的数组
#scale_factor：缩放因子，用于指定 block 数组在各个维度上的缩放比例。
#order：可选参数，默认值为 3。它指定了scipy.ndimage.zoom函数在进行缩放时所使用的插值方法的阶数。阶数越高，插值的精度越高，但计算量也会相应增加。
"""
order = 0:最近邻插值，速度最快，但精度相对较低，可能会出现锯齿状边缘。
order = 1:双线性插值，精度适中，计算速度也较快。
order = 3:三次样条插值，精度较高，能得到更平滑的结果，但计算量相对较大。
"""
def zoom_block(block, scale_factor, order=3):
    #.astype:常用于数据类型转换
    block = block.astype(np.float32)
    return scipy.ndimage.zoom(block, scale_factor, order=order)

#使用PyTorch的`interpolate`对输入张量（假设是占用网格）进行缩放。
def parallel_zoom(occupancy_grid, scale_factor):
    result = torch.nn.functional.interpolate(occupancy_grid.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor)
    #连续两次 squeeze(0) 操作，是将之前添加的两个额外维度移除，将五维的结果张量重新转换为三维张量，使其形状与输入的 occupancy_grid 维度一致，方便后续使用。
    return result.squeeze(0).squeeze(0)


#核心思想:先在低分辨率（粗粒度）下快速定位感兴趣区域（如表面附近的点），再在高分辨率（细粒度）下细化这些区域，避免对整个空间进行密集计算，提升效率。
@torch.no_grad()
def hierarchical_extract_geometry(geometric_func: Callable,
                     device: torch.device,
                     bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
                     dense_octree_depth: int = 8,
                     hierarchical_octree_depth: int = 9,
                     ):
    """

    Args:
        geometric_func:可调用对象。输入三维坐标点，返回对应的几何属性（如 SDF 值、占用概率、密度等）
        device:
        bounds:表示三维空间的边界。定义几何提取的三维空间范围
        dense_octree_depth:八叉树的 密集层深度，控制初始密集网格的分辨率。在粗粒度下生成初始密集网格，用于快速筛选表面附近的候选点。
        hierarchical_octree_depth:八叉树的 层次化层深度，控制细化阶段的分辨率。对粗粒度筛选出的区域进行高分辨率细化，提升几何细节精度。
    Returns:

    """
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

    bbox_min = torch.tensor(bounds[0:3]).to(device)
    bbox_max = torch.tensor(bounds[3:6]).to(device)
    bbox_size = bbox_max - bbox_min

    xyz_samples, grid_size, length = generate_dense_grid_points_gpu(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_depth=dense_octree_depth,
        indexing="ij"
    )
    

    #代码核心:通过多分辨率的方式对初始网格进行细化，逐步增加网格的分辨率以获取更精确的几何信息，最后使用 marching_cubes 算法从最终的网格数据中提取三维网格。
    print(f'step 1 query num: {xyz_samples.shape[0]}')
    #.view(grid_size[0], grid_size[1], grid_size[2])：将计算结果重新调整为三维网格的形状。
    grid_logits = geometric_func(xyz_samples.unsqueeze(0)).to(torch.float16).view(grid_size[0], grid_size[1], grid_size[2])
    # print(f'step 1 grid_logits shape: {grid_logits.shape}')
    # 多分辨率细化循环
    for i in range(hierarchical_octree_depth - dense_octree_depth):
        curr_octree_depth = dense_octree_depth + i + 1 #计算当前的八叉树深度
        # upsample
        grid_size = 2**curr_octree_depth #根据当前八叉树深度计算网格大小。
        normalize_offset = grid_size / 2 #计算归一化偏移量。
        #上采样
        high_res_occupancy = parallel_zoom(grid_logits, 2) #将当前的网格 logits 上采样为更高分辨率的网格。

        #查找候选边缘坐标
        band_threshold = 1.0  #设置带宽阈值
        edge_coords = find_candidates_band(grid_logits, band_threshold) #查找满足带宽阈值的边缘坐标。
        #扩展边缘区域，得到扩展后的坐标。
        expanded_coords = expand_edge_region_fast(edge_coords, grid_size=int(grid_size/2)).to(torch.float16)
        print(f'step {i+2} query num: {len(expanded_coords)}')
        #对扩展后的坐标进行归一化处理
        expanded_coords_norm = (expanded_coords - normalize_offset) * (abs(bounds[0]) / normalize_offset)

        all_logits = None

        #计算扩展坐标的 logits
        all_logits = geometric_func(expanded_coords_norm.unsqueeze(0)).to(torch.float16)
        all_logits = torch.cat([expanded_coords_norm, all_logits[0]], dim=1)
        # print("all logits shape = ", all_logits.shape)

        indices = all_logits[..., :3]
        #将坐标反归一化。
        indices = indices * (normalize_offset / abs(bounds[0]))  + normalize_offset
        indices = indices.type(torch.IntTensor)
        values = all_logits[:, 3]
        # breakpoint()
        #将计算得到的 logits 值更新到高分辨率网格中。
        high_res_occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        grid_logits = high_res_occupancy #更新当前的网格 logits。
        torch.cuda.empty_cache() #清空 CUDA 缓存，释放内存。

    #提取三维网格
    mesh_v_f = []
    try:
        print("final grids shape = ", grid_logits.shape)
        vertices, faces, normals, _ = measure.marching_cubes(grid_logits.float().cpu().numpy(), 0, method="lewiner")
        vertices = vertices / (2**hierarchical_octree_depth) * bbox_size.cpu().numpy() + bbox_min.cpu().numpy()
        mesh_v_f = (vertices.astype(np.float32), np.ascontiguousarray(faces))
    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        mesh_v_f = (None, None)

    return [mesh_v_f]