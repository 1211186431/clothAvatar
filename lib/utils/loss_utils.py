
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import torch.nn as nn
import torchvision.models as models
def crop_by_mask(rgb, alpha, base_size=64):
    mask = (alpha[0,0] > 0).float()
    h, w = mask.shape
    y = torch.arange(0, h, dtype=torch.float).to(mask)
    x = torch.arange(0, w, dtype=torch.float).to(mask)
    y, x = torch.meshgrid(y, x)
    x_max = int((mask * x).view(-1).max(-1)[0])
    x_min = int(w - (mask * (w-x)).view(-1).max(-1)[0])
    y_max = int((mask * y).view(-1).max(-1)[0])
    y_min = int(h - (mask * (h-y)).view(-1).max(-1)[0])
    if (x_max - x_min) % base_size > 0:
        x_max = min(x_max + base_size - ((x_max - x_min) % base_size), w-1)
        if (x_max - x_min) % base_size > 0:
            x_min = max(x_min - base_size + ((x_max - x_min) % base_size), 0)
    if (y_max - y_min) % base_size > 0:
        y_max = min(y_max + base_size - ((y_max - y_min) % base_size), h-1)
        if (y_max - y_min) % base_size > 0:
            y_min = max(y_min - base_size + ((y_max - y_min) % base_size), 0)
    #print(y_min, y_max, x_min, x_max)
    return rgb[:, :, y_min:y_max, x_min:x_max], alpha[:, :, y_min:y_max, x_min:x_max]

def silhouette_loss(alpha, gt_mask, edt=None, loss_mask=None, kernel_size=7, edt_power=0.25, l2_weight=0.01, edge_weight=0.01):
    """
    Inputs:
        alpha: Bx1xHxW Tensor, predicted alpha,
        gt_mask: Bx1xHxW Tensor, ground-truth mask
        loss_mask[Optional]: Bx1xHxW Tensor, loss mask, calculate loss inside the mask only
        kernel_size: edge filter kernel size
        edt_power: edge distance power in the loss
        l2_weight: loss weight of the l2 loss
        edge_weight: loss weight of the edge loss
    Output:
        loss
    """
    sil_l2loss = (gt_mask - alpha) ** 2
    if loss_mask is not None:
        sil_l2loss = sil_l2loss * loss_mask
    def compute_edge(x):
        return F.max_pool2d(x, kernel_size, 1, kernel_size // 2) - x
    if edt is None:
        gt_edge = compute_edge(gt_mask).cpu().numpy()
        edt = torch.tensor(distance_transform_edt(1 - (gt_edge > 0)) ** (edt_power * 2), dtype=torch.float32, device=gt_mask.device)
    if loss_mask is not None:
        pred_edge = pred_edge * loss_mask
    pred_edge = compute_edge(alpha)
    sil_edgeloss = torch.sum(pred_edge * edt.to(pred_edge.device)) / (pred_edge.sum()+1e-7)
    return sil_l2loss.mean() * l2_weight + sil_edgeloss * edge_weight

def get_edt(gt_mask, loss_mask=None, kernel_size=7, edt_power=0.25, l2_weight=0.01, edge_weight=0.01):
    def compute_edge(x):
        return F.max_pool2d(x, kernel_size, 1, kernel_size // 2) - x
    gt_edge = compute_edge(gt_mask).cpu().numpy()
    edt = torch.tensor(distance_transform_edt(1 - (gt_edge > 0)) ** (edt_power * 2), dtype=torch.float32, device=gt_mask.device)
    return edt


def laplacian_cot(verts, faces):
    """
    Compute the cotangent laplacian
    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """

    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    return L


def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

def laplacian_smooth_loss(v_pos, t_pos_idx):
    term = torch.zeros_like(v_pos)
    norm = torch.zeros_like(v_pos[..., 0:1])

    v0 = v_pos[t_pos_idx[:, 0], :]
    v1 = v_pos[t_pos_idx[:, 1], :]
    v2 = v_pos[t_pos_idx[:, 2], :]

    term.scatter_add_(0, t_pos_idx[:, 0:1].repeat(1, 3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, t_pos_idx[:, 1:2].repeat(1, 3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, t_pos_idx[:, 2:3].repeat(1, 3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, t_pos_idx[:, 0:1], two)
    norm.scatter_add_(0, t_pos_idx[:, 1:2], two)
    norm.scatter_add_(0, t_pos_idx[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term ** 2)



# 定义下采样函数
def downsample_image(image, scale_factor):
    """
    使用双线性插值对图像进行下采样。
    Args:
        image (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
        scale_factor (float): 缩放因子，<1 表示下采样。

    Returns:
        torch.Tensor: 下采样后的图像张量。
    """
    return F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=False)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(vgg[:23])).eval().to('cuda:0')
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.features(x)
        y_vgg = self.features(y)
        loss = torch.mean((x_vgg - y_vgg) ** 2)
        return loss

perceptualloss = PerceptualLoss()

def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    grid = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    grid /= grid.sum()
    return grid.reshape(1, 1, size, 1) * grid.reshape(1, 1, 1, size)

def ssim(x, y, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2, size_average=True):
    window = gaussian_window(window_size, sigma).to(x.device)
    
    mu_x = F.conv2d(x, window, padding=window_size//2, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=y.shape[1])
    
    sigma_x = F.conv2d(x**2, window, padding=window_size//2, groups=x.shape[1]) - mu_x**2
    sigma_y = F.conv2d(y**2, window, padding=window_size//2, groups=y.shape[1]) - mu_y**2
    sigma_xy = F.conv2d(x*y, window, padding=window_size//2, groups=x.shape[1]) - mu_x*mu_y
    
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)  # returns a batch of SSIM scores
def img_loss(pred_img, gt_img):
    l2_loss = F.mse_loss(pred_img, gt_img)
    l1_loss = F.l1_loss(pred_img, gt_img)
    x = pred_img
    y = gt_img
    ssim_r = ssim(x[:, 0:1, :, :], y[:, 0:1, :, :])
    ssim_g = ssim(x[:, 1:2, :, :], y[:, 1:2, :, :])
    ssim_b = ssim(x[:, 2:3, :, :], y[:, 2:3, :, :])

    # Average SSIM over all channels
    ssim_avg = (ssim_r + ssim_g + ssim_b) / 3
    perceptual_loss = perceptualloss(x, y)
    loss = 100*l1_loss+(1-ssim_avg)*50 + 5*perceptual_loss
    # loss = l1_loss+(1-ssim_avg)*1 + perceptual_loss
    return loss


def calculate_rgb_depth_loss(pred_rgb, pred_depth, pred_rgb_gt, pred_depth_gt, rgb_weight=0.5, depth_weight=0.5):
    """
    Inputs:
        pred_rgb: Bx3xHxW Tensor, predicted RGB image
        pred_depth: Bx1xHxW Tensor, predicted depth map
        pred_rgb_gt: Bx3xHxW Tensor, ground truth RGB image
        pred_depth_gt: Bx1xHxW Tensor, ground truth depth map
        rgb_weight: weight for the RGB loss
        depth_weight: weight for the depth loss
    Output:
        loss: combined loss for RGB and depth
    """
    # RGB loss (L2 loss)
    rgb_loss = (pred_rgb - pred_rgb_gt) ** 2
    rgb_loss = rgb_loss.sum()

    # Depth loss (L2 loss)
    depth_loss = (pred_depth - pred_depth_gt) ** 2
    depth_loss = depth_loss.sum()

    # Weighted sum of RGB and depth losses
    total_loss = rgb_loss * rgb_weight + depth_loss * depth_weight
    
    return total_loss