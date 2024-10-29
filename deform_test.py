import trimesh
import torch
import torch.nn as nn
from lib.model.smplx_deformer import SMPLXDeformer
import torch.optim as optim
def arap_loss(vertices, original_vertices, neighbors):
    """
    ARAP (As-Rigid-As-Possible) loss with vectorized operations for better performance.
    
    Args:
        vertices (torch.Tensor): Deformed vertices after skinning. Shape: [N, 3]
        original_vertices (torch.Tensor): Original vertices in canonical pose. Shape: [N, 3]
        neighbors (list of lists): List where each entry contains the neighboring vertex indices.

    Returns:
        loss (torch.Tensor): ARAP loss value.
    """
    # Create a mask for the neighborhood indices
    vertex_indices = torch.arange(vertices.shape[0], device=vertices.device)
    
    # Prepare storage for all pairs of vertices and their neighbors
    losses = []

    # Iterate over each vertex and its neighbors
    for i, neighbor_indices in enumerate(neighbors):
        # Get the coordinates of the current vertex and its neighbors (original and deformed)
        v_current = vertices[i]
        v_neighbors = vertices[neighbor_indices]
        
        v_orig_current = original_vertices[i]
        v_orig_neighbors = original_vertices[neighbor_indices]
        
        # Compute the distance differences
        deformed_distances = torch.norm(v_current - v_neighbors, dim=-1)
        original_distances = torch.norm(v_orig_current - v_orig_neighbors, dim=-1)
        
        # Compute the ARAP loss for this vertex's neighborhood
        loss_per_neighbor = (deformed_distances - original_distances) ** 2
        losses.append(loss_per_neighbor)

    # Combine all the losses into a single tensor and sum them
    total_loss = torch.cat(losses).sum()

    return total_loss

tfs = torch.load("/home/mycode3/t1022/smpl_tfs.pt")
smplx_d = trimesh.load("/home/clothAvatar/data/template/smplx_d.obj")
smplx_d_v = torch.tensor(smplx_d.vertices).float().cuda()
smplx_c = trimesh.load("/home/clothAvatar/data/template/smplx_c.obj")
smplx_c_v = torch.tensor(smplx_c.vertices).float().cuda()

scan = trimesh.load("/home/clothAvatar/data/scan.obj")
scan_v = torch.tensor(scan.vertices).float().cuda()
deformer = SMPLXDeformer()

x,outlier_mask = deformer.forward(scan_v, smpl_verts=smplx_d_v, inverse=True, smpl_tfs=tfs,return_weights=False)

neighbors = scan.vertex_neighbors
offset = nn.Parameter(x).cuda()
optimizer = optim.Adam([offset], lr=1e-3)
num_epochs = 200
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # 加入 offset 学习顶点位置
    scan_c = offset

    # # 正向形变，得到形变后的顶点
    scan_d, _ = deformer.forward(scan_c, smpl_verts=smplx_c_v, inverse=False, smpl_tfs=tfs, return_weights=False)

    # 计算 ARAP 损失
    loss = arap_loss(scan_c, scan_v, neighbors)
    loss += 0.1*nn.functional.l1_loss(scan_d, scan_v, reduction='sum')
    # 反向传播并更新 offset
    loss.backward()
    optimizer.step()

    # 打印损失
    # if epoch % 100 == 0:
    print(f"Epoch {epoch}, Loss: {loss.item()}")
    print(f"offset: {offset.max()}")


## 可视化
trimesh.Trimesh(vertices=scan_c.detach().cpu().numpy(), faces=scan.faces).export("/home/mycode3/t1022/scan_deformed.obj")
