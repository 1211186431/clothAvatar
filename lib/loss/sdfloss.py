# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch
import torch_scatter

###############################################################################
# Pytorch implementation of the developability regularizer introduced in paper 
# "Developability of Triangle Meshes" by Stein et al.
###############################################################################
def mesh_developable_reg(mesh):

    verts = mesh.vertices
    tris = mesh.faces

    device = verts.device
    V = verts.shape[0]
    F = tris.shape[0]

    POS_EPS = 1e-6
    REL_EPS = 1e-6

    def normalize(vecs):
        return vecs / (torch.linalg.norm(vecs, dim=-1, keepdim=True) + POS_EPS)

    tri_pos = verts[tris]

    vert_normal_covariance_sum = torch.zeros((V, 9), device=device)
    vert_area = torch.zeros(V, device=device)
    vert_degree = torch.zeros(V, dtype=torch.int32, device=device)

    for iC in range(3):  # loop over three corners of each triangle

        # gather tri verts
        pRoot = tri_pos[:, iC, :]
        pA = tri_pos[:, (iC + 1) % 3, :]
        pB = tri_pos[:, (iC + 2) % 3, :]

        # compute the corner angle & normal
        vA = pA - pRoot
        vAn = normalize(vA)
        vB = pB - pRoot
        vBn = normalize(vB)
        area_normal = torch.linalg.cross(vA, vB, dim=-1)
        face_area = 0.5 * torch.linalg.norm(area_normal, dim=-1)
        normal = normalize(area_normal)
        corner_angle = torch.acos(torch.clamp(torch.sum(vAn * vBn, dim=-1), min=-1., max=1.))

        # add up the contribution to the covariance matrix
        outer = normal[:, :, None] @ normal[:, None, :]
        contrib = corner_angle[:, None] * outer.reshape(-1, 9)

        # scatter the result to the appropriate matrices
        vert_normal_covariance_sum = torch_scatter.scatter_add(src=contrib,
                                                               index=tris[:, iC],
                                                               dim=-2,
                                                               out=vert_normal_covariance_sum)

        vert_area = torch_scatter.scatter_add(src=face_area / 3.,
                                              index=tris[:, iC],
                                              dim=-1,
                                              out=vert_area)

        vert_degree = torch_scatter.scatter_add(src=torch.ones(F, dtype=torch.int32, device=device),
                                                index=tris[:, iC],
                                                dim=-1,
                                                out=vert_degree)

    # The energy is the smallest eigenvalue of the outer-product matrix
    vert_normal_covariance_sum = vert_normal_covariance_sum.reshape(
        -1, 3, 3)  # reshape to a batch of matrices
    vert_normal_covariance_sum = vert_normal_covariance_sum + torch.eye(
        3, device=device)[None, :, :] * REL_EPS

    min_eigvals = torch.min(torch.linalg.eigvals(vert_normal_covariance_sum).abs(), dim=-1).values

    # Mask out degree-3 vertices
    vert_area = torch.where(vert_degree == 3, torch.tensor(0, dtype=vert_area.dtype,device=vert_area.device), vert_area)

    # Adjust the vertex area weighting so it is unit-less, and 1 on average
    vert_area = vert_area * (V / torch.sum(vert_area, dim=-1, keepdim=True))

    return vert_area * min_eigvals 

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff


def compute_edge_to_face_mapping(attr_idx):
    with torch.no_grad():
        # Get unique edges
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # Compute edge to face table
        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge

@torch.cuda.amp.autocast(enabled=False)
def normal_consistency(face_normals, t_pos_idx):
    face_normals_length = torch.norm(face_normals, dim=1, keepdim=True)

    # 过滤退化的三角形，如果法线模长非常小（如接近零），设置为默认法线
    #threshold = 1e-6  # 设定一个阈值来判断退化三角形
    #face_normals = torch.where(face_normals_length > threshold, face_normals, torch.tensor([0.0, 0.0, 1.0], device=face_normals.device))

    # 对法线进行归一化处理，确保法线向量是单位长度
    face_normals = face_normals / (face_normals_length + 1e-8)  # 避免除以零
    tris_per_edge = compute_edge_to_face_mapping(t_pos_idx)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(torch.sum(n0 * n1, -1, keepdim=True), min=-1.0, max=1.0)
    term = (1.0 - term)

    return torch.mean(torch.abs(term))


def laplacian_uniform(verts, faces):

    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()


@torch.cuda.amp.autocast(enabled=False)
def laplacian_smooth_loss(verts, faces):
    with torch.no_grad():
        L = laplacian_uniform(verts, faces.long())
    loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss