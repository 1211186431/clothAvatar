import torch
import torch.nn as nn
import kaolin as kal
from tqdm import tqdm
import random
import trimesh
import kaolin
from .network_utils import Decoder, HashDecoder
# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

# def loss_f(mesh_verts, mesh_faces, points, it):
#     pred_points = kal.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
#     chamfer = kal.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
#     laplacian_weight = 0.1
#     if it > iterations//2:
#         lap = laplace_regularizer_const(mesh_verts, mesh_faces)
#         return chamfer + lap * laplacian_weight
#     return chamfer

###############################################################################
# Compact tet grid
###############################################################################

def compact_tets(pos_nx3, sdf_n, tet_fx4):
    with torch.no_grad():
        # Find surface tets
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)  # one value per tet, these are the surface tets

        valid_vtx = tet_fx4[valid_tets].reshape(-1)
        unique_vtx, idx_map = torch.unique(valid_vtx, dim=0, return_inverse=True)
        new_pos = pos_nx3[unique_vtx]
        new_sdf = sdf_n[unique_vtx]
        new_tets = idx_map.reshape(-1, 4)
        return new_pos, new_sdf, new_tets
def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff



class FlexiMesh(nn.Module):
    def __init__(self, device: str='cuda', use_explicit=False, geo_network='mlp', hash_max_res=1024, hash_num_levels=16, num_subdiv=0,voxel_grid_res=128) -> None:
        super().__init__()
        self.device = device
        self.use_explicit = use_explicit

        self.num_subdiv = num_subdiv
        
        self.voxel_grid_res = voxel_grid_res
        self.fc = kaolin.non_commercial.FlexiCubes(device)
        x_nx3, cube_fx8 = self.fc.construct_voxel_grid(self.voxel_grid_res)
        self.all_edges = cube_fx8[:, self.fc.cube_edges].reshape(-1, 2)
        self.grid_edges = torch.unique(self.all_edges, dim=0)
        self.x_nx3 = x_nx3 * 1.1
        self.cube_fx8 = cube_fx8
        weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
        self.weight= torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
        self.sdf_regularizer = 0.2
        if self.use_explicit:
            self.sdf = nn.Parameter(torch.zeros_like(self.x_nx3[:, 0]), requires_grad=True)
            self.deform = nn.Parameter(torch.zeros_like(self.x_nx3), requires_grad=True)
        elif geo_network == 'mlp':
            self.decoder = Decoder().to(device)
        elif geo_network == 'hash':
            pts_bounds = (self.x_nx3.min(dim=0)[0], self.x_nx3.max(dim=0)[0])
            self.decoder = HashDecoder(input_bounds=pts_bounds, max_res=hash_max_res, num_levels=hash_num_levels).to(device)
        
        

    def query_decoder(self, tet_v):
        if tet_v.shape[0] < 1000000:
            return self.decoder(tet_v)
        else:
            chunk_size = 1000000
            results = []
            for i in range((tet_v.shape[0] // chunk_size) + 1):
                if i*chunk_size < tet_v.shape[0]:
                    results.append(self.decoder(tet_v[i*chunk_size: (i+1)*chunk_size]))
            return torch.cat(results, dim=0)

    def get_mesh(self, return_loss=False, num_subdiv=None,t_iter=0.5):
        if num_subdiv is None:
            num_subdiv = self.num_subdiv
        if self.use_explicit:
            sdf = self.sdf * 1
            deform = self.deform * 1
        else:
            pred = self.query_decoder(self.x_nx3)
            sdf, deform = pred[:,0], pred[:,1:]
        grid_verts = self.x_nx3 + (2-1e-8) / (self.voxel_grid_res * 2) * torch.tanh(deform)
        mesh_verts, mesh_faces, L_dev = self.fc(grid_verts, sdf, self.cube_fx8, self.voxel_grid_res, beta=self.weight[:,:12], alpha=self.weight[:,12:20],gamma_f=self.weight[:,20], training=True)
        reg_loss = self.regularizer_loss(t_iter,sdf,L_dev)
        return mesh_verts, mesh_faces, reg_loss
    
    def sdf_loss(self,sample_num):
        indices = torch.randint(0, self.x_nx3.shape[0], (sample_num,))
        sample_points = self.x_nx3[indices]
        gt_sdf = self.gt_sdf[indices]
        pred = self.decoder(sample_points)
        pred_sdf, deform = pred[:,0], pred[:,1:]
        loss = nn.functional.l1_loss(pred_sdf, gt_sdf, reduction='sum')
        return loss
    
    def regularizer_loss(self,t_iter,sdf,L_dev):
        sdf_weight = self.sdf_regularizer - (self.sdf_regularizer - self.sdf_regularizer/20)*min(1.0, 4.0 * t_iter)
        reg_loss = sdf_reg_loss(sdf, self.grid_edges).mean() * sdf_weight # Loss to eliminate internal floaters that are not visible
        reg_loss += L_dev.mean() * 0.5
        reg_loss += (self.weight[:,:20]).abs().mean() * 0.1
        return reg_loss
        

    
    def set_sdf(self,mesh_v,mesh_f):
        mesh = trimesh.Trimesh(mesh_v.detach().cpu().numpy(), mesh_f.detach().cpu().numpy())
        import mesh_to_sdf
        sdf_tet = torch.tensor(mesh_to_sdf.mesh_to_sdf(mesh, self.x_nx3.cpu().numpy()), dtype=torch.float32).to(self.device)
        self.gt_sdf = sdf_tet
        return sdf_tet
    
    def init_mesh(self, mesh_v, mesh_f, init_padding=0.):
        num_pts = self.x_nx3
        mesh = trimesh.Trimesh(mesh_v.cpu().numpy(), mesh_f.cpu().numpy())
        import mesh_to_sdf
        sdf_tet = self.set_sdf(mesh_v,mesh_f)
        if self.use_explicit:
            self.sdf.data[...] = sdf_tet[...]
        else:
            optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
            batch_size = 300000
            iter = 200
            points, sdf_gt = mesh_to_sdf.sample_sdf_near_surface(mesh) 
            points = torch.tensor(points, dtype=torch.float32).to(self.device)
            sdf_gt = torch.tensor(sdf_gt, dtype=torch.float32).to(self.device)
            points = torch.cat([points, self.x_nx3], dim=0)
            sdf_gt = torch.cat([sdf_gt, sdf_tet], dim=0)
            num_pts = len(points)
            for i in tqdm(range(iter)):
                sampled_ind = random.sample(range(num_pts), min(batch_size, num_pts))
                p = points[sampled_ind]
                pred = self.decoder(p)
                sdf, deform = pred[:,0], pred[:,1:]
                loss = nn.functional.mse_loss(sdf, sdf_gt[sampled_ind])# + (deform ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                # mesh_v, mesh_f = self.get_lower_mesh(x_nx3=x_nx3,cube_fx8=cube_fx8,voxel_grid_res=64,weight=weight)
                mesh_v, mesh_f,_ = self.get_mesh()
            pred_mesh = trimesh.Trimesh(mesh_v.cpu().numpy(), mesh_f.cpu().numpy())
            print('fitted mesh with num_vertex {}, num_faces {}'.format(mesh_v.shape[0], mesh_f.shape[0]))