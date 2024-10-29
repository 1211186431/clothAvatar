from .deformer import ForwardDeformer, skinning
import numpy as np
import torch
import torch.nn as nn

import pickle as pkl
import sys
sys.path.append('/home/clothAvatar')
from lib.data.obj import Mesh
from lib.smpl import SMPLXServer

def load_smplx_data(smplx_path):
    f = pkl.load(open(smplx_path, 'rb'), encoding='latin1')
    betas = f['betas']
    num_hand_pose = 12
    smpl_params = np.zeros(99+2*num_hand_pose)
    smpl_params[0] = 1
    smpl_params[1:4] = f['transl']
    smpl_params[4:7] = f['global_orient']
    smpl_params[7:70] = f['body_pose']
    smpl_params[70:70+num_hand_pose] = f['left_hand_pose']
    smpl_params[70+num_hand_pose:70+2*num_hand_pose] = f['right_hand_pose']
    smpl_params[70+2*num_hand_pose:73+2*num_hand_pose] = np.zeros(3)
    smpl_params[73+2*num_hand_pose:76+2*num_hand_pose] = np.zeros(3)
    smpl_params[76+2*num_hand_pose:79+2*num_hand_pose] = f['jaw_pose']
    smpl_params[79+2*num_hand_pose:89+2*num_hand_pose] = betas
    smpl_params[89+2*num_hand_pose:99+2*num_hand_pose] = f['expression']
    smpl_params= torch.tensor(smpl_params).unsqueeze(0).float().cuda()
    return smpl_params,betas,num_hand_pose

class deform_mesh(nn.Module):
    def __init__(self,cfg=None,scale_def=None,scale_c=None):
        super().__init__()
        smplx_path = "/home/clothAvatar/data/mesh-f00011_smplx.pkl"
        smplx_model_path = "/home/clothAvatar/lib/smplx/smplx_model"
        init_model = init_model = torch.load("/home/clothAvatar/data/occ_init_male.pth")
        smplx_params,betas,num_hand_pose = load_smplx_data(smplx_path)
        smplx_server = SMPLXServer(gender="male",
                                betas=betas,
                                use_pca=True,
                                flat_hand_mean=False,       
                                num_pca_comps=12,model_path=smplx_model_path)
        smplx_data = smplx_server.forward(smplx_params, absolute=False)
        self.smpl_tfs = smplx_data['smpl_tfs']
        self.deformer = ForwardDeformer(d_out=59,model_type='smplx').to("cuda:0")
        self.deformer.load_state_dict(init_model['deformer_state_dict'])
        for p in self.deformer.parameters():
            p.requires_grad = False
        self.scale_def = scale_def
        self.scale_c = scale_c


    def forward(self, mesh, def_scale=False):
        verts = mesh.v
        faces = mesh.f
        v_homogeneous = torch.cat([verts, torch.ones(verts.shape[0], 1, device="cuda:0",requires_grad=True)], dim=1)
        restored_v = (self.scale_c['resize_matrix_inv'] @ v_homogeneous.T).T[:, :3]
        weights = self.deformer.query_weights(restored_v[None],
                                    None).clamp(0, 1)[0]
        verts_mesh_deformed = skinning(restored_v.unsqueeze(0),
                                        weights.unsqueeze(0),
                                        self.smpl_tfs)
        verts = verts_mesh_deformed[0].float()
        if def_scale:
            mesh = Mesh(v=verts,f=faces)
            mesh.auto_size_by_body(self.scale_def)
        else:
            mesh = Mesh(base=mesh)
            mesh.v = verts
            mesh.auto_size()
        mesh.auto_normal()
        return mesh
    
    def set_scale_def(self,body_mesh):
        def_body = self(body_mesh)
        scale_def ={
            'scale': def_body.scale,
            'resize_matrix_inv': def_body.resize_matrix_inv,
            'offset':def_body.offset
        }
        self.scale_def = scale_def