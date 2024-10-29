import torch
from yacs.config import CfgNode as CN
import numpy as np
from lib.model.guidance import StableDiffusion
from lib.data.provider import ViewDataset
from lib.render.renderer import Renderer
from lib.trainer import Trainer
from lib.data.obj import Mesh
from lib.utils.loss_utils import calculate_rgb_depth_loss,img_loss
import tqdm
import os
import random
import torchvision.transforms as transforms
from lib.model.smplx_deformer import SMPLXDeformer
os.environ['PYOPENGL_PLATFORM'] = "osmesa"
def load_config(path, default_path=None):
    cfg = CN(new_allowed=True)
    if default_path is not None:
        cfg.merge_from_file(default_path)
    cfg.merge_from_file(path)
    return cfg
def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.  


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


import argparse
parser = argparse.ArgumentParser()

train_model = "geometry"

if train_model == "geometry":
    config_path = "/home/clothAvatar/configs/tech_geometry.yaml"
elif train_model == "texture":
    config_path = "/home/clothAvatar/configs/tech_texture.yaml"
cfg = load_config(config_path, default_path="/home/clothAvatar/configs/default.yaml")

cfg.data.part = "body"
cfg.model.use_fleximesh = True
cfg.model.voxel_grid_res = 200
cfg.model.use_deformer = False
if train_model == "geometry":
    cfg.data.body_template = "/home/clothAvatar/data/template/smplx_c.obj"
    cfg.data.upper_template = "/home/clothAvatar/data/template/upper_smplx_c.obj"
    cfg.data.lower_template = "/home/clothAvatar/data/template/lower_smplx_c2.obj"
    cfg.data.last_model = cfg.data.body_template
    if not cfg.model.use_deformer:
        cfg.data.last_model = "/home/clothAvatar/data/template/smplx_d.obj"
elif train_model == "texture":
    # cfg.data.last_model = "/home/fleximesh_c.obj"
    cfg.data.last_model = "/home/mycode3/t1029/def_body.obj"
    # cfg.data.last_model = "/home/output_without_colors.obj"
cfg.workspace = os.path.join("/home/clothAvatar/data/tet",cfg.data.part)
cfg.guidance.controlnet = None
guidance = StableDiffusion("cuda:0", "1.5", cfg.guidance.hf_key, cfg.guidance.step_range, controlnet=cfg.guidance.controlnet, lora=cfg.guidance.lora, cfg=cfg, head_hf_key=cfg.guidance.head_hf_key)
seed_everything(33)
for p in guidance.parameters():
    p.requires_grad = False
       
train_loader = ViewDataset(cfg, device="cuda:0", type='train', H=cfg.train.h, W=cfg.train.w, size=100).dataloader()
model = Renderer(cfg).to("cuda:0")
lr = 0.001
epoch = 20
cfg.guidance.text = "a red short sleeved shirt, tie"
cfg.guidance.negative = ""
cfg.guidance.controlnet_normal_guidance = True
cfg.train.head_position = np.array([0., 0.4, 0.], dtype=np.float32).tolist()    
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x)) 
scaler = torch.cuda.amp.GradScaler(enabled=True)
trainer = Trainer("text",cfg,model, guidance)
gt_mesh = Mesh().load_obj("/home/clothAvatar/data/scan.obj")
# gt_mesh = Mesh().load_obj("/home/mesh-f00001.obj",use_vertex_tex=True)
scale_data = {
    'scale': gt_mesh.scale,
    'resize_matrix_inv': gt_mesh.resize_matrix_inv,
    'offset':gt_mesh.offset
}
cloth_gt_mesh = Mesh().load_obj("/home/clothAvatar/data/lower_uv.obj",scale_data=scale_data)
# model.dmtet_network.set_sdf(cloth_gt_mesh.v,cloth_gt_mesh.f)
# smpx_gt_mesh = Mesh().load_obj("/home/clothAvatar/data/template/smplx_c.obj")
# model.dmtet_network.set_sdf(smpx_gt_mesh.v,smpx_gt_mesh.f)

for it in range(epoch):
    print(f"Epoch {it}:")
    for i, data in tqdm.tqdm(enumerate(train_loader)):
        loss = 0
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            pred_rgb, pred_depth, smooth_loss ,pred_rgb_gt,pred_depth_gt,mesh,can_mesh = trainer.train_step(data,gt_mesh)
        if train_model == "texture":
            loss = img_loss(pred_rgb_gt,pred_rgb)    
        if train_model == "geometry":
            l2_loss = calculate_rgb_depth_loss(pred_rgb, pred_depth, pred_rgb_gt, pred_depth_gt)  
            loss = loss + l2_loss
            loss = loss + smooth_loss
            # sdf_loss = model.dmtet_network.sdf_loss(200000)
            # loss = loss + sdf_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    # if it % 10 == 0:
        to_pil = transforms.ToPILImage()
        rgb_image = to_pil(pred_rgb.squeeze(0).clamp(0, 1))  # [3, H, W] to PIL Image
        output_dir = "/home/clothAvatar/output"
        rgb_image.save(f"{output_dir}/pred_rgb_{it}.png")
        torch.cuda.empty_cache()  # 释放显存
 
    print(loss)
trainer.save_mesh("/home/clothAvatar/output")