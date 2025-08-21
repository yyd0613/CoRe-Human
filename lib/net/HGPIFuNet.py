# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

#from lib.net.voxelize import Voxelization
from lib.renderer.mesh import compute_normal_batch
from lib.dataset.mesh_util import feat_select, read_smpl_constants, surface_field_deformation
from lib.net.NormalNet import NormalNet
from lib.net.MLP import MLP, DeformationMLP, TransformerEncoderLayer, SDF2Density, SDF2Occ
from lib.net.spatial import SpatialEncoder
from lib.dataset.PointFeat import PointFeat
from lib.dataset.mesh_util import SMPLX
from lib.net.VE import VolumeEncoder
from lib.net.ResBlkPIFuNet import ResnetFilter
from lib.net.UNet import UNet
from lib.net.HGFilters import *
from lib.net.Transformer import ViTVQ
from termcolor import colored
from lib.net.BasePIFuNet import BasePIFuNet
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from lib.net.nerf_util import raw2outputs
from lib.net.Uncertainty import UncertaintyEstimator, DependencyPredictor
import math


def normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def visualize_feature_map(feature_map, title, filename):
    feature_map=feature_map.permute(0, 2, 3, 1)
    # 选择一个样本（如果有多个）
    sample_index = 0
    sample = feature_map[sample_index]
    
    # 选择一个通道（如果有多个）
    channel_index = 0
    channel = sample[:, :, channel_index]
    channel= normalize(channel)
    
    plt.imshow(channel.cpu().numpy(), cmap='hot')
    # plt.title(title)
    # plt.colorbar()
    plt.axis('off')
    plt.savefig(filename, dpi=300,bbox_inches='tight', pad_inches=0)  # 保存图片到文件
    plt.close()  # 关闭图形，释放资源

def compute_uncertainty_from_features(F_feat, B_feat, R_feat, L_feat, estimator):
    """
    从特征图中采样到的点特征中计算不确定性
    - F_feat: Tensor [B, C, N]（正视图特征）
    - B_feat/R_feat/L_feat: Tensor [B, C, N]（三个侧视图特征）
    - estimator: 不确定性估计器实例
    返回: U(x) [B, 1, N]
    """
    B, C, N = F_feat.shape
    F_feat_flat = F_feat.permute(0, 2, 1).reshape(-1, C)
    B_feat_flat = B_feat.permute(0, 2, 1).reshape(-1, C)
    R_feat_flat = R_feat.permute(0, 2, 1).reshape(-1, C)
    L_feat_flat = L_feat.permute(0, 2, 1).reshape(-1, C)

    u_flat = estimator(F_feat_flat, B_feat_flat, R_feat_flat, L_feat_flat)  # [B*N, 1]
    U = u_flat.reshape(B, N).unsqueeze(1)  # [B, 1, N]
    return U

def pe_encode_sdf(sdf, num_freqs=3):
    """
    将 SDF 进行 PE 编码。
    输入：
        sdf: Tensor [B, N, 1]，每个点的 SDF 值
        num_freqs: 使用的频率数，默认3，对应输出维度为 6
    输出：
        pe_sdf: Tensor [B, N, 2*num_freqs]，编码后的特征
    """
    B, N, _ = sdf.shape  # [B, N, 1]=[1,N,1]
    
    # [1, 1, 3]
    freq_exponents = torch.arange(num_freqs, dtype=torch.float32, device=sdf.device) * (2.0 / (2 * num_freqs))
    omega = 1.0 / (10000 ** freq_exponents)  # [3]
    omega = omega.view(1, 1, num_freqs)  # [1, 1, 3]

    scaled_sdf = sdf * omega  # [B, N, 3]
    sin_comp = torch.sin(scaled_sdf)  # [B, N, 3]
    cos_comp = torch.cos(scaled_sdf)  # [B, N, 3]
    
    pe_list = []
    for i in range(num_freqs):
        pe_list.append(sin_comp[:, :, i:i+1])  # [B, N, 1]
        pe_list.append(cos_comp[:, :, i:i+1])  # [B, N, 1]

    pe_sdf = torch.cat(pe_list, dim=2)  # [B, N, 6]
    return pe_sdf

class HGPIFuNet(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """

    def __init__(self,
                 cfg,
                 projection_mode="orthogonal",
                 error_term=nn.MSELoss()):

        super(HGPIFuNet, self).__init__(projection_mode=projection_mode,
                                        error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()
        self.opt = cfg.net
        self.root = cfg.root
        self.overfit = cfg.overfit

        channels_IF = self.opt.mlp_dim

        self.use_filter = self.opt.use_filter
        self.prior_type = self.opt.prior_type
        self.smpl_feats = self.opt.smpl_feats

        self.smpl_dim = self.opt.smpl_dim
        self.voxel_dim = self.opt.voxel_dim
        self.hourglass_dim = self.opt.hourglass_dim

        self.in_geo = [item[0] for item in self.opt.in_geo] # ['normal_F', 'normal_B']
        self.in_nml = [item[0] for item in self.opt.in_nml] # ['image', 'T_normal_F', 'T_normal_B']

        self.in_geo_dim = sum([item[1] for item in self.opt.in_geo])
        self.in_nml_dim = sum([item[1] for item in self.opt.in_nml])

        self.in_total = self.in_geo + self.in_nml # ['normal_F', 'normal_B', 'image', 'T_normal_F', 'T_normal_B']
        self.smpl_feat_dict = None
        self.smplx_data = SMPLX()

        image_lst = [0, 1, 2]
        normal_F_lst = [0, 1, 2] if "image" not in self.in_geo else [3, 4, 5]
        normal_B_lst = [3, 4, 5] if "image" not in self.in_geo else [6, 7, 8]

        # only ICON or ICON-Keypoint use visibility

        if self.prior_type in ["icon", "keypoint"]:
            if "image" in self.in_geo:
                self.channels_filter = [
                    image_lst + normal_F_lst,
                    image_lst + normal_B_lst,
                ]
            else:
                self.channels_filter = [normal_F_lst, normal_B_lst] # [[0, 1, 2], [3, 4, 5]]

        else:
            if "image" in self.in_geo:
                self.channels_filter = [
                    image_lst + normal_F_lst + normal_B_lst
                ]
            else:
                self.channels_filter = [normal_F_lst + normal_B_lst]

        use_vis = (self.prior_type in ["icon", "keypoint"
                                       ]) and ("vis" in self.smpl_feats)
        if self.prior_type in ["pamir", "pifu"]:
            use_vis = 1

        if self.use_filter:
            channels_IF[0] = (self.hourglass_dim) * (2 - use_vis)
        else:
            channels_IF[0] = len(self.channels_filter[0]) * (2 - use_vis)

        if self.prior_type in ["icon", "keypoint"]:
            channels_IF[0] += self.smpl_dim
        
        elif self.prior_type == "pifu":
            channels_IF[0] += 1
        else:
            print(f"don't support {self.prior_type}!")

        self.base_keys = ["smpl_verts", "smpl_faces"]

        self.icon_keys = self.base_keys + [
            f"smpl_{feat_name}" for feat_name in self.smpl_feats
        ]
        self.keypoint_keys = self.base_keys + [
            f"smpl_{feat_name}" for feat_name in self.smpl_feats
        ]

        self.pamir_keys = [
            "voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"
        ]
        self.pifu_keys = []

        # channels_IF[0]+=self.hourglass_dim
        # self.if_regressor = MLP(
        #     filter_channels=channels_IF,
        #     name="if",
        #     res_layers=self.opt.res_layers,
        #     norm=self.opt.norm_mlp,
        #     last_op=nn.Sigmoid() if not cfg.test_mode else None,
        # )

        self.deform_dim=64
        
        #self.image_filter = ResnetFilter(self.opt, norm_layer=norm_type)
        #self.image_filter = UNet(3,128)
        # self.xy_plane_filter=ResnetFilter(self.opt, norm_layer=norm_type)
        # self.yz_plane_filter=ViTVQ(image_size=512) # ResnetFilter(self.opt, norm_layer=norm_type)
        # self.xz_plane_filter=ViTVQ(image_size=512)
        self.image_filter=ViTVQ(image_size=512,channels=9)
        self.uncertainty_estimator = UncertaintyEstimator(noise_std=0.1, mode='l2')
        self.dependency_net = DependencyPredictor(input_dim=34)
        # input_dim = bary_centric_feat=32维 + sdf=1维 + U=1维

        # self.deformation_mlp=DeformationMLP(input_dim=self.deform_dim,opt=self.opt)
        self.mlp=TransformerEncoderLayer(skips=4,multires=6,opt=self.opt)
        # self.sdf2density=SDF2Density()
        # self.sdf2occ=SDF2Occ()
        self.color_loss=nn.L1Loss()
        self.sp_encoder = SpatialEncoder()
        self.step=0
        self.features_costume=None

        # network
        if self.use_filter:
            if self.opt.gtype == "HGPIFuNet":
                self.F_filter = HGFilter(self.opt, self.opt.num_stack,
                                         len(self.channels_filter[0]))
                # self.refine_filter = FuseHGFilter(self.opt, self.opt.num_stack,
                #                                 len(self.channels_filter[0]))
                
            else:
                print(
                    colored(f"Backbone {self.opt.gtype} is unimplemented",
                            "green"))

        summary_log = (f"{self.prior_type.upper()}:\n" +
                       f"w/ Global Image Encoder: {self.use_filter}\n" +
                       f"Image Features used by MLP: {self.in_geo}\n")

        if self.prior_type == "icon":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (ICON): {self.smpl_dim}\n"
        elif self.prior_type == "keypoint":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (Keypoint): {self.smpl_dim}\n"
        elif self.prior_type == "pamir":
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PaMIR): {self.voxel_dim}\n"
        else:
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PIFu): 1 (z-value)\n"

        summary_log += f"Dim of MLP's first layer: {channels_IF[0]}\n"

        print(colored(summary_log, "yellow"))

        self.normal_filter = NormalNet(cfg)

        init_net(self, init_type="normal")

    def get_normal(self, in_tensor_dict):

        # insert normal features
        if (not self.training) and (not self.overfit):
            # print(colored("infer normal","blue"))
            with torch.no_grad():
                feat_lst = []
                if "image" in self.in_geo:
                    feat_lst.append(
                        in_tensor_dict["image"])  # [1, 3, 512, 512]
                if "normal_F" in self.in_geo and "normal_B" in self.in_geo:
                    if ("normal_F" not in in_tensor_dict.keys()
                            or "normal_B" not in in_tensor_dict.keys()):
                        (nmlF, nmlB) = self.normal_filter(in_tensor_dict)
                    else:
                        nmlF = in_tensor_dict["normal_F"]
                        nmlB = in_tensor_dict["normal_B"]
                    feat_lst.append(nmlF)  # [1, 3, 512, 512]
                    feat_lst.append(nmlB)  # [1, 3, 512, 512]
            in_filter = torch.cat(feat_lst, dim=1) # in_tensor_dict["normal_F"]+in_tensor_dict["normal_B"]

        else:
            in_filter = torch.cat([in_tensor_dict[key] for key in self.in_geo],
                                  dim=1)

        return in_filter

    def get_mask(self, in_filter, size=128):

        mask = (F.interpolate(
            in_filter[:, self.channels_filter[0]],
            size=(size, size),
            mode="bilinear",
            align_corners=True,
        ).abs().sum(dim=1, keepdim=True) != 0.0)

        return mask


    def filter(self, in_tensor_dict, return_inter=False):
        """
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        """

        in_filter = self.get_normal(in_tensor_dict) #[1,6,512,512] 正面和背面法相图合并
        image= in_tensor_dict["image"]
        fuse_image=torch.cat([image,in_filter], dim=1) 
        smpl_normals={
            "T_normal_B":in_tensor_dict['normal_B'],
            "T_normal_R":in_tensor_dict['T_normal_R'],
            "T_normal_L":in_tensor_dict['T_normal_L']
        }
        features_G = []

        # self.smpl_normal=in_tensor_dict['T_normal_L']

        if self.prior_type in ["icon", "keypoint"]:
            if self.use_filter:
                triplane_features = self.image_filter(fuse_image,smpl_normals)
                
                features_F = self.F_filter(in_filter[:,
                                                     self.channels_filter[0]]
                                           )  # [(B,hg_dim,128,128) * 4]
                features_B = self.F_filter(in_filter[:,
                                                     self.channels_filter[1]]
                                           )  # [(B,hg_dim,128,128) * 4]
            else:
                assert 0

            F_plane_feat,B_plane_feat,R_plane_feat,L_plane_feat=triplane_features
            
            refine_F_plane_feat=F_plane_feat
            features_G.append(refine_F_plane_feat)
            features_G.append(B_plane_feat)
            features_G.append(R_plane_feat)
            features_G.append(L_plane_feat)
            features_G.append(torch.cat([features_F[-1],features_B[-1]], dim=1))

        else:
            assert 0

        self.smpl_feat_dict = {
            k: in_tensor_dict[k] if k in in_tensor_dict.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        } # [smpl_verts,smpl_faces,smpl_sdf,smpl_cmap,smpl_norm,smpl_vis,smpl_sample_id]
        if 'animated_smpl_verts' not in in_tensor_dict.keys():
            self.point_feat_extractor = PointFeat(self.smpl_feat_dict["smpl_verts"],
                                               self.smpl_feat_dict["smpl_faces"])
        else:
            assert 0
            
        self.features_G = features_G
        
        # If it is not in training, only produce the last im_feat
        if not self.training:
            features_out = features_G
        else:
            features_out = features_G

        if return_inter:
            return features_out, in_filter
        else:
            return features_out
        
        

    def query(self, features, points, calibs, transforms=None,type='shape'):
        print("N=",points.shape)

        xyz = self.projection(points, calibs, transforms) # project to image plane [B,C,N]=[1,3,N]
     
        (xy, z) = xyz.split([2, 1], dim=1)
        
       
        zy=torch.cat([xyz[:,2:3],xyz[:,1:2]],dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=True).detach().float()

        preds_list = []
      

        if self.prior_type in ["icon", "keypoint"]:

            
            
            densely_smpl=self.smpl_feat_dict['smpl_verts'].permute(0,2,1) # [1,3,10475]
            #smpl_origin=self.projection(densely_smpl, torch.inverse(calibs), transforms)
            smpl_vis=self.smpl_feat_dict['smpl_vis'].permute(0,2,1) # [1,1,10475]
            #verts_ids=self.smpl_feat_dict['smpl_sample_id']

            

            (smpl_xy,smpl_z)=densely_smpl.split([2,1],dim=1)
            smpl_zy=torch.cat([densely_smpl[:,2:3],densely_smpl[:,1:2]],dim=1)
                                
            point_feat_out = self.point_feat_extractor.query(  # this extractor changes if has animated smpl
                xyz.permute(0, 2, 1).contiguous(), self.smpl_feat_dict)
            vis=point_feat_out['vis'].permute(0,2,1) #[B,C,N]=[1,1,N]
            sdf = -point_feat_out['sdf']    # [B,N,C]=[1,N,1] this sdf needs to be multiplied by -1
            pe_sdf = pe_encode_sdf(sdf) # [B,N,6]
            
            feat_lst = [
                point_feat_out[key] for key in self.smpl_feats
                if key in point_feat_out.keys()
            ] # [sdf,camp,norm,vis] #[B,N,C] C:1+3+3+1
            feat_pe = feat_lst[1:] # [camp,norm,vis] [B,N,C] C:3+3+1
            smpl_feat = torch.cat(feat_lst, dim=2).permute(0, 2, 1) # [B,C N] = [1,8,N]
            smpl_feat_part = torch.cat(feat_pe, dim=2).permute(0, 2, 1) # [B,C N] = [1,7,N]

        if len(features)==5: 
            
            F_plane_feat1,F_plane_feat2=features[0].chunk(2,dim=1)
            B_plane_feat1,B_plane_feat2=features[1].chunk(2,dim=1)
            R_plane_feat1,R_plane_feat2=features[2].chunk(2,dim=1)
            L_plane_feat1,L_plane_feat2=features[3].chunk(2,dim=1)
            in_feat=features[4]
            
           
            F_feat=self.index(F_plane_feat1,xy)
            B_feat=self.index(B_plane_feat1,xy) # [1,16,N]
            R_feat=self.index(R_plane_feat1,zy) # [1,16,N]
            L_feat=self.index(L_plane_feat1,zy) # [1,16,N]
            normal_feat=feat_select(self.index(in_feat, xy),vis) # [1,6,N]
            three_plane_feat=(B_feat+R_feat+L_feat)/3 # [1,16,N]
            triplane_feat=torch.cat([F_feat,three_plane_feat],dim=1)  # [1,32,N]       
            
            # 通过对比学习计算不确定性 [1,1,N]
            U_x = compute_uncertainty_from_features(F_feat, B_feat, R_feat, L_feat, self.uncertainty_estimator)


            ### smpl query ###
            smpl_F_feat=self.index(F_plane_feat2,smpl_xy) # [1,16,10475]
            smpl_B_feat=self.index(B_plane_feat2,smpl_xy) 
            smpl_R_feat=self.index(R_plane_feat2,smpl_zy)
            smpl_L_feat=self.index(L_plane_feat2,smpl_zy)



            smpl_three_plane_feat=(smpl_B_feat+smpl_R_feat+smpl_L_feat)/3 # [1,16,10475]
            smpl_triplane_feat=torch.cat([smpl_F_feat,smpl_three_plane_feat],dim=1)  # [1,32,10475]     
            bary_centric_feat=self.point_feat_extractor.query_barycentirc_feats(xyz.permute(0,2,1).contiguous()
                                                                      ,smpl_triplane_feat.permute(0,2,1)) # [1, 2048, 32]
            
            # 计算依赖系数，共有三个输入，第一部分几何信息，第二部分不确定性信息
            # 原始的shape：bary_centric_feat[1,2048,32] sdf[1,2048,1] U_x[1,1,2048] 需要转化为[B,D,N]
            dependency_input = torch.cat([bary_centric_feat.permute(0, 2, 1), sdf.permute(0, 2, 1), U_x], dim=1)  # 输入维度[B, 34, N]
            D_x = self.dependency_net(dependency_input)  # [B, 1, N] = [1,1,2048], 依赖系数
            
            # D(x)控制SDF的可信程度
            # g_sdf.shape = [B,C,N] D_x.shape[B, 1, N] = [1,1,2048] pe_sdf.shape=[1,2048,6]
            g_sdf = D_x * pe_sdf.permute(0, 2, 1) 
            smpl_feat_all = torch.cat([g_sdf,smpl_feat_part],dim=1) # [B,C,N] C=6+7=13

            
            final_feat=torch.cat([triplane_feat,bary_centric_feat.permute(0,2,1),normal_feat],dim=1)  # shape[B,C,N] C=32+32+6=70

            if self.features_costume is not None:
                assert 0
            if type=='shape':
                if 'animated_smpl_verts' in self.smpl_feat_dict.keys():
                    animated_smpl=self.smpl_feat_dict['animated_smpl_verts']
                    
                    occ=self.mlp(xyz.permute(0,2,1).contiguous(),animated_smpl,
                                                        final_feat,smpl_feat_all,training=self.training,type=type)
                else:
                    
                    occ=self.mlp(xyz.permute(0,2,1).contiguous(),densely_smpl.permute(0,2,1),
                                                        final_feat,smpl_feat_all,training=self.training,type=type)
                occ=occ*in_cube
                preds_list.append(occ)   

            elif type=='color':
                if 'animated_smpl_verts' in self.smpl_feat_dict.keys():
                    animated_smpl=self.smpl_feat_dict['animated_smpl_verts']
                    color_preds=self.mlp(xyz.permute(0,2,1).contiguous(),animated_smpl,
                                                        final_feat,smpl_feat_all,training=self.training,type=type)
                    
                
                else:
                    color_preds=self.mlp(xyz.permute(0,2,1).contiguous(),densely_smpl.permute(0,2,1),
                                                        final_feat,smpl_feat_all,training=self.training,type=type)
                preds_list.append(color_preds)  

        return preds_list




    def get_error(self, preds_if_list, labels):
        """calculate error

        Args:
            preds_list (list): list of torch.tensor(B, 3, N)
            labels (torch.tensor): (B, N_knn, N)

        Returns:
            torch.tensor: error
        """
        error_if = 0

        for pred_id in range(len(preds_if_list)):
            pred_if = preds_if_list[pred_id]
            error_if += F.binary_cross_entropy(pred_if, labels)

        error_if /= len(preds_if_list)

        return error_if


    def forward(self, in_tensor_dict):
       
        sample_tensor = in_tensor_dict["sample"]
        calib_tensor = in_tensor_dict["calib"]
        label_tensor = in_tensor_dict["label"]
       
        color_sample=in_tensor_dict["sample_color"]
        color_label=in_tensor_dict["color"]


        in_feat = self.filter(in_tensor_dict)
       
        

        preds_if_list = self.query(in_feat,
                                   sample_tensor,
                                   calib_tensor,type='shape')

        BCEloss = self.get_error(preds_if_list, label_tensor)

        color_preds=self.query(in_feat,
                               color_sample,
                               calib_tensor,type='color')
        color_loss=self.color_loss(color_preds[0],color_label)



        if self.training:
           
            self.color3d_loss= color_loss
            error=BCEloss+color_loss
            self.grad_loss=torch.tensor(0.).float().to(BCEloss.device)
        else:
            error=BCEloss

        return preds_if_list[-1].detach(), error