import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyEstimator(nn.Module):
    """
    通过对比正视图特征和侧视图+高斯噪声增强特征，估计每个点的不确定性 U(x)
    """
    def __init__(self, noise_std=0.1, mode='l2'):
        super(UncertaintyEstimator, self).__init__()
        assert mode in ['l2', 'cosine'], "mode must be 'l2' or 'cosine'"
        self.noise_std = noise_std
        self.mode = mode

    def forward(self, f_front, f_back, f_left, f_right):
        """
        输入：
        - f_front: Tensor [B, C], 点在正视图上的图像特征
        - f_back, f_left, f_right: Tensor [B, C]，点在三个侧视图上的图像特征

        输出：
        - U: Tensor [B, 1]，每个点的不确定性值
        """
        # 构造增强特征 z'：侧视图平均 + 高斯噪声
        side_avg = (f_back + f_left + f_right) / 3
        noise = torch.randn_like(side_avg) * self.noise_std
        f_aug = side_avg + noise  # z'

        # 原始特征 z
        f_orig = f_front  # z

        # 不确定性计算
        if self.mode == 'l2':
            u = torch.norm(f_orig - f_aug, dim=1, keepdim=True)  # [B, 1]
        elif self.mode == 'cosine':
            sim = F.cosine_similarity(f_orig, f_aug, dim=1, eps=1e-6)
            u = 1.0 - sim.unsqueeze(1)

        return u

class DependencyPredictor(nn.Module):
    """
    预测依赖系数
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(DependencyPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, x):
        """
        输入包括bary_centric_feat,SDF,不确定性估计U_x
        """
        # 输入: [B, D, N]，先转成 [B*N, D]
        B, D, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, D)
        out = self.mlp(x)  # [B*N, 1]
        out = out.view(B, N, 1).permute(0, 2, 1)  # [B, 1, N]
        return out
    
class BranchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  # [B, D, N]
        B, D, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, D)  # [B*N, D]
        out = self.mlp(x)  # [B*N, hidden]
        return out.view(B, N, -1).permute(0, 2, 1)  # [B, hidden, N]

class DependencyPredictorMultiBranch(nn.Module):
    def __init__(self, geo_dim=33, unc_dim=1, hidden_dim=64):
        super().__init__()
        self.geo_branch = BranchMLP(geo_dim, hidden_dim)
        self.unc_branch = BranchMLP(unc_dim, hidden_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, bary_centric_feat, sdf, U_x):
        """
        bary_centric_feat: [B, N, 32]
        sdf:                [B, N, 1]
        U_x:                [B, 1, N]
        """
        B, N, _ = bary_centric_feat.shape

        # 1. 准备几何输入：[B, N, 33] → [B, 33, N]
        geo_input = torch.cat([bary_centric_feat, sdf], dim=-1).permute(0, 2, 1)  # [B, 33, N]
        
        # 2. 不确定性输入已经是 [B, 1, N]
        unc_input = U_x  # [B, 1, N]

        # 3. 分支 MLP 提取特征
        geo_feat = self.geo_branch(geo_input)  # [B, hidden, N]
        unc_feat = self.unc_branch(unc_input)  # [B, hidden, N]

        # 4. 特征融合 → 依赖系数预测
        fused_feat = torch.cat([geo_feat, unc_feat], dim=1)  # [B, hidden*2, N]
        D_x = self.fusion_mlp(fused_feat)  # [B, 1, N]
        return D_x
