"""Lightning wrapper of the pytorch model."""
"""现在只能预测单段位移"""
from typing import Dict,Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from smplx import SMPLLayer
from smplx.lbs import vertices2joints
from torch.optim.lr_scheduler import MultiStepLR

from pedgen.model.diffusion_utils import (MLPHead, MotionTransformer,
                                          cosine_beta_schedule, get_dct_matrix)
from pedgen.utils.occupancy_builder import OccupancyGridBuilder
from pedgen.utils.rot import positional_encoding_2d, rotation_6d_to_matrix

class PointNetEncoder(nn.Module):  #场景编码
    """PointNet-style global encoder for scene/depth/semantic points."""
    #把4096个点压缩成一个256维全局场景向量
    def __init__(self, in_dim: int = 4, hidden_dim: int = 128, out_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )#对每个点单独做 MLP（逐点提特征）

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: [B, N, C]batch size、点数、维度
        points = points.transpose(1, 2).contiguous()#转置为[B, C, N]
        feat = self.mlp(points)#逐点卷积[B,256,N]
        return torch.max(feat, dim=-1).values#对所有点取最大值[B,256]

class Predictor(nn.Module):
    def __init__(self, latent_dim: int, use_image: bool = False) -> None:
        super().__init__()
        self.use_image = use_image
        
        # 用简单的CNN从RGB图像提取全局视觉特征
        self.rgb_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )#从输入图像提取视觉特征
        self.rgb_proj = nn.Linear(128, 256)

        # [x, y, z, semantic]编码场景点云
        self.scene_backbone = PointNetEncoder(in_dim=4, hidden_dim=128, out_dim=256)

        # [fx, fy, cx, cy] 编码相机内参
        self.cam_backbone = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        #把RGB, scene, intrinsics特征拼接后映射到latent_dim
        self.context_proj = nn.Sequential(
            nn.Linear(256 + 256 + 64, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        # 输出预测值
        self.init_head = nn.Linear(latent_dim, 3)
        self.goal_head = nn.Linear(latent_dim, 3)

    def _select_img(self, img: torch.Tensor) -> torch.Tensor:
        # train/test: [B,C,H,W], pred mode may be [B,T,C,H,W]
        if img.ndim == 5:
            return img[:, 0]#若是5维，就取第0帧图像进行预测（不用整段图像序列）
        return img
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        img = self._select_img(batch["img"])
        rgb_feat = self.rgb_backbone(img).flatten(1)
        rgb_feat = self.rgb_proj(rgb_feat)

        if "scene_points_raw" not in batch:
            raise RuntimeError("scene_points_raw is required for Predictor")
        # 此时 batch["scene_points_raw"] 已是[B, 4096, 4]
        scene_points = batch["scene_points_raw"].to(img.device)
        scene_feat = self.scene_backbone(scene_points)#[B,256]

        intrinsics = batch["intrinsics"].to(img.device)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        cam_feat = self.cam_backbone(torch.stack([fx, fy, cx, cy], dim=-1))

        predictor_feature = self.context_proj(torch.cat([rgb_feat, scene_feat, cam_feat], dim=-1))#拼接三路信息
        pre_init_pos = self.init_head(predictor_feature)
        pre_goal_rel = self.goal_head(predictor_feature)
        return {"pre_init_pos": pre_init_pos, "pre_goal_rel": pre_goal_rel}

class PedGenModel(LightningModule):
    """Lightning model for pedestrian generation."""

    def __init__(
        self,
        gpus: int,
        batch_size_per_device: int,
        diffuser_conf: Dict,
        noise_steps: int,
        ddim_timesteps: int,
        optimizer_conf: Dict,
        mod_train: float,
        num_sample: int,
        lr_scheduler_conf: Dict,
        #多模态条件输入
        use_goal: bool = False,
        use_image: bool = False,
        use_beta: bool = False,
    ) -> None:
        super().__init__()#调用pl.LightningModule的构造方法
        self.noise_steps = noise_steps
        self.ddim_timesteps = ddim_timesteps
        self.beta = cosine_beta_schedule(self.noise_steps)#加噪率
        alpha = 1. - self.beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)
        self.diffuser = MotionTransformer(**diffuser_conf)#将其初始化为MotionTransformer类的一个实例，配置参数是**
        self.predictor = Predictor(diffuser_conf["latent_dim"],use_image=use_image)

        self.criterion = F.mse_loss#重建损失用 MSE
        self.criterion_traj = F.l1_loss#轨迹损失用 L1
        self.criterion_goal = F.l1_loss#起点/目标损失用 L1

        self.optimizer_conf = optimizer_conf
        self.lr_scheduler_conf = lr_scheduler_conf
        self.gpus = gpus
        self.batch_size_per_device = batch_size_per_device
        self.mod_train = mod_train

        self.num_sample = num_sample
        self.use_goal = use_goal
        self.use_beta = use_beta
        self.use_image = use_image

        self.smpl = SMPLLayer(model_path="smpl", gender='neutral')
        for param in self.smpl.parameters():
            param.requires_grad = False

        if self.use_goal:
            self.goal_embed = MLPHead(3, diffuser_conf["latent_dim"])
        if self.use_beta:
            self.beta_embed = MLPHead(10, diffuser_conf["latent_dim"])

        if self.use_image:#使用Cross-Attention，让生成的动作与环境图像进行交互
            img_ch_in = 40  # hardcoded
            self.img_embed = MLPHead(img_ch_in, diffuser_conf["latent_dim"])
            self.img_cross_attn_norm = nn.LayerNorm(diffuser_conf["latent_dim"])
            self.img_cross_attn = nn.MultiheadAttention(
                diffuser_conf["latent_dim"],
                diffuser_conf["num_heads"],
                dropout=0.2,
                batch_first=True)

        self.cond_embed = nn.Parameter(torch.zeros(diffuser_conf["latent_dim"]))#cond_embed是可学习的参数

        self.mask_embed = nn.Parameter(torch.zeros(diffuser_conf["input_feats"]))

        self.ddim_timestep_seq = np.asarray(
            list(
                range(0, self.noise_steps,
                      self.noise_steps // self.ddim_timesteps))) + 1
        self.ddim_timestep_prev_seq = np.append(np.array([0]),
                                                self.ddim_timestep_seq[:-1])

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:  #原始输入图像、时间步、随机噪声
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]#去噪部分：干净信号的比例
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]#噪声部分
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise #返回xt
    
    #定义模型在训练时如何学习
    def forward(self, batch: Dict) -> Dict:
        B = batch['img'].shape[0]
        predictor_dict = self.predict_context(batch)#先跑预测器
        full_motion = self.get_full_motion(batch)#得到GT动作
        cond_embed = self.get_condition(batch, predictor_dict)#最终给扩散模型的条件

        # classifier free sampling
        if np.random.random() > self.mod_train:
            cond_embed = None

        # randomly sample timesteps
        ts = torch.randint(0, self.noise_steps, ((B + 1) // 2,))
        if B % 2 == 1:
            ts = torch.cat([ts, self.noise_steps - ts[:-1] - 1], dim=0).long()
        else:
            ts = torch.cat([ts, self.noise_steps - ts - 1], dim=0).long()
        ts = ts.to(self.device)

        # generate Gaussian noise
        noise = torch.randn_like(full_motion)

        # calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=full_motion, t=ts, noise=noise)

        #如果某些时间步被mask，就用特殊可学习向量替换
        if "motion_mask" in batch:
            x_t[batch["motion_mask"] == 1] = self.mask_embed.unsqueeze(0).unsqueeze(0)

        # predict noise
        pred_motion = self.diffuser(x_t, ts, cond_embed=cond_embed)#扩散模型预测

        # calculate loss
        if "motion_mask" in batch:
            pred_motion[batch["motion_mask"] == 1] = 0
            full_motion[batch["motion_mask"] == 1] = 0

        loss = self.criterion(pred_motion, full_motion)

        #loss_dict = {'loss': loss, 'loss_rec': loss.item()}
        loss_dict = {
            'loss': loss,
            'loss_rec': loss.item(),
            'loss_init': predictor_dict['loss_init'],
            'loss_goal': predictor_dict['loss_goal'],
        }

        local_trans = pred_motion[..., :3]
        gt_local_trans = full_motion[..., :3]

        local_trans_sum = torch.cumsum(local_trans, dim=-2)
        gt_local_trans_sum = torch.cumsum(gt_local_trans, dim=-2)
        #轨迹损失
        loss_traj = self.criterion_traj(local_trans_sum, gt_local_trans_sum) * 1.0
        loss_dict["loss_traj"] = loss_traj
        loss_dict["loss"] += loss_traj

        #把预测动作和 GT 动作都喂进 SMPL，得到两边对应的人体关节位置，然后比较关节位置
        betas = batch["betas"].unsqueeze(1).repeat(1, 60, 1).reshape(-1, 10)
        pred_smpl_output = self.smpl(
            transl=None,
            betas=betas,
            global_orient=None,
            body_pose=rotation_6d_to_matrix(pred_motion[..., 9:].reshape(-1, 23, 6)),
        )

        pred_joint_locations = vertices2joints(self.smpl.J_regressor, pred_smpl_output.vertices)

        gt_smpl_output = self.smpl(
            transl=None,
            betas=betas,
            global_orient=None,
            body_pose=rotation_6d_to_matrix(full_motion[..., 9:].reshape(-1, 23, 6)),
        )

        gt_joint_locations = vertices2joints(self.smpl.J_regressor, gt_smpl_output.vertices)
        loss_geo = self.criterion(pred_joint_locations, gt_joint_locations)#几何损失

        loss_dict["loss_geo"] = loss_geo.item()
        loss_dict["loss"] += loss_geo
        #loss_dict["loss"] += predictor_dict["loss_init"] + predictor_dict["loss_goal"]
        loss_dict["loss"] += (predictor_dict["loss_init"] + predictor_dict["loss_goal"]) * 0.05
        loss_dict.update({
            "pre_init_pos": predictor_dict["pre_init_pos"],
            "pre_goal_rel": predictor_dict["pre_goal_rel"],
        })
        return loss_dict

    #========================================
    # 把 Predictor 的输出，整理成扩散模型真正要用的条件
    def predict_context(self, batch: Dict) -> Dict[str, torch.Tensor]:
        predictor_output = self.predictor(batch)#提取预测值
        gt_init_pos = batch.get("gt_init_pos", None)
        gt_goal_rel = batch.get("gt_goal_rel", None)
        # 只有在有 GT 时才计算 Loss（训练和验证阶段）
        if gt_init_pos is not None and gt_goal_rel is not None:
            loss_init = self.criterion_goal(predictor_output["pre_init_pos"], gt_init_pos)
            loss_goal = self.criterion_goal(predictor_output["pre_goal_rel"], gt_goal_rel)
        else:
            # 纯推理阶段，给默认值 0 防止返回的字典缺少键值
            loss_init = torch.tensor(0.0, device=self.device)
            loss_goal = torch.tensor(0.0, device=self.device)
        
        predictor_output["loss_init"] = loss_init
        predictor_output["loss_goal"] = loss_goal

        if self.training and gt_goal_rel is not None and gt_init_pos is not None:
            current_epoch = getattr(self, "current_epoch", 0)
            max_epochs = getattr(getattr(self, "trainer", None), "max_epochs", 1) or 1
            epoch_ratio = float(current_epoch) / float(max_epochs)

            if epoch_ratio < 0.3:
                use_gt_mask = torch.ones(batch["img"].shape[0], 1, device=self.device, dtype=torch.bool)
            elif epoch_ratio < 0.7:
                gt_prob = (0.7 - epoch_ratio) / 0.4
                use_gt_mask = torch.rand(batch["img"].shape[0], 1, device=self.device) < gt_prob
            else:
                use_gt_mask = torch.zeros(batch["img"].shape[0], 1, device=self.device, dtype=torch.bool)

            predictor_output["tf_init_pos"] = torch.where(
                use_gt_mask,
                gt_init_pos,
                predictor_output["pre_init_pos"].detach(),
            )#.detach()是为了不让扩散网络的梯度传回predictor
            
            predictor_output["tf_goal_rel"] = torch.where(
                use_gt_mask,
                gt_goal_rel,
                predictor_output["pre_goal_rel"].detach(),
            )

            # 关键：new_img 不能再使用 gt_new_img，只能用预测起点在线构建
        else:
            predictor_output["tf_init_pos"] = predictor_output["pre_init_pos"]
            predictor_output["tf_goal_rel"] = predictor_output["pre_goal_rel"]

        is_sequence = False
        predictor_output["pre_new_img"] = self.build_pre_new_img(
            batch,
            predictor_output["tf_init_pos"],
            predictor_output["tf_goal_rel"],
            is_sequence=is_sequence,
        )
            
        predictor_output["tf_new_img"] = predictor_output["pre_new_img"]
        return predictor_output

    def build_pre_new_img(self, batch: Dict, pre_init_pos: torch.Tensor,
                      pre_goal_rel: torch.Tensor, is_sequence: bool) -> torch.Tensor:
        occupancy_builder = OccupancyGridBuilder(batch, self.device)
        return occupancy_builder.build(pre_init_pos, pre_goal_rel, is_sequence=is_sequence)

    #Lightning的标准训练入口
    def training_step(self, batch: Dict) -> Dict:
        loss_dict = self(batch)#调用forward，得到loss，把其中标量项写入日志
        for key, val in loss_dict.items():
            # 过滤掉多维张量，只允许标量 (Scalar Tensor 或 float) 写入日志
            if isinstance(val, torch.Tensor) and val.numel() > 1:
                continue
            
            self.log("train/" + key,
                     val,
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=False,
                     batch_size=batch["batch_size"])
        return loss_dict
    #============================================
    
    #把各种条件融合成 cond_embed
    def get_condition(self, batch, predictor_dict: Optional[Dict] = None):
        B = batch['img'].shape[0]#取 batch size
        cond_embed = self.cond_embed.unsqueeze(0).repeat(B, 1)

        if self.use_goal:
            cond_embed = cond_embed + self.goal_embed(predictor_dict["tf_goal_rel"])
        if self.use_beta:
            cond_embed = cond_embed + self.beta_embed(batch["betas"])

        if self.use_image:
            #img = batch['new_img']
            img = predictor_dict["tf_new_img"]
            img_feature = img[..., :-2]
            img_pos = img[..., -2:]
            img_pos_embed = positional_encoding_2d(img_pos, self.diffuser.latent_dim)
            img_embed = self.img_embed(img_feature) + img_pos_embed
            cond_embed = cond_embed.unsqueeze(1)
            #学习条件cond_embed与场景img_embed的关系
            cond_embed_res = self.img_cross_attn(
                query=cond_embed,
                key=self.img_cross_attn_norm(img_embed),
                value=self.img_cross_attn_norm(img_embed))
            cond_embed = (cond_embed + cond_embed_res[0]).squeeze(1)

        return cond_embed
    #把人体的位移、朝向、肢体动作打包成一个大向量
    def get_full_motion(self, batch):
        #=============================
        if "gt_init_pos" not in batch:
            batch["gt_init_pos"] = batch["global_trans"][:, 0, :]
        if "gt_goal_rel" not in batch:
            batch["gt_goal_rel"] = batch["global_trans"][:, -1, :] - batch["global_trans"][:, 0, :]
        #=============================
        local_trans = batch["global_trans"].clone()

        local_trans[:, 0, :] = 0
        local_trans[:, 1:, :] -= batch["global_trans"][:, :-1, :]

        local_orient = batch["global_orient"]

        full_motion = torch.cat([local_trans, local_orient, batch["body_pose"]],
                                dim=-1)
        return full_motion
    
    #扩散采样阶段
    def sample_ddim_progressive(self,
                            batch_size,
                            cond_embed,
                            target_goal_rel=None,
                            hand_shake=False):
        seq_len = self.diffuser.num_frames
        feat_dim = self.diffuser.input_feats
        x = torch.randn(batch_size, seq_len, feat_dim, device=self.device)

        with torch.no_grad():
            for i in reversed(range(0, self.ddim_timesteps)):
                t = (torch.ones(batch_size, device=self.device) *
                    self.ddim_timestep_seq[i]).long()
                prev_t = (torch.ones(batch_size, device=self.device) *
                        self.ddim_timestep_prev_seq[i]).long()

                alpha_hat = self.alpha_hat[t][:, None, None]
                alpha_hat_prev = self.alpha_hat[prev_t][:, None, None]

                predicted_x0 = self.diffuser(x, t, cond_embed=cond_embed)
                predicted_x0 = self.inpaint_cond(
                    predicted_x0,
                    target_goal_rel=target_goal_rel,
                )

                if hand_shake:
                    predicted_x0 = self.hand_shake(predicted_x0)

                predicted_noise = (
                    x - torch.sqrt(alpha_hat) * predicted_x0
                ) / torch.sqrt(1 - alpha_hat)

                if i > 0:
                    pred_dir_xt = torch.sqrt(1 - alpha_hat_prev) * predicted_noise
                    x_prev = torch.sqrt(alpha_hat_prev) * predicted_x0 + pred_dir_xt
                else:
                    x_prev = predicted_x0

                x = x_prev

        return x

    def sample_ddim_progressive_partial(self, xt, x0):
        """
        Generate samples from the model and yield samples from each timestep.

        Args are the same as sample_ddim()
        Returns a generator contains x_{prev_t}, shape as [sample_num, n_pre, 3 * joints_num]
        """
        sample_num = xt.shape[0]
        x = xt

        with torch.no_grad():
            for i in reversed(range(0, 70)):  # hardcoded as add noise t=100
                t = (torch.ones(sample_num) *
                     self.ddim_timestep_seq[i]).long().to(self.device)
                prev_t = (torch.ones(sample_num) *
                          self.ddim_timestep_prev_seq[i]).long().to(self.device)

                alpha_hat = self.alpha_hat[t][:, None, None]  # type: ignore
                alpha_hat_prev = self.alpha_hat[prev_t][  # type: ignore
                    :, None, None]

                predicted_x0 = self.diffuser(x, t, cond_embed=None)
                predicted_x0 = self.inpaint_soft(predicted_x0, x0)

                predicted_noise = (x - torch.sqrt(
                    (alpha_hat)) * predicted_x0) / torch.sqrt(1 - alpha_hat)

                if i > 0:
                    pred_dir_xt = torch.sqrt(1 -
                                             alpha_hat_prev) * predicted_noise
                    x_prev = torch.sqrt(
                        alpha_hat_prev) * predicted_x0 + pred_dir_xt
                else:
                    x_prev = predicted_x0

                x = x_prev

            return x

    #用于长序列拼接时，对中间某一段施加软 mask，让生成结果和已有片段平滑混合。
    def inpaint_soft(self, predicted_x0, x0):
        mask = torch.ones([60]).to(self.device).float()
        mask[10:20] = torch.linspace(0.80, 0.1, 10).to(self.device)
        mask[20:30] = 0.1
        mask[30:40] = torch.linspace(0.1, 0.8, 10).to(self.device)
        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(x0.shape[0], 1, x0.shape[2])
        predicted_x0 = predicted_x0 * (1. - mask) + x0 * mask

        return predicted_x0

    # ==============================================================
    # 新增，确保生成轨迹别偏离目标太多
    def inpaint_cond(self, x0, target_goal_rel=None):#target_goal=预测出的pre_goal_rel
        x0[:, 0, :3] = 0.0 # 强制首帧相对位移为0
        
        if self.use_goal and target_goal_rel is not None:
            pred_rel = torch.sum(x0[:, :, :3], dim=1) # 扩散模型当前生成的相对位移
            rel_residual = (target_goal_rel - pred_rel).unsqueeze(1)
            x0[:, :, :3] = x0[:, :, :3] + rel_residual / x0.shape[1]#残差均摊
        return x0
    # ==============================================================

    def hand_shake(self, x0):#对相邻片段前后 10 帧做线性混合
        mask = torch.linspace(1.0, 0.0, 10).to(self.device)
        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(x0.shape[0] - 1, 1, x0.shape[2])

        x0_prev = x0[:-1, -10:, :].clone()
        x0_next = x0[1:, :10, :].clone()
        x0[:-1, -10:, :] = x0_prev * mask + (1.0 - mask) * x0_next
        x0[1:, :10, :] = x0_prev * mask + (1.0 - mask) * x0_next

        return x0

    def smooth_motion(self, samples):#用 DCT / IDCT 对动作做频域平滑
        dct, idct = get_dct_matrix(samples.shape[2])
        dct = dct.to(samples.device)
        idct = idct.to(samples.device)
        dct_frames = samples.shape[2] // 6
        dct = dct[:dct_frames, :]
        idct = idct[:, :dct_frames]
        samples = idct @ (dct @ samples)
        return samples

    @torch.no_grad()
    def sample(self,
            batch_size,
            cond_embed,
            num_samples=50,
            target_goal_rel=None,
            hand_shake=False) -> torch.Tensor:
        samples = []
        for _ in range(num_samples):
            samples.append(
                self.sample_ddim_progressive(
                    batch_size,
                    cond_embed,
                    target_goal_rel=target_goal_rel,
                    hand_shake=hand_shake,
                )
            )
        samples = torch.stack(samples, dim=1)   # [B, num_samples, T, D]
        return samples

    def eval_step(self, batch: Dict) -> Dict:
        predictor_dict = self.predict_context(batch)
        cond_embed = self.get_condition(batch, predictor_dict)

        batch_size = batch["img"].shape[0]

        samples = self.sample(
            batch_size,
            cond_embed,
            self.num_sample,
            target_goal_rel=predictor_dict["tf_goal_rel"],
            hand_shake=False,
        )
        samples = self.smooth_motion(samples)

        out_dict = {}
        local_trans = samples[..., :3]
        out_dict["pred_global_orient"] = samples[..., 3:9]

        init_global_trans = predictor_dict["pre_init_pos"][:, None, None, :]
        pred_global_trans = torch.cumsum(local_trans, dim=-2)
        pred_global_trans = pred_global_trans + init_global_trans

        out_dict["pred_global_trans"] = pred_global_trans
        out_dict["pred_body_pose"] = samples[..., 9:]
        out_dict["pre_init_pos"] = predictor_dict["pre_init_pos"]
        out_dict["pre_goal_rel"] = predictor_dict["pre_goal_rel"]

        return out_dict

    def validation_step(self, batch: Dict) -> Dict:
        return self.eval_step(batch)

    def test_step(self, batch: Dict) -> Dict:
        return self.eval_step(batch)

    def predict_step(self, batch: Dict) -> Dict:#用于长序列预测
        predictor_dict = self.predict_context(batch)
        cond_embed = self.get_condition(batch, predictor_dict)

        batch_size = batch["img"].shape[0]

        samples = self.sample(
            batch_size,
            cond_embed,
            self.num_sample,
            target_goal_rel=predictor_dict["tf_goal_rel"],
            hand_shake=True,
        )

        current_samples = samples[0]

        for i in range(samples.shape[0] - 1):
            x0 = torch.cat(
                [current_samples[:, -30:, :], samples[i + 1, :, 10:40, :]],
                dim=1
            )

            noise = torch.randn_like(x0)
            t = torch.tensor([700]).to(self.device).long().repeat(self.num_sample)
            xt = self.q_sample(x0=x0, t=t, noise=noise)
            x0_pred = self.sample_ddim_progressive_partial(xt, x0)

            current_samples = torch.cat([
                current_samples[:, :-30, :], x0_pred, samples[i + 1, :, 41:, :]
            ], dim=1)

        samples = current_samples.unsqueeze(0)
        samples = self.smooth_motion(samples)

        out_dict = {}
        local_trans = samples[..., :3]
        out_dict["pred_global_orient"] = samples[..., 3:9]

        init_global_trans = predictor_dict["pre_init_pos"][:, None, None, :]# 修改为动态保留 Batch 维度
        pred_global_trans = torch.cumsum(local_trans, dim=-2)
        pred_global_trans = pred_global_trans + init_global_trans

        out_dict["pred_global_trans"] = pred_global_trans
        out_dict["pred_body_pose"] = samples[..., 9:]
        out_dict["pre_init_pos"] = predictor_dict["pre_init_pos"]
        out_dict["pre_goal_rel"] = predictor_dict["pre_goal_rel"]

        return out_dict
    
    def configure_optimizers(self):#给backbone和其余部分设置了不同的学习率
        lr = self.optimizer_conf["basic_lr_per_img"] * self.batch_size_per_device * self.gpus

        # Create a list of parameter groups with different learning rates
        param_groups = []
        param_group_1 = {'params': [], 'lr': lr * 0.1}
        param_group_2 = {'params': [], 'lr': lr}
        for name, param in self.named_parameters():
            if "backbone" in name:
                param_group_1['params'].append(param)
            else:
                param_group_2['params'].append(param)
        param_groups.append(param_group_1)
        param_groups.append(param_group_2)

        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=1e-7)

        scheduler = MultiStepLR(optimizer,
                                milestones=self.lr_scheduler_conf["milestones"],
                                gamma=self.lr_scheduler_conf["gamma"])
        return [[optimizer], [scheduler]]
