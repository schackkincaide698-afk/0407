"""Lightning wrapper of the pytorch model."""
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
#from pedgen.utils.rot import (create_ground_map, positional_encoding_2d, rotation_6d_to_matrix)
from pedgen.utils.rot import positional_encoding_2d, rotation_6d_to_matrix

#改用transformer
class predictor(nn.Module):
    def __init__(self, latent_dim: int, use_image: bool = False) -> None:
        super().__init__()
        self.use_image = use_image
        self.num_semantic_classes = 19
        self.num_plan_steps = 4
        sem_dim = latent_dim // 2
        xyz_dim = latent_dim - sem_dim
        self.scene_xyz_embed = nn.Sequential(
            nn.Linear(3, xyz_dim),
            nn.ReLU(inplace=True),
            nn.Linear(xyz_dim, xyz_dim),
        )

        self.scene_semantic_embed = nn.Embedding(self.num_semantic_classes, sem_dim)
        #把坐标特征和语义特征拼接后融合(B, N, D)=(batch size,点数,latent_dim)
        self.scene_token_embed = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.scene_memory_norm = nn.LayerNorm(latent_dim)
        #场景编码器，对所有场景点token做self-attention
        self.scene_memory_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=latent_dim * 4,
                dropout=0.1,
                batch_first=True,#输入格式为(B, seq_len, feature_dim)
                activation="gelu",
            ),
            num_layers=2,#2层Transformer Encoder
        )

        self.autoreg_pos_embed = nn.Parameter(torch.randn(1, self.num_plan_steps, latent_dim) * 0.02)
        self.autoreg_input_embed = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.autoreg_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.autoreg_out_norm = nn.LayerNorm(latent_dim)
        self.autoreg_out_head = nn.Linear(latent_dim, 3)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        if "scene_points_raw" not in batch:
            raise RuntimeError("scene_points_raw is required for predictor")
        scene_points = batch["scene_points_raw"].to(self.autoreg_pos_embed.device)#(B, N, 4)
        scene_xyz = scene_points[..., :3]#(B, N, 3)取前三维坐标
        scene_semantic_idx = scene_points[..., 3].long().clamp(min=0, max=self.num_semantic_classes - 1)#(B, N)取第4维语义类别编号

        scene_xyz_feat = self.scene_xyz_embed(scene_xyz)#(B, N, xyz_dim)
        scene_sem_feat = self.scene_semantic_embed(scene_semantic_idx)#(B, N, sem_dim)
        scene_tokens = self.scene_token_embed(torch.cat([scene_xyz_feat, scene_sem_feat], dim=-1))#(B, N, D)
        scene_memory = self.scene_memory_encoder(self.scene_memory_norm(scene_tokens))#经过2层Transformer Encoder后，每个场景点token都融合了全局场景上下文，得到 scene_memory

        if "gt_init_pos" in batch:
            current_pos = batch["gt_init_pos"].to(scene_tokens.device)
        elif "global_trans" in batch:
            current_pos = batch["global_trans"][:, 0, :].to(scene_tokens.device)
        elif "gt_init_pos_seq" in batch:
            current_pos = batch["gt_init_pos_seq"][:, 0, :].to(scene_tokens.device)
        else:
            raise RuntimeError("gt_init_pos or global_trans is required for predictor")

        gt_init_pos_seq = batch.get("gt_init_pos_seq", None)
        gt_init_pos_seq_mask = batch.get("gt_init_pos_seq_mask", None)
        if gt_init_pos_seq is not None:
            gt_init_pos_seq = gt_init_pos_seq.to(scene_tokens.device)
        if gt_init_pos_seq_mask is not None:
            gt_init_pos_seq_mask = gt_init_pos_seq_mask.to(scene_tokens.device)

        scheduled_sampling_prob = float(batch.get("scheduled_sampling_prob", 0.0))

        pred_init_pos_seq_output = [current_pos]  # 纯净预测序列：专供外部算loss和建图（无GT）
        feedback_history_list = [current_pos]     # 混合历史序列：专供内部 Transformer 循环输入
        pred_goal_rel_seq = []

        for step_idx in range(self.num_plan_steps):
            # 构建输入 (使用包含 GT 的混合历史)
            history_pos = torch.stack(feedback_history_list, dim=1)
            decoder_input = self.autoreg_input_embed(history_pos)
            decoder_input = decoder_input + self.autoreg_pos_embed[:, :decoder_input.shape[1], :]
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                decoder_input.shape[1],
                device=decoder_input.device,
            )
            
            decoded = self.autoreg_decoder(tgt=decoder_input, memory=scene_memory, tgt_mask=tgt_mask)
            pred_goal_rel = self.autoreg_out_head(self.autoreg_out_norm(decoded[:, -1, :]))
            pred_goal_rel_seq.append(pred_goal_rel)

            # 当前的真实基准点 + 预测出的相对位移 = 纯物理预测的下一位置
            next_pred_pos = feedback_history_list[-1] + pred_goal_rel

            # 记录预测结果，防止外层Loss算错
            if step_idx < self.num_plan_steps - 1:
                pred_init_pos_seq_output.append(next_pred_pos)

            # Teacher Forcing与下一轮状态装载
            if (self.training and gt_init_pos_seq is not None and step_idx < self.num_plan_steps - 1):
                use_pred_mask = torch.rand(current_pos.shape[0], 1, device=scene_tokens.device) < scheduled_sampling_prob
                if gt_init_pos_seq_mask is not None:
                    # 如果GT该段失效，强制要求模型使用预测值往下走
                    use_pred_mask = use_pred_mask | (gt_init_pos_seq_mask[:, step_idx + 1:step_idx + 2] < 0.5)
                next_pos_for_feedback = torch.where(use_pred_mask, next_pred_pos, gt_init_pos_seq[:, step_idx + 1, :])
            else:
                next_pos_for_feedback = next_pred_pos
            
            if step_idx < self.num_plan_steps - 1:
                feedback_history_list.append(next_pos_for_feedback)

        # 返回的是纯净的、没有被 GT 污染的预测序列
        pred_init_pos_seq = torch.stack(pred_init_pos_seq_output, dim=1)
        pred_goal_rel_seq = torch.stack(pred_goal_rel_seq, dim=1)

        return {
                    "pred_init_pos": pred_init_pos_seq[:, 0, :],#首段绝对起点
                    "pred_init_pos_seq": pred_init_pos_seq,#全部4段轨迹的绝对起点序列
                    "pred_goal_rel": pred_goal_rel_seq[:, 0, :],#首段相对位移
                    "pred_goal_rel_seq": pred_goal_rel_seq,#全部4段相对位移序列
                }



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
            self.predictor = predictor(diffuser_conf["latent_dim"],use_image=use_image)

            self.criterion = F.mse_loss#重建损失用 MSE
            self.criterion_traj = F.l1_loss#轨迹损失用 L1
            self.criterion_goal = F.l1_loss#起点/目标损失用 L1
            self.env_loss_weight = 0.1

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
            self.ddim_timestep_predv_seq = np.append(np.array([0]),
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
            x_t[batch["motion_mask"] == 1] = self.mask_embed

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
            'loss_init_seq': predictor_dict['loss_init_seq'],
            'loss_goal': predictor_dict['loss_goal'],
            'loss_goal_seq': predictor_dict['loss_goal_seq'],
            'loss_env': predictor_dict['loss_env'],
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
        loss_dict["loss"] += (
            predictor_dict["loss_init_seq"] +
            predictor_dict["loss_goal_seq"]
        ) * 0.05
        loss_dict["loss"] += predictor_dict["loss_env"] * self.env_loss_weight
        loss_dict.update({
            "pred_init_pos": predictor_dict["pred_init_pos"],
            "pred_goal_rel": predictor_dict["pred_goal_rel"],
        })
        return loss_dict

    #========================================
    # 把predictor的输出，整理成扩散模型真正要用的条件
    def predict_context(self, batch: Dict) -> Dict[str, torch.Tensor]:
        # predictor_output = self.predictor(batch)#提取预测值
        gt_init_pos = batch.get("gt_init_pos", None)
        gt_init_pos_seq = batch.get("gt_init_pos_seq", None)
        gt_init_pos_seq_mask = batch.get("gt_init_pos_seq_mask", None)#数据集提供，若起点在原始轨迹长度范围内，GT 有效
        gt_goal_rel = batch.get("gt_goal_rel", None)
        gt_goal_rel_seq = batch.get("gt_goal_rel_seq", None)
        gt_goal_rel_seq_mask = batch.get("gt_goal_rel_seq_mask", None)#终点end_idx在有效轨迹范围内，目标位移 GT 有效

        if gt_init_pos is not None:
            gt_init_pos = gt_init_pos.to(self.device)
        if gt_init_pos_seq is not None:
            gt_init_pos_seq = gt_init_pos_seq.to(self.device)
        if gt_init_pos_seq_mask is not None:
            gt_init_pos_seq_mask = gt_init_pos_seq_mask.to(self.device)
        if gt_goal_rel is not None:
            gt_goal_rel = gt_goal_rel.to(self.device)
        if gt_goal_rel_seq is not None:
            gt_goal_rel_seq = gt_goal_rel_seq.to(self.device)
        if gt_goal_rel_seq_mask is not None:
            gt_goal_rel_seq_mask = gt_goal_rel_seq_mask.to(self.device)

        predictor_batch = batch
        if self.training and gt_init_pos_seq is not None and gt_goal_rel_seq is not None:
            current_epoch = getattr(self, "current_epoch", 0)
            max_epochs = getattr(getattr(self, "trainer", None), "max_epochs", 1) or 1
            epoch_ratio = float(current_epoch) / float(max_epochs)
            if epoch_ratio < 0.3:
                scheduled_sampling_prob = 0.0
            elif epoch_ratio < 0.7:
                scheduled_sampling_prob = (epoch_ratio - 0.3) / 0.4
            else:
                scheduled_sampling_prob = 1.0
            predictor_batch = dict(batch)
            predictor_batch["scheduled_sampling_prob"] = scheduled_sampling_prob

        predictor_output = self.predictor(predictor_batch)#提取预测值
        # 只有在有 GT 时才计算 Loss（训练和验证阶段）
        if gt_init_pos is not None and gt_goal_rel is not None:
            loss_init = self.criterion_goal(predictor_output["pred_init_pos"], gt_init_pos)
            if gt_init_pos_seq is not None:
                if gt_init_pos_seq_mask is not None:
                    init_mask = gt_init_pos_seq_mask.unsqueeze(-1)
                    init_diff = torch.abs(predictor_output["pred_init_pos_seq"] - gt_init_pos_seq)
                    init_denom = torch.clamp(init_mask.sum() * init_diff.shape[-1], min=1.0)
                    loss_init_seq = (init_diff * init_mask).sum() / init_denom
                else:
                    loss_init_seq = self.criterion_goal(predictor_output["pred_init_pos_seq"], gt_init_pos_seq)
            else:
                loss_init_seq = torch.tensor(0.0, device=self.device)#预测的4个起点，和GT4个起点之间的平均L1误差，只统计有效段
            loss_goal = self.criterion_goal(predictor_output["pred_goal_rel"], gt_goal_rel)
            if gt_goal_rel_seq is not None:
                if gt_goal_rel_seq_mask is not None:
                    seq_mask = gt_goal_rel_seq_mask.unsqueeze(-1)
                    seq_diff = torch.abs(predictor_output["pred_goal_rel_seq"] - gt_goal_rel_seq)
                    denom = torch.clamp(seq_mask.sum() * seq_diff.shape[-1], min=1.0)
                    loss_goal_seq = (seq_diff * seq_mask).sum() / denom
                else:
                    loss_goal_seq = self.criterion_goal(predictor_output["pred_goal_rel_seq"], gt_goal_rel_seq)
            else:
                loss_goal_seq = torch.tensor(0.0, device=self.device)#预测的4段目标位移，和GT4段目标位移之间的平均有效L1误差
        else:
            # 纯推理阶段，给默认值 0 防止返回的字典缺少键值
            loss_init = torch.tensor(0.0, device=self.device)
            loss_init_seq = torch.tensor(0.0, device=self.device)
            loss_goal = torch.tensor(0.0, device=self.device)
            loss_goal_seq = torch.tensor(0.0, device=self.device)
        
        predictor_output["loss_init"] = loss_init
        predictor_output["loss_init_seq"] = loss_init_seq
        predictor_output["loss_goal"] = loss_goal
        predictor_output["loss_goal_seq"] = loss_goal_seq
        predictor_output["loss_env"] = self.compute_walkability_loss(predictor_output, batch)#预测出来的4个起点，是否大致贴近场景地面、且没有跑出场景边界

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
                predictor_output["pred_init_pos_seq"][:, 0, :].detach(),
            )#.detach()是为了不让扩散网络的梯度传回predictor
            
            predictor_output["tf_goal_rel"] = torch.where(
                use_gt_mask,
                gt_goal_rel,
                predictor_output["pred_goal_rel_seq"][:, 0, :].detach(),
            )

            # new_img不能再使用gt_new_img，只能用预测的起点在线构建
        else:
            predictor_output["tf_init_pos"] = predictor_output["pred_init_pos_seq"][:, 0, :]
            predictor_output["tf_goal_rel"] = predictor_output["pred_goal_rel_seq"][:, 0, :]

        is_sequence = False
        predictor_output["pred_new_img"] = self.build_pred_new_img(
            batch,
            predictor_output["tf_init_pos"],
            predictor_output["tf_goal_rel"],
            is_sequence=is_sequence,
        )
            
        predictor_output["tf_new_img"] = predictor_output["pred_new_img"]
        return predictor_output

    def build_pred_new_img(self, batch: Dict, pred_init_pos: torch.Tensor,
                      pred_goal_rel: torch.Tensor, is_sequence: bool) -> torch.Tensor:
        occupancy_builder = OccupancyGridBuilder(batch, self.device)
        return occupancy_builder.build(pred_init_pos, pred_goal_rel, is_sequence=is_sequence)

    #Lightning的训练入口
    def training_step(self, batch: Dict) -> Dict:
        loss_dict = self(batch)#调用forward，得到loss，把其中标量项写入日志
        for key, val in loss_dict.items():
            # 过滤掉多维张量，只允许标量写入日志
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
        if "gt_init_pos" not in batch:
            batch["gt_init_pos"] = batch["global_trans"][:, 0, :]
        if "gt_goal_rel" not in batch:
            batch["gt_goal_rel"] = batch["global_trans"][:, -1, :] - batch["global_trans"][:, 0, :]
        local_trans = batch["global_trans"].clone()

        local_trans[:, 0, :] = 0
        local_trans[:, 1:, :] -= batch["global_trans"][:, :-1, :]

        local_orient = batch["global_orient"]

        full_motion = torch.cat([local_trans, local_orient, batch["body_pose"]],
                                dim=-1)
        return full_motion

    # 计算轨迹是否在可行走区域
    def compute_walkability_loss(self, predictor_output: Dict[str, torch.Tensor], batch: Dict) -> torch.Tensor:
        if "scene_points_raw" not in batch:
            return torch.tensor(0.0, device=self.device)
        
        if "pred_init_pos_seq" not in predictor_output or "pred_goal_rel_seq" not in predictor_output:
            return torch.tensor(0.0, device=self.device)

        scene_points = batch["scene_points_raw"].to(self.device)
        pred_init_pos_seq = predictor_output["pred_init_pos_seq"]
        pred_goal_rel_seq = predictor_output["pred_goal_rel_seq"]
        
        # 1. 连续轨迹插值 (将4个离散锚点扩展为连续轨迹)
        seg_end_pos_seq = pred_init_pos_seq + pred_goal_rel_seq
        num_interp = 20
        interp = torch.linspace(0.0, 1.0, num_interp, device=self.device).view(1, 1, num_interp, 1)
        traj_points = pred_init_pos_seq.unsqueeze(2) * (1.0 - interp) + seg_end_pos_seq.unsqueeze(2) * interp
        traj_points = traj_points.reshape(traj_points.shape[0], -1, 3) # [B, 4*num_interp, 3]

        # 仅提取水平维度 (X, Z) 进行区域约束
        traj_xz = traj_points[..., [0, 2]]
        
        losses = []
        MARGIN = 0.5  # 物理安全容差 (米)：允许行人在路面边缘0.5米内活动

        for b in range(scene_points.shape[0]):
            sp = scene_points[b]
            
            # 强制转换为整型，对齐 0~18 的离散标签体系
            semantic_labels = sp[:, 3].long()
            
            # 精准对齐数据集: 0=road, 1=sidewalk
            walkable_mask = ((semantic_labels == 0) | (semantic_labels == 1))
            valid_sp = sp[walkable_mask]

            #兜底逻辑：防止该视角下没有任何路面点云导致约束断裂
            if len(valid_sp) < 1:
                grid_size = batch["grid_size"][0] if batch["grid_size"].ndim > 1 else batch["grid_size"]
                x_min, x_max, _, _, z_min, z_max = grid_size.cpu().tolist()
                bound_penalty = (
                    torch.relu(x_min - traj_xz[b, :, 0]) +
                    torch.relu(traj_xz[b, :, 0] - x_max) +
                    torch.relu(z_min - traj_xz[b, :, 1]) + # 注意 traj_xz 的第1维是原图的 Z
                    torch.relu(traj_xz[b, :, 1] - z_max)
                )
                losses.append(bound_penalty.mean())
                continue

            # 绝对欧氏空间距离约束
            valid_xz = valid_sp[:, [0, 2]]  # [M, 2]
            
            # 计算轨迹点到所有可行走路面点的距离矩阵 [4*num_interp, M]
            dist_matrix = torch.cdist(traj_xz[b], valid_xz, p=2.0)
            
            # 提取每个轨迹点到最近路面点的距离
            min_dist, _ = torch.min(dist_matrix, dim=1)
            
            # 扣除安全半径，超出 MARGIN 的部分产生线性惩罚梯度
            walkability_loss = torch.relu(min_dist - MARGIN)
            losses.append(walkability_loss.mean())

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
        
    
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
                predv_t = (torch.ones(batch_size, device=self.device) *
                        self.ddim_timestep_predv_seq[i]).long()

                alpha_hat = self.alpha_hat[t][:, None, None]
                alpha_hat_predv = self.alpha_hat[predv_t][:, None, None]

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
                    pred_dir_xt = torch.sqrt(1 - alpha_hat_predv) * predicted_noise
                    x_predv = torch.sqrt(alpha_hat_predv) * predicted_x0 + pred_dir_xt
                else:
                    x_predv = predicted_x0

                x = x_predv

        return x

    # def sample_ddim_progressive_partial(self, xt, x0):
    #     """
    #     Generate samples from the model and yield samples from each timestep.

    #     Args are the same as sample_ddim()
    #     Returns a generator contains x_{predv_t}, shape as [sample_num, n_pred, 3 * joints_num]
    #     """
    #     sample_num = xt.shape[0]
    #     x = xt

    #     with torch.no_grad():
    #         for i in reversed(range(0, 70)):  # hardcoded as add noise t=100
    #             t = (torch.ones(sample_num) *
    #                  self.ddim_timestep_seq[i]).long().to(self.device)
    #             predv_t = (torch.ones(sample_num) *
    #                       self.ddim_timestep_predv_seq[i]).long().to(self.device)

    #             alpha_hat = self.alpha_hat[t][:, None, None]  # type: ignore
    #             alpha_hat_predv = self.alpha_hat[predv_t][  # type: ignore
    #                 :, None, None]

    #             predicted_x0 = self.diffuser(x, t, cond_embed=None)
    #             predicted_x0 = self.inpaint_soft(predicted_x0, x0)

    #             predicted_noise = (x - torch.sqrt(
    #                 (alpha_hat)) * predicted_x0) / torch.sqrt(1 - alpha_hat)

    #             if i > 0:
    #                 pred_dir_xt = torch.sqrt(1 -
    #                                          alpha_hat_predv) * predicted_noise
    #                 x_predv = torch.sqrt(
    #                     alpha_hat_predv) * predicted_x0 + pred_dir_xt
    #             else:
    #                 x_predv = predicted_x0

    #             x = x_predv

    #         return x

    # #用于长序列拼接时，对中间某一段施加软 mask，让生成结果和已有片段平滑混合。
    # def inpaint_soft(self, predicted_x0, x0):
    #     mask = torch.ones([60]).cuda().float()
    #     mask[10:20] = torch.linspace(0.80, 0.1, 10).cuda()
    #     mask[20:30] = 0.1
    #     mask[30:40] = torch.linspace(0.1, 0.8, 10).cuda()
    #     mask = mask.unsqueeze(0).unsqueeze(-1).repeat(x0.shape[0], 1, x0.shape[2])
    #     predicted_x0 = predicted_x0 * (1. - mask) + x0 * mask

    #     return predicted_x0

    # 确保生成轨迹别偏离目标太多
    def inpaint_cond(self, x0, target_goal_rel=None):#target_goal=预测出的pred_goal_rel
        x0[:, 0, :3] = 0.0 # 强制首帧相对位移为0
        if self.use_goal and target_goal_rel is not None:
            pred_rel = torch.sum(x0[:, :, :3], dim=1) # 扩散模型当前生成的相对位移
            rel_residual = (target_goal_rel - pred_rel).unsqueeze(1)
            x0[:, 1:, :3] = x0[:, 1:, :3] + rel_residual / (x0.shape[1] - 1)#残差均摊，但不分配给首帧，保证首帧相对位移为0
            x0[:, 0, :3] = 0.0
        return x0

    def hand_shake(self, x0):#对相邻片段前后 10 帧做线性混合
        mask = torch.linspace(1.0, 0.0, 10).cuda()
        mask = mask.unsqueeze(0).unsqueeze(-1).repeat(x0.shape[0] - 1, 1, x0.shape[2])

        x0_predv = x0[:-1, -10:, :].clone()
        x0_next = x0[1:, :10, :].clone()
        x0[:-1, -10:, :] = x0_predv * mask + (1.0 - mask) * x0_next
        x0[1:, :10, :] = x0_predv * mask + (1.0 - mask) * x0_next

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

        init_global_trans = predictor_dict["pred_init_pos"][:, None, None, :]
        pred_global_trans = torch.cumsum(local_trans, dim=-2)
        pred_global_trans = pred_global_trans + init_global_trans

        out_dict["pred_global_trans"] = pred_global_trans
        out_dict["pred_body_pose"] = samples[..., 9:]
        out_dict["pred_init_pos"] = predictor_dict["pred_init_pos"]
        out_dict["pred_init_pos_seq"] = predictor_dict["pred_init_pos_seq"]
        out_dict["pred_goal_rel"] = predictor_dict["pred_goal_rel"]
        out_dict["pred_goal_rel_seq"] = predictor_dict["pred_goal_rel_seq"]

        return out_dict

    def validation_step(self, batch: Dict) -> Dict:
        return self.eval_step(batch)

    def test_step(self, batch: Dict) -> Dict:
        return self.eval_step(batch)

    # 用于推理demo。提取Predictor输出的4个点，执行4次扩散生成
    # 然后将这4段生成结果转换到绝对世界坐标系下，执行30帧的线性软融合，最终截断并输出150帧的完整轨迹
    def predict_step(self, batch: Dict) -> Dict:
        predictor_dict = self.predict_context(batch)
        planned_init_pos_seq = predictor_dict["pred_init_pos_seq"]
        planned_goal_rel_seq = predictor_dict["pred_goal_rel_seq"]
        planned_init_pos = [planned_init_pos_seq[:, 0, :]]
        planned_goal_rel = [planned_goal_rel_seq[:, 0, :]]
        num_segments = planned_goal_rel_seq.shape[1]
        for i in range(1, num_segments):
            planned_init_pos.append(planned_init_pos_seq[:, i, :])
            planned_goal_rel.append(planned_goal_rel_seq[:, i, :])

        segment_samples = []
        for i in range(num_segments):#把4段动作合并
            seg_predictor_dict = dict(predictor_dict)
            seg_predictor_dict["pred_init_pos"] = planned_init_pos[i]
            seg_predictor_dict["tf_init_pos"] = planned_init_pos[i]
            seg_predictor_dict["pred_goal_rel"] = planned_goal_rel[i]
            seg_predictor_dict["tf_goal_rel"] = planned_goal_rel[i]
            seg_predictor_dict["pred_new_img"] = self.build_pred_new_img(
                batch,
                seg_predictor_dict["tf_init_pos"],
                seg_predictor_dict["tf_goal_rel"],
                is_sequence=False,
            )
            seg_predictor_dict["tf_new_img"] = seg_predictor_dict["pred_new_img"]
            cond_embed = self.get_condition(batch, seg_predictor_dict)
            seg_samples = self.sample(
                batch["img"].shape[0],
                cond_embed,
                self.num_sample,
                target_goal_rel=seg_predictor_dict["tf_goal_rel"],
                hand_shake=False,
            )
            segment_samples.append(seg_samples)

        current_samples = segment_samples[0].clone()
        # 此时维度为 [B, num_samples, 60, D]，对时间轴 dim=-2 进行积分
        current_samples[..., :3] = torch.cumsum(current_samples[..., :3], dim=-2) + planned_init_pos[0][:, None, None, :]

        for i in range(1, len(segment_samples)):
            next_samples = segment_samples[i].clone()
            next_samples[..., :3] = torch.cumsum(next_samples[..., :3], dim=-2) + planned_init_pos[i][:, None, None, :]#把4段局部相对运动全部映射回真实世界的绝对3D坐标系
            mask = torch.linspace(1.0, 0.0, 30, device=self.device).view(1, 1, 30, 1)# 在30帧内权重从1线性衰减至0的张量
            x0 = current_samples[..., -30:, :] * mask + next_samples[..., :30, :] * (1.0 - mask)

            # 提取前一段动作的后30帧与后一段动作的前30帧在时间轴（dim=-2）执行加权计算
            current_samples = torch.cat([
                current_samples[..., :-30, :], 
                x0, 
                next_samples[..., 30:, :]
            ], dim=-2)

        current_samples = current_samples[..., :150, :] # 截断时间轴到 150 帧

        # 还原回相对位移
        abs_trans = current_samples[..., :3].clone()
        current_samples[..., :3] = 0.0
        current_samples[..., 1:, :3] = abs_trans[..., 1:, :] - abs_trans[..., :-1, :]

        samples = current_samples 
        
        smoothed_samples = self.smooth_motion(samples)
        samples[..., 3:] = smoothed_samples[..., 3:]

        out_dict = {}

        local_trans = samples[..., :3]
        out_dict["pred_global_orient"] = samples[..., 3:9]

        init_global_trans = planned_init_pos[0][:, None, None, :]
        pred_global_trans = torch.cumsum(local_trans, dim=-2)
        pred_global_trans = pred_global_trans + init_global_trans

        out_dict["pred_global_trans"] = pred_global_trans
        out_dict["pred_body_pose"] = samples[..., 9:]
        
        # 直接传递首个片段的起点
        out_dict["pred_init_pos"] = planned_init_pos[0]
        out_dict["pred_goal_rel"] = planned_goal_rel[0]
        
        out_dict["pred_init_pos_seq"] = planned_init_pos_seq
        out_dict["pred_goal_rel_seq"] = planned_goal_rel_seq

        return out_dict
    
    # def configure_optimizers(self):#给backbone和其余部分设置了不同的学习率
    #     lr = self.optimizer_conf["basic_lr_per_img"] * self.batch_size_per_device * self.gpus

    #     # Create a list of parameter groups with different learning rates
    #     param_groups = []
    #     param_group_1 = {'params': [], 'lr': lr * 0.1}
    #     param_group_2 = {'params': [], 'lr': lr}
    #     for name, param in self.named_parameters():
    #         if "backbone" in name:
    #             param_group_1['params'].append(param)
    #         else:
    #             param_group_2['params'].append(param)
    #     param_groups.append(param_group_1)
    #     param_groups.append(param_group_2)

    #     optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=1e-7)

    #     scheduler = MultiStepLR(optimizer,
    #                             milestones=self.lr_scheduler_conf["milestones"],
    #                             gamma=self.lr_scheduler_conf["gamma"])
    #     return [[optimizer], [scheduler]]
    def configure_optimizers(self):
        lr = self.optimizer_conf["basic_lr_per_img"] * self.batch_size_per_device * self.gpus
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer,
                                milestones=self.lr_scheduler_conf["milestones"],
                                gamma=self.lr_scheduler_conf["gamma"])
        return [[optimizer], [scheduler]]