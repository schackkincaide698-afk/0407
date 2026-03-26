"""Our custom pedestrian dataset."""
import copy
import os
import pickle
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pedgen.utils.colors import IMG_MEAN, IMG_STD
from pedgen.utils.rot import (
    axis_angle_to_matrix,
    depth_to_3d,
    matrix_to_rotation_6d,
)

class CityWalkersDataset(Dataset):
    """Lightning dataset for pedestrian generation."""

    def __init__(
        self,
        label_file: str,
        mode: str,
        data_root: str,
        img_root: str,
        img_dim: List,
        min_timestamp: int,
        use_partial: bool,
        num_timestamp: int,
        depth_root: str,
        semantic_root: str,
        sample_interval: int,
        sample_start_idx: int,
        grid_size: List,
        grid_points: List,
        use_image: bool,
        use_data_augmentation: bool,
        train_percent: float,
    ) -> None:
        with open(os.path.join(data_root, label_file), "rb") as f:
            labels = pickle.load(f)#读取WHAM的.pkl文件
        self.label_list = []
        self.img_mean = np.array(IMG_MEAN)
        self.img_std = np.array(IMG_STD)
        self.img_w = img_dim[1]
        self.img_h = img_dim[0]
        self.mode = mode
        self.num_timestamp = num_timestamp
        self.min_timestamp = min_timestamp  # hardcoded
        self.use_partial = use_partial
        self.grid_size = grid_size
        self.use_image = use_image
        self.grid_points = grid_points
        self.use_data_augmentation = use_data_augmentation

        for idx, val in enumerate(labels):
            if np.isnan(val["global_trans"]).any() or np.isnan(
                    val["local_trans"]).any():
                continue
            image_path = os.path.join(data_root, img_root, val["image"])
            if self.mode != "pred":
                i = sample_start_idx
                max_i = val["global_trans"].shape[
                    0] - self.min_timestamp + 1 if self.use_partial and self.mode == "train" else val[
                        "global_trans"].shape[0] - self.num_timestamp + 1
                while i < max_i:
                    safe_path = image_path.replace('\\', '/')
                    img_id = int(image_path.split("/")[-1].split(".")[0]) + i
                    new_val = copy.deepcopy(val)
                    new_val["start_t"] = i
                    new_image_path = image_path[:-10] + str(img_id).zfill(
                        6) + ".jpg"
                    new_val["image"] = new_image_path
                    if not os.path.exists(new_val["image"]):
                        print(f"[Debug] 找不到图片被过滤: {new_val['image']}")  
                        break
                    depth_path = new_image_path.replace(img_root, depth_root)
                    depth_path = depth_path.replace("jpg", "png")
                    new_val["depth"] = depth_path

                    semantic_path = new_image_path.replace(
                        img_root, semantic_root)
                    semantic_path = semantic_path.replace("jpg", "png")
                    new_val["semantic"] = semantic_path

                    voxel_path = new_image_path.replace(img_root, "voxel")
                    voxel_path = voxel_path.replace("jpg", "npy")
                    new_val["voxel"] = voxel_path
                    new_val["index"] = idx

                    img_id = new_val["image"].split("/")
                    img_id = img_id[-2] + "_" + img_id[-1][:-4] + "_" + str(idx)
                    img_id = img_id[-2] + "_" + \
                    img_id[-1][:-4] + "_" + str(idx)
                    self.label_list.append(new_val)
                    i += sample_interval
            else:
                val["start_t"] = 0
                val["image"] = image_path
                if not os.path.exists(val["image"]):
                    continue
                depth_path = image_path.replace(img_root, depth_root)
                depth_path = depth_path.replace("jpg", "png")
                val["depth"] = depth_path
                semantic_path = image_path.replace(img_root, semantic_root)
                semantic_path = semantic_path.replace("jpg", "png")
                val["semantic"] = semantic_path
                voxel_path = image_path.replace(img_root, "voxel")
                voxel_path = voxel_path.replace("jpg", "npy")
                val["voxel"] = voxel_path
                # ground_path = image_path.replace(img_root, "ground")
                # ground_path = ground_path.replace("jpg", "npy")
                # val["ground"] = ground_path
                self.label_list.append(val)

        if self.mode == "train" and train_percent < 1.0:
            self.label_list = self.label_list[:int(
                len(self.label_list) * train_percent)]

    def __len__(self) -> int:
        return len(self.label_list)
    
    # 每个点是4维：[x,y,z,semantic]
    def load_scene_points_raw(self, label, intrinsics_old):
        tt = lambda x: torch.from_numpy(x).float()

        try:
            depth_3d = np.load(label["voxel"])
            depth_3d = tt(depth_3d)  # [H, W, 4]
        except Exception:
            depth = cv2.imread(label["depth"], -1)
            depth = np.array(depth, dtype=np.float32) / 256.0
            depth_3d = depth_to_3d(depth, intrinsics_old)
            depth_3d = tt(depth_3d)

            semantic_raw = cv2.imread(label["semantic"], -1)
            semantic_raw = np.array(semantic_raw, dtype=np.float32) / 18.0
            semantic_raw = tt(semantic_raw).unsqueeze(-1)

            depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)

            voxel_dir = label["voxel"].rsplit("/", 1)[0]
            os.makedirs(voxel_dir, exist_ok=True)
            np.save(label["voxel"], depth_3d.cpu().numpy())

        points = depth_3d.reshape(-1, 4)

        valid = torch.isfinite(points).all(dim=-1)
        valid = valid & (points[:, 2] > 1e-5)
        points = points[valid]

        #现在的“均匀下采样”只是按一维展开顺序抽点，不是真正空间均匀
        num_target = 4096
        num_points = points.shape[0]

        if num_points == 0:
            # 极小概率的容错：深度图全黑
            return torch.zeros((num_target, 4), dtype=torch.float32)

        if num_points >= num_target:
            # 确定性均匀下采样：按比例步长截取，最大程度保留全局空间分布
            step = num_points / num_target
            indices = (torch.arange(num_target).float() * step).long()
            points = points[indices]
        else:
            # 确定性填充：顺序重复已有序列，不引入额外的随机噪声
            pad_indices = torch.arange(num_target - num_points) % num_points
            points = torch.cat([points, points[pad_indices]], dim=0)

        return points


    #全改
    def __getitem__(self, index: int) -> Dict:
        label = self.label_list[index]
        tt = lambda x: torch.from_numpy(x).float()
        data_dict = {}

        img_id = label["image"].split("/")
        img_id = img_id[-2] + "_" + img_id[-1][:-4] + "_" + str(label["index"])
        meta = {"source": "pedmotion", "img_id": img_id}
        data_dict["meta"] = meta

        rgb = cv2.imread(label["image"])
        if rgb is None:
            raise FileNotFoundError(f"Failed to read image: {label['image']}")

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32)
        mean = self.img_mean.reshape(1, 1, 3)
        std = self.img_std.reshape(1, 1, 3)
        rgb = (rgb - mean) / std
        img = torch.from_numpy(rgb).float().permute(2, 0, 1)

        f = (1280**2 + 720**2)**0.5
        cx = 0.5 * 1280
        cy = 0.5 * 720
        intrinsics_old = np.eye(3)
        intrinsics_old[0, 0] = f
        intrinsics_old[1, 1] = f
        intrinsics_old[0, 2] = cx
        intrinsics_old[1, 2] = cy

        start_t = label["start_t"]

        global_orient_source = axis_angle_to_matrix(
            tt(label["global_orient"][start_t])
        )
        global_orient_target = axis_angle_to_matrix(
            tt(label["local_orient"][start_t])
        )

        transl_source = tt(label["global_trans"][[start_t]])
        transl_target = tt(label["local_trans"][[start_t]])

        source_to_target_rotation = global_orient_target @ global_orient_source.T

        global_orient = axis_angle_to_matrix(
            tt(label["global_orient"][start_t:, :3])
        )
        global_orient = source_to_target_rotation @ global_orient

        source_to_target_translation = (
            transl_target.T - source_to_target_rotation @ transl_source.T
        )

        transl = tt(label["global_trans"][start_t:])
        transl = source_to_target_rotation @ transl.T + source_to_target_translation
        transl = transl.T

        data_dict["img"] = img
        data_dict["intrinsics"] = tt(intrinsics_old)
        data_dict["global_trans"] = transl[:self.num_timestamp]
        data_dict["motion_mask"] = torch.zeros((self.num_timestamp,), dtype=torch.bool)

        data_dict["global_orient"] = matrix_to_rotation_6d(
            global_orient[:self.num_timestamp]
        )
        data_dict["betas"] = torch.mean(
            tt(label["betas"][start_t:start_t + self.num_timestamp]),
            dim=0
        )

        data_dict["body_pose"] = matrix_to_rotation_6d(
            axis_angle_to_matrix(
                tt(label["body_pose"][start_t:start_t + self.num_timestamp])
            )
        ).reshape(-1, 23 * 6)

        if data_dict["global_trans"].shape[0] < self.num_timestamp:
            motion_length = data_dict["global_trans"].shape[0]
            data_dict["motion_mask"][motion_length:] = True
            data_dict["global_trans"] = torch.cat([
                data_dict["global_trans"],
                torch.zeros(self.num_timestamp - motion_length, 3)
            ], dim=0)
            data_dict["global_orient"] = torch.cat([
                data_dict["global_orient"],
                torch.zeros(self.num_timestamp - motion_length, 6)
            ], dim=0)
            data_dict["body_pose"] = torch.cat([
                data_dict["body_pose"],
                torch.zeros(self.num_timestamp - motion_length, 23 * 6)
            ], dim=0)

        valid_len = self.num_timestamp - int(data_dict["motion_mask"].sum().item())
        valid_len = max(valid_len, 1)

        data_dict["gt_init_pos"] = data_dict["global_trans"][0].clone()
        data_dict["gt_goal_rel"] = (
            data_dict["global_trans"][valid_len - 1] - data_dict["global_trans"][0]
        ).clone()

        if not self.use_image:
            return data_dict

        data_dict["grid_size"] = torch.tensor(self.grid_size).float()
        data_dict["grid_points"] = torch.tensor(self.grid_points).long()
        data_dict["scene_points_raw"] = self.load_scene_points_raw(label, intrinsics_old)

        return data_dict


def collate_fn_pedmotion(data):
    img_batch = []
    intrinsics_batch = []
    global_trans_batch = []
    betas_batch = []
    global_orient_batch = []
    body_pose_batch = []
    meta_batch = []
    motion_mask_batch = []
    gt_init_pos_batch = []
    gt_goal_rel_batch = []
    scene_points_raw_batch = []

    for data_dict in data:
        img_batch.append(data_dict["img"])
        intrinsics_batch.append(data_dict["intrinsics"])
        global_trans_batch.append(data_dict["global_trans"])
        betas_batch.append(data_dict["betas"])
        global_orient_batch.append(data_dict["global_orient"])
        body_pose_batch.append(data_dict["body_pose"])
        meta_batch.append(data_dict["meta"])

        if "motion_mask" in data_dict:
            motion_mask_batch.append(data_dict["motion_mask"])
        if "gt_init_pos" in data_dict:
            gt_init_pos_batch.append(data_dict["gt_init_pos"])
        if "gt_goal_rel" in data_dict:
            gt_goal_rel_batch.append(data_dict["gt_goal_rel"])
        if "scene_points_raw" in data_dict:
            scene_points_raw_batch.append(data_dict["scene_points_raw"])

    ret_dict = {
        "img": torch.stack(img_batch),
        "intrinsics": torch.stack(intrinsics_batch),
        "meta": meta_batch,
        "global_trans": torch.stack(global_trans_batch),
        "betas": torch.stack(betas_batch),
        "global_orient": torch.stack(global_orient_batch),
        "body_pose": torch.stack(body_pose_batch),
        "batch_size": len(img_batch),
        "gt_init_pos": torch.stack(gt_init_pos_batch),
        "gt_goal_rel": torch.stack(gt_goal_rel_batch),
        "scene_points_raw": torch.stack(scene_points_raw_batch),
        "grid_size": data[0]["grid_size"],
        "grid_points": data[0]["grid_points"],
    }

    if len(motion_mask_batch) > 0:
        ret_dict["motion_mask"] = torch.stack(motion_mask_batch)

    return ret_dict

def collate_fn_pedmotion_pred(data):
    """Custom collate function."""
    return collate_fn_pedmotion(data)
