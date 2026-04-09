"""Utilities for rebuilding occupancy-grid conditions from raw scene points."""
from typing import Dict, List, Sequence, Union

import torch

from pedgen.utils.rot import create_2d_grid, create_occupancy_grid


class OccupancyGridBuilder:
    """Rebuild occupancy-grid conditions from raw scene points and predicted init positions."""

    def __init__(self, batch: Dict, device: torch.device) -> None:
        if "scene_points_raw" not in batch:
            raise KeyError("scene_points_raw is required to rebuild occupancy grids")
        if "grid_size" not in batch or "grid_points" not in batch:
            raise KeyError("grid_size and grid_points are required")

        self.device = device
        self.scene_points_batch = self._normalize_scene_points_batch(
            batch["scene_points_raw"]
        )
        self.grid_size = self._to_list(batch["grid_size"])
        self.grid_points = self._to_list(batch["grid_points"])

        self.grid_2d = torch.from_numpy(
            create_2d_grid(
                num_points=self.grid_points,
                grid_size=self.grid_size,
            )
        ).float().to(self.device)

    def build(self,
              pre_init_pos: torch.Tensor,
              pre_goal_rel: torch.Tensor = None,
              is_sequence: bool = False) -> torch.Tensor:
        init_pos_seq = self._build_init_pos_seq(
            pre_init_pos,
            pre_goal_rel,
            is_sequence=is_sequence,
        )

        occupancy_batch = []
        for scene_points, current_init_pos in zip(self.scene_points_batch, init_pos_seq):
            occupancy_batch.append(self._build_single(scene_points, current_init_pos))

        return torch.stack(occupancy_batch, dim=0)

    def _build_single(self,
                      scene_points_raw: torch.Tensor,
                      current_init_pos: torch.Tensor) -> torch.Tensor:
        # rot.py 中 create_occupancy_grid 依赖 CPU 逻辑，先统一放到 CPU
        scene_points_raw = scene_points_raw.float().cpu()
        current_init_pos = current_init_pos.float().cpu()

        shifted_scene_points = scene_points_raw.clone()
        shifted_scene_points[..., :3] = (
            shifted_scene_points[..., :3] - current_init_pos.unsqueeze(0)
        )

        grid_mask = (
            (shifted_scene_points[:, 0] >= self.grid_size[0] + 1e-5) &
            (shifted_scene_points[:, 0] < self.grid_size[1] - 1e-5) &
            (shifted_scene_points[:, 1] >= self.grid_size[2] + 1e-5) &
            (shifted_scene_points[:, 1] < self.grid_size[3] - 1e-5) &
            (shifted_scene_points[:, 2] >= self.grid_size[4] + 1e-5) &
            (shifted_scene_points[:, 2] < self.grid_size[5] - 1e-5)
        )#离散操作，导数处处为0
        shifted_scene_points = shifted_scene_points[grid_mask]

        if shifted_scene_points.shape[0] == 0:
            shifted_scene_points = torch.zeros((1, 4), dtype=torch.float32)

        occupancy_grid = create_occupancy_grid(
            shifted_scene_points,
            self.grid_size,
            self.grid_points,
        ).to(self.device)

        occupancy_grid = occupancy_grid.permute(0, 2, 1)
        occupancy_grid = torch.cat([occupancy_grid, self.grid_2d], dim=-1)
        occupancy_grid = occupancy_grid.reshape(
            occupancy_grid.shape[0] * occupancy_grid.shape[1], -1
        )
        return occupancy_grid

    def _build_init_pos_seq(self,
                            pre_init_pos: torch.Tensor,
                            pre_goal_rel: torch.Tensor,
                            is_sequence: bool) -> torch.Tensor:
        if not is_sequence:
            return pre_init_pos

        if pre_goal_rel is None:
            raise ValueError("pre_goal_rel is required when is_sequence=True")

        cumulative_goal_rel = torch.cumsum(pre_goal_rel, dim=0)
        return torch.cat(
            [pre_init_pos[[0]], pre_init_pos[[0]] + cumulative_goal_rel[:-1]],
            dim=0,
        )

    def _normalize_scene_points_batch(
            self,
            scene_points_batch: Union[torch.Tensor, Sequence]) -> List[torch.Tensor]:
        if isinstance(scene_points_batch, torch.Tensor):
            # 若误传成单个 tensor，按 batch_size=1 处理
            if scene_points_batch.ndim == 2:
                return [scene_points_batch]
            return [scene_points_batch[i] for i in range(scene_points_batch.shape[0])]

        normalized_batch = []
        for scene_points in scene_points_batch:
            if isinstance(scene_points, torch.Tensor):
                normalized_batch.append(scene_points.float())
            else:
                normalized_batch.append(torch.as_tensor(scene_points).float())
        return normalized_batch

    def _to_list(self, val: Union[torch.Tensor, Sequence, int, float]) -> List:
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().tolist()
        if isinstance(val, (list, tuple)):
            return list(val)
        return [val]