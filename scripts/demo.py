from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from smplx import SMPLLayer
from smplx.lbs import vertices2joints
from transformers import (SegformerFeatureExtractor,
                          SegformerForSemanticSegmentation)

from pedgen.model.pedgen_model import PedGenModel
from pedgen.utils.colors import IMG_MEAN, IMG_STD, get_colors
from pedgen.utils.renderer import Renderer
from pedgen.utils.rot import depth_to_3d, rotation_6d_to_matrix

IMAGE_PATH = "my_inputs/zebra.png"
NUM_pred_STEPS = 1
BETAS = np.zeros(10,)
CKPT_PATH = "experiments/pedgen/with_context/ckpts/epoch=199-step=7800.ckpt"
CFG_PATH = "cfgs/pedgen_with_context.yaml"
MAX_SCENE_POINTS = 4096

def normalize_rgb(rgb_bgr: np.ndarray) -> torch.Tensor:
    rgb = rgb_bgr.astype(np.float32)
    mean = np.float64(IMG_MEAN).reshape(1, -1)
    stdinv = 1 / np.float64(IMG_STD).reshape(1, -1)
    cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB, rgb)
    cv2.subtract(rgb, mean, rgb)
    cv2.multiply(rgb, stdinv, rgb)
    return torch.from_numpy(rgb).float().permute(2, 0, 1)

def build_intrinsics(width: int, height: int) -> np.ndarray:
    f = (width**2 + height**2)**0.5
    cx = 0.5 * width
    cy = 0.5 * height

    intrinsics_old = np.eye(3, dtype=np.float32)
    intrinsics_old[0, 0] = f
    intrinsics_old[1, 1] = f
    intrinsics_old[0, 2] = cx
    intrinsics_old[1, 2] = cy
    return intrinsics_old

def infer_depth(image_path: str, device: str) -> np.ndarray:
    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(device).eval()
    image = Image.open(image_path).convert("RGB")
    depth = model_zoe_nk.infer_pil(image)
    return depth.astype(np.float32)

def infer_semantic(image_path: str, device: str) -> np.ndarray:
    image_processor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    ).to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    pred = model(**inputs)
    logits = F.interpolate(
        pred.logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    segmentation = logits.argmax(dim=1)[0].cpu().numpy().astype(np.float32)
    return segmentation

def build_scene_points(depth: np.ndarray, segmentation: np.ndarray,
                       intrinsics: np.ndarray, max_points: int) -> torch.Tensor:
    depth_3d = depth_to_3d(depth, intrinsics)
    depth_3d = torch.from_numpy(depth_3d).float()

    # semantic_raw = torch.from_numpy(segmentation / 18.0).float().unsqueeze(-1)
    semantic_raw = torch.from_numpy(segmentation).float().unsqueeze(-1)
    scene_points = torch.cat([depth_3d, semantic_raw], dim=-1).reshape(-1, 4)
    valid_mask = torch.isfinite(scene_points).all(dim=-1) & (scene_points[:, 2] > 1e-5)
    scene_points = scene_points[valid_mask]
    
    num_target = max_points
    num_points = scene_points.shape[0]

    if num_points == 0:
        scene_points = torch.zeros((num_target, 4), dtype=torch.float32)
    elif num_points >= num_target:
        step = num_points / num_target
        indices = (torch.arange(num_target).float() * step).long()
        scene_points = scene_points[indices]
    else:
        pad_indices = torch.arange(num_target - num_points) % num_points
        scene_points = torch.cat([scene_points, scene_points[pad_indices]], dim=0)

    return scene_points

def vis_smpl_impl(render, img, pred_vertices, output_png, output_mp4):
    img_vis = img.copy()
    colors = get_colors()
    img_smpl, valid_mask = render.visualize_all(
        pred_vertices[[0, 10, 20, 30, 40, 50, 59]].cpu().numpy(),
        colors[[7, 6, 5, 4, 3, 2, 1]],
    )
    img_vis = img_smpl[:, :, :3] * valid_mask + (1 - valid_mask) * img_vis / 255.
    img_vis = (img_vis * 255).astype(np.uint8)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_png), img_vis)
    writer = imageio.get_writer(
        str(output_mp4),
        fps=30,
        mode='I',
        format='FFMPEG',
        macro_block_size=1,
    )

    for t in range(pred_vertices.shape[0]):
        img_pred = img.copy()
        img_smpl, valid_mask = render.visualize_all(pred_vertices[[t]].cpu().numpy(), colors[[1]])
        img_pred = img_smpl[:, :, :3] * valid_mask + (1 - valid_mask) * img_pred / 255.
        writer.append_data((img_pred * 255).astype(np.uint8))
    writer.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(CFG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    model_conf = config["model"]
    num_timestamp = config["data"]["num_timestamp"]
    grid_size = config["data"]["grid_size"]
    grid_points = config["data"]["grid_points"]

    model = PedGenModel.load_from_checkpoint(CKPT_PATH, **model_conf, map_location="cpu")
    model = model.to(device)
    model.eval()

    rgb = cv2.imread(IMAGE_PATH)
    if rgb is None:
        raise FileNotFoundError(f"Failed to read {IMAGE_PATH}")
    img = normalize_rgb(rgb)
    _, height, width = img.shape
    intrinsics = build_intrinsics(width, height)

    depth = infer_depth(IMAGE_PATH, device)
    segmentation = infer_semantic(IMAGE_PATH, device)
    scene_points = build_scene_points(depth, segmentation, intrinsics, MAX_SCENE_POINTS)

    num_pred_steps = NUM_pred_STEPS
    template_init_pos = torch.zeros(3)

    batch = {
        "img": img.unsqueeze(0).repeat(num_pred_steps, 1, 1, 1),
        "intrinsics": torch.from_numpy(intrinsics).float().unsqueeze(0).repeat(num_pred_steps, 1, 1),
        "global_trans": template_init_pos.unsqueeze(0).unsqueeze(0).repeat(num_pred_steps, num_timestamp, 1),
        "global_orient": torch.zeros(num_pred_steps, num_timestamp, 6),
        "body_pose": torch.zeros(num_pred_steps, num_timestamp, 23 * 6),
        "betas": torch.from_numpy(BETAS).float().unsqueeze(0).repeat(num_pred_steps, 1),
        "batch_size": num_pred_steps,
        "scene_points_raw": torch.stack([scene_points.clone() for _ in range(num_pred_steps)]),
        "grid_size": grid_size,
        "grid_points": grid_points,
    }

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        pred = model.predict_step(batch)
    pred = {k: pred[k].cpu() for k in pred}
    print("pred keys:", pred.keys())#打印预测值
    if "pred_init_pos" in pred:
        print("pred_init_pos shape:", pred["pred_init_pos"].shape)
        print("pred_init_pos[0]:", pred["pred_init_pos"][0])
    else:
        print("pred_init_pos not found")

    if "pred_goal_rel" in pred:
        print("pred_goal_rel shape:", pred["pred_goal_rel"].shape)
        print("pred_goal_rel[0]:", pred["pred_goal_rel"][0])
    else:
        print("pred_goal_rel not found")

    if "pred_init_pos" in pred and "pred_goal_rel" in pred:
        print("pred_goal_abs[0]:", pred["pred_init_pos"][0] + pred["pred_goal_rel"][0])

    smpl = SMPLLayer(model_path="smpl", gender='neutral')
    output_png = Path("my_output/zebra_output.png")
    output_mp4 = Path("my_output/zebra_output.mp4")

    b, n, t, _ = pred["pred_global_trans"].shape
    for i in range(b):
        body_pose = rotation_6d_to_matrix(pred["pred_body_pose"][i].reshape(-1, 23, 6))
        pred_transl = pred["pred_global_trans"][i]
        pred_rot = rotation_6d_to_matrix(pred["pred_global_orient"][i])

        pred_smpl_output = smpl(
            transl=pred_transl.reshape(-1, 3),
            betas=batch["betas"][i].unsqueeze(0).unsqueeze(0).repeat(n, t, 1).reshape(-1, 10).cpu(),
            global_orient=pred_rot.reshape(-1, 3, 3),
            body_pose=body_pose,
        )
        _ = vertices2joints(smpl.J_regressor, pred_smpl_output.vertices)

        intri = batch["intrinsics"][i].cpu().numpy()
        render = Renderer(
            focal_length=[intri[0, 0], intri[1, 1]],
            camera_center=[intri[0, 2], intri[1, 2]],
            img_res=[width, height],
            faces=smpl.faces,
            metallicFactor=0.0,
            roughnessFactor=0.7,
        )

        img_vis = batch["img"][i, :3].cpu().permute(1, 2, 0).numpy()
        img_vis = img_vis * np.array(IMG_STD)[None, None, :] + np.array(IMG_MEAN)[None, None, :]
        img_vis = cv2.resize(img_vis.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)

        pred_vertices = pred_smpl_output.vertices.reshape(n, t, -1, 3)
        vis_smpl_impl(render, img_vis, pred_vertices[0], output_png, output_mp4)
        del render
        break

if __name__ == "__main__":
    main()
