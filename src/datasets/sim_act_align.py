import os
import gc
import glob
import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import IterableDataset

# Action normalization stats for align-only dataset (99th percentile, symmetric)
# delta_pose(6) + gripper(1)
# Each dimension uses max(abs(p1), abs(p99)) so that normalized 0 = no movement.
# Original p99 (asymmetric):
#   min = [-0.6779, -0.5033, -0.4874, -0.00342, -0.000949, -0.00542, -1.0]
#   max = [+0.5353, +0.5128, +0.4601, +0.00292, +0.001237, +0.00326, -1.0]

#action_min_sim_align = [-0.677914559841156, -0.5127751231193542, -0.48736560344696045, -0.0034193717874586582, -0.0012368694879114628, -0.005416739732027054, -1.0]
#action_max_sim_align = [+0.677914559841156, +0.5127751231193542, +0.48736560344696045, +0.0034193717874586582, +0.0012368694879114628, +0.005416739732027054, -1.0]

action_min_sim_align = [-0.5957266092300415, -0.6034851670265198, -0.5240848660469055, -0.002589409239590168, -0.0008707013912498951, -0.003319802926853299, -1.0]
action_max_sim_align = [0.5957266092300415, 0.6034851670265198, 0.5240848660469055, 0.002589409239590168, 0.0008707013912498951, 0.003319802926853299, -1.0]

# Must match Save_dataset_align_only.py and sim_eval_align_only.py
TASK_INSTRUCTION = "Align the needle tip to the small grey circular trocar port on the eye model, next to the larger lens opening"


class SimActAlign(IterableDataset):
    """
    HDF5-based IterableDataset for fine-alignment simulation data.
    Same structure as SimAct but with align-only instruction and normalization.
    """

    def __init__(
        self,
        data_dir,
        dataset_name="sim_align",
        length=None,
        history_len=8,
        future_len=8,
        full_sequence=True,
        input_modality="image",
        view_mode="multi",
        load_future_image=False,
        future_image_mode="horizon",
        buffer_size=10000,
        cam_exterior="side_camera",
        cam_wrist="tool_camera",
        cam_top="top_camera",
        skip_history_padding=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.length = length
        self.history_len = history_len
        self.future_len = future_len
        self.skip_history_padding = skip_history_padding
        self.full_sequence = full_sequence
        self.input_modality = input_modality
        self.view_mode = view_mode
        self.load_future_image = load_future_image
        self.future_image_mode = future_image_mode
        self.buffer_size = buffer_size
        self.cam_exterior = cam_exterior
        self.cam_wrist = cam_wrist
        self.cam_top = cam_top

        # Action normalization to [-1, 1]
        self.action_min = np.array(action_min_sim_align, dtype=np.float32)
        self.action_max = np.array(action_max_sim_align, dtype=np.float32)

        # Collect all h5 files (recursive — searches subdirectories too)
        self.episode_paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True))
        if not self.episode_paths:
            raise FileNotFoundError(f"No .h5 files found in {data_dir} (recursive)")

    @staticmethod
    def _decode_jpeg(jpeg_data):
        """Decode JPEG vlen data to uint8 RGB numpy array."""
        if isinstance(jpeg_data, np.ndarray):
            buf = jpeg_data.flatten().astype(np.uint8)
        elif isinstance(jpeg_data, (bytes, bytearray)):
            buf = np.frombuffer(jpeg_data, dtype=np.uint8)
        else:
            buf = np.array(jpeg_data, dtype=np.uint8).flatten()

        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to decode JPEG (size={buf.size})")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _load_episode(self, h5_path):
        """Load a single episode from HDF5 file and return arrays."""
        with h5py.File(h5_path, "r") as f:
            traj_len = f["action"].shape[0]

            # --- Actions (N, 7): normalize delta_pose + gripper to [-1, 1] ---
            actions_np = f["action"][:].astype(np.float32)
            denominator = self.action_max - self.action_min
            denominator = np.where(denominator == 0, 1.0, denominator)
            actions_np = 2.0 * (actions_np - self.action_min) / denominator - 1.0
            actions_np = np.clip(actions_np, -1.0, 1.0)

            # --- Proprioception: ee_pose (N, 7) + sensor_dist (N, 1) → (N, 8) ---
            proprio_np = f["observations"]["ee_pose"][:].astype(np.float32)  # (N, 7)
            if "sensor_dist" in f["observations"]:
                sensor_dist = f["observations"]["sensor_dist"][:].astype(np.float32)  # (N,) or (N,1)
                if sensor_dist.ndim == 1:
                    sensor_dist = sensor_dist[:, None]  # (N, 1)
                # 20mm 클리핑 + 정규화: [0, 20] → [0, 1]
                sensor_dist = np.where((sensor_dist < 0) | (sensor_dist > 20.0), 20.0, sensor_dist)
                sensor_dist = sensor_dist / 20.0  # normalize to [0, 1]
                proprio_np = np.concatenate([proprio_np, sensor_dist], axis=-1)  # (N, 8)

            # --- Spatial auxiliary targets (backward compatible) ---
            spatial_targets_np = None
            if "keypoints_wrist" in f["observations"]:
                needle_tip = f["observations"]["needle_tip_pos"][:].astype(np.float32)
                trocar_entry = f["observations"]["trocar_entry_pos"][:].astype(np.float32)
                kp_wrist = f["observations"]["keypoints_wrist"][:].astype(np.float32)
                kp_vis = f["observations"]["keypoints_visibility"][:].astype(np.float32)
                phase_raw = f["phase"][:].astype(np.float32)

                dist = np.linalg.norm(trocar_entry - needle_tip, axis=-1, keepdims=True)
                dist_normalized = dist / 100.0
                phase_binary = np.clip(phase_raw - 1, 0, 1).reshape(-1, 1)

                spatial_targets_np = np.concatenate(
                    [kp_wrist, kp_vis, dist_normalized, phase_binary], axis=-1
                )

            # --- Action weight: uniform ---
            action_weight_np = np.ones(traj_len, dtype=np.float32)

            # --- Images: decode all frames for selected cameras ---
            img_grp = f["observations"]["images"]

            images_np = np.stack(
                [self._decode_jpeg(img_grp[self.cam_exterior][i]) for i in range(traj_len)],
                axis=0,
            )

            wrist_np = None
            top_np = None
            if self.view_mode == "multi":
                if self.cam_wrist in img_grp:
                    wrist_np = np.stack(
                        [self._decode_jpeg(img_grp[self.cam_wrist][i]) for i in range(traj_len)],
                        axis=0,
                    )
                else:
                    wrist_np = images_np.copy()
                if self.cam_top and self.cam_top in img_grp:
                    top_np = np.stack(
                        [self._decode_jpeg(img_grp[self.cam_top][i]) for i in range(traj_len)],
                        axis=0,
                    )

        return traj_len, actions_np, proprio_np, images_np, wrist_np, top_np, spatial_targets_np, action_weight_np

    def __iter__(self):
        # --- Shard by rank and worker ---
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id

        # Select episodes for this shard
        episode_paths = [p for i, p in enumerate(self.episode_paths) if i % total_shards == shard_index]
        if self.length is not None:
            episode_paths = episode_paths[: self.length]

        shuffle_buffer = []

        while True:  # Infinite loop — no epoch boundary, no buffer flush
            np.random.shuffle(episode_paths)

            for ep_path in episode_paths:
                try:
                    traj_len, actions_np, proprio_np, images_np, wrist_np, top_np, spatial_targets_np, action_weight_np = self._load_episode(ep_path)
                except Exception as e:
                    print(f"[Warn] Skipping {ep_path}: {e}")
                    continue

                if traj_len < self.history_len + 1:
                    del images_np, actions_np, proprio_np
                    if wrist_np is not None:
                        del wrist_np
                    continue

                # --- Sample indices ---
                if self.full_sequence:
                    start_idx = (self.history_len - 1) if self.skip_history_padding else 0
                    sample_indices = np.arange(start_idx, traj_len)
                else:
                    num_samples = max(1, traj_len // (15 * 5))
                    sample_indices = np.random.choice(traj_len, size=num_samples, replace=False)

                for t in sample_indices:
                    # History observation indices (for image / proprio)
                    start_hist_obs = t - self.history_len + 1
                    hist_indices_obs = np.arange(start_hist_obs, t + 1)
                    hist_indices_obs = np.clip(hist_indices_obs, 0, traj_len - 1)

                    # History action indices (shifted by 1)
                    start_hist_act = t - self.history_len
                    hist_indices_act = np.arange(start_hist_act, t)

                    # Future action indices
                    end_fut = t + self.future_len
                    fut_indices = np.arange(t, end_fut)

                    # --- Proprioception ---
                    hist_proprio = torch.from_numpy(proprio_np[hist_indices_obs])

                    # --- History actions ---
                    hist_actions = np.zeros((self.history_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask = hist_indices_act >= 0
                    if np.any(valid_mask):
                        valid_indices = np.clip(hist_indices_act[valid_mask], 0, traj_len - 1)
                        hist_actions[valid_mask] = actions_np[valid_indices]
                    hist_actions = torch.from_numpy(hist_actions)

                    # --- Future actions ---
                    fut_acts_np = np.zeros((self.future_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask_fut = fut_indices < traj_len
                    if np.any(valid_mask_fut):
                        fut_acts_np[valid_mask_fut] = actions_np[fut_indices[valid_mask_fut]]
                    fut_acts = torch.from_numpy(fut_acts_np)

                    # --- Instruction (fixed, matches eval) ---
                    instruction = TASK_INSTRUCTION

                    sample = {
                        "proprioception": hist_proprio,
                        "history_actions": hist_actions,
                        "future_actions": fut_acts,
                        "instruction": instruction,
                    }

                    # Spatial target for auxiliary loss (current timestep)
                    if spatial_targets_np is not None:
                        sample["spatial_target"] = torch.from_numpy(spatial_targets_np[t].copy())
                    else:
                        sample["spatial_target"] = None

                    # Action loss weight
                    sample["action_weight"] = torch.tensor(action_weight_np[t], dtype=torch.float32)

                    # --- Future image (optional) ---
                    if self.load_future_image:
                        if self.future_image_mode == "last":
                            target_idx = traj_len - 1
                        else:
                            target_idx = min(t + self.future_len, traj_len - 1)
                        sample["future_image"] = images_np[target_idx].copy()

                    # --- Visual input ---
                    if self.input_modality == "video":
                        sample["video"] = images_np[hist_indices_obs]
                        if self.view_mode == "multi":
                            sample["video_wrist"] = wrist_np[hist_indices_obs] if wrist_np is not None else images_np[hist_indices_obs]
                            if top_np is not None:
                                sample["video_top"] = top_np[hist_indices_obs]
                    elif self.input_modality == "image":
                        sample["image"] = images_np[t].copy()
                        if self.view_mode == "multi":
                            sample["image_wrist"] = wrist_np[t].copy() if wrist_np is not None else images_np[t].copy()
                            if top_np is not None:
                                sample["image_top"] = top_np[t].copy()
                    else:
                        raise ValueError(f"Unknown input_modality: {self.input_modality}")

                    # --- Shuffle buffer ---
                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= self.buffer_size:
                        idx = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                        yield shuffle_buffer.pop()

                # Cleanup per episode
                del images_np, actions_np, proprio_np, action_weight_np
                if wrist_np is not None:
                    del wrist_np
                if top_np is not None:
                    del top_np
                if spatial_targets_np is not None:
                    del spatial_targets_np
                gc.collect()

        # No flush — loop back and keep filling the buffer
