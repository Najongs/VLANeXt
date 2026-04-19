import os
import gc
import glob
from pathlib import Path
import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import IterableDataset

# Action normalization stats for approach dataset
# Approach has larger movements than align — uses full pipeline stats.
# delta_pose(6) + gripper(1)
# Gripper is always -1.0 (open) during approach.
# action_min_sim_approach = [-1.9950628280639648, -1.9578518867492676, -1.855233907699585, -0.043651919811964035, -0.043956458568573, -0.08516393601894379, -1.0]
# action_max_sim_approach = [1.9631338119506836, 1.9140794277191162, 1.8595972061157227, 0.03279181942343712, 0.03137323260307312, 0.03784353286027908, -1.0]

# 웬만하면 정규화 값은 대칭값으로 학습하자.

action_min_sim_approach = [-1, -1, -1, -0.01, -0.01, -0.01, -1.0]
action_max_sim_approach = [1, 1, 1, 0.01, 0.01, 0.01, 1.0]

TASK_INSTRUCTION = "Approach the needle tip to the small grey circular trocar port on the eye model, next to the larger lens opening"


class SimActApproach(IterableDataset):
    """
    HDF5-based IterableDataset for approach simulation data.
    Same structure as SimActAlign but with approach instruction and normalization.
    Gripper is always open (-1.0) during approach.
    """

    def __init__(
        self,
        data_dir,
        dataset_name="sim_approach",
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
        use_sensor=True,
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
        self.use_sensor = use_sensor

        # Action normalization to [-1, 1]
        self.action_min = np.array(action_min_sim_approach, dtype=np.float32)
        self.action_max = np.array(action_max_sim_approach, dtype=np.float32)

        # Collect all h5 files (recursive — searches subdirectories too)
        # data_dir can be:
        #   - str: single path
        #   - list of str: multiple paths, all episodes used
        #   - list of dict: multiple paths with optional max_episodes per path
        #     e.g. [{"path": "/data/...", "max_episodes": 10000}, ...]
        if isinstance(data_dir, (list, tuple)):
            self.episode_paths = []
            for d in data_dir:
                if isinstance(d, dict):
                    p = d["path"]
                    max_ep = d.get("max_episodes", None)
                else:
                    p = d
                    max_ep = None
                eps = sorted(glob.glob(os.path.join(p, "**", "*.h5"), recursive=True))
                if max_ep is not None and len(eps) > max_ep:
                    rng = np.random.RandomState(42)
                    eps = sorted(rng.choice(eps, size=max_ep, replace=False).tolist())
                self.episode_paths.extend(eps)
            self.episode_paths = sorted(self.episode_paths)
            if not self.episode_paths:
                raise FileNotFoundError(f"No .h5 files found in any of {data_dir}")
        else:
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
        """Load a single episode from HDF5 file and return arrays.
        If phase data exists, only keep phase==1 (approach/align) timesteps.
        """
        with h5py.File(h5_path, "r") as f:
            traj_len = f["action"].shape[0]

            # --- Phase filtering: keep only phase==1 (approach/align) ---
            phase_mask = None
            if "phase" in f:
                phase_all = f["phase"][:].astype(np.int32)
                phase_mask = (phase_all == 1)
                if not np.any(phase_mask):
                    return None  # no approach data in this episode

            # --- Actions (N, 7): normalize delta_pose + gripper to [-1, 1] ---
            actions_np = f["action"][:].astype(np.float32)
            denominator = self.action_max - self.action_min
            denominator = np.where(denominator == 0, 1.0, denominator)
            actions_np = 2.0 * (actions_np - self.action_min) / denominator - 1.0
            actions_np = np.clip(actions_np, -1.0, 1.0)

            # --- Proprioception: ee_pose (N, 7) + optional sensor_dist (N, 1) ---
            proprio_np = f["observations"]["ee_pose"][:].astype(np.float32)  # (N, 7)
            if self.use_sensor and "sensor_dist" in f["observations"]:
                sensor_dist = f["observations"]["sensor_dist"][:].astype(np.float32)
                if sensor_dist.ndim == 1:
                    sensor_dist = sensor_dist[:, None]
                sensor_dist = np.where((sensor_dist < 0) | (sensor_dist > 20.0), 20.0, sensor_dist)
                proprio_np = np.concatenate([proprio_np, sensor_dist], axis=-1)  # (N, 8)

            # --- Spatial auxiliary targets ---
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

            # --- Images ---
            img_grp = f["observations"]["images"]

            images_np = np.stack(
                [self._decode_jpeg(img_grp[self.cam_exterior][i]) for i in range(traj_len)],
                axis=0,
            )

            wrist_np = None
            top_np = None
            if self.view_mode == "multi":
                if self.cam_wrist and self.cam_wrist in img_grp:
                    wrist_np = np.stack(
                        [self._decode_jpeg(img_grp[self.cam_wrist][i]) for i in range(traj_len)],
                        axis=0,
                    )
                if self.cam_top and self.cam_top in img_grp:
                    top_np = np.stack(
                        [self._decode_jpeg(img_grp[self.cam_top][i]) for i in range(traj_len)],
                        axis=0,
                    )

            # --- Apply phase filter (keep phase==1 only) ---
            if phase_mask is not None:
                indices = np.where(phase_mask)[0]
                actions_np = actions_np[indices]
                proprio_np = proprio_np[indices]
                images_np = images_np[indices]
                if wrist_np is not None:
                    wrist_np = wrist_np[indices]
                if top_np is not None:
                    top_np = top_np[indices]
                if spatial_targets_np is not None:
                    spatial_targets_np = spatial_targets_np[indices]
                traj_len = len(indices)

            # --- Action weight: uniform ---
            action_weight_np = np.ones(traj_len, dtype=np.float32)

        return traj_len, actions_np, proprio_np, images_np, wrist_np, top_np, spatial_targets_np, action_weight_np

    def __iter__(self):
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

        episode_paths = [p for i, p in enumerate(self.episode_paths) if i % total_shards == shard_index]
        if self.length is not None:
            episode_paths = episode_paths[: self.length]

        shuffle_buffer = []

        while True:
            np.random.shuffle(episode_paths)

            for ep_path in episode_paths:
                try:
                    result = self._load_episode(ep_path)
                    if result is None:
                        continue
                    traj_len, actions_np, proprio_np, images_np, wrist_np, top_np, spatial_targets_np, action_weight_np = result
                except Exception as e:
                    print(f"[Warn] Skipping {ep_path}: {e}")
                    continue

                if traj_len < self.history_len + 1:
                    del images_np, actions_np, proprio_np
                    if wrist_np is not None:
                        del wrist_np
                    continue

                if self.full_sequence:
                    start_idx = (self.history_len - 1) if self.skip_history_padding else 0
                    sample_indices = np.arange(start_idx, traj_len)
                else:
                    num_samples = max(1, traj_len // (15 * 5))
                    sample_indices = np.random.choice(traj_len, size=num_samples, replace=False)

                for t in sample_indices:
                    start_hist_obs = t - self.history_len + 1
                    hist_indices_obs = np.arange(start_hist_obs, t + 1)
                    hist_indices_obs = np.clip(hist_indices_obs, 0, traj_len - 1)

                    start_hist_act = t - self.history_len
                    hist_indices_act = np.arange(start_hist_act, t)

                    end_fut = t + self.future_len
                    fut_indices = np.arange(t, end_fut)

                    hist_proprio = torch.from_numpy(proprio_np[hist_indices_obs])

                    hist_actions = np.zeros((self.history_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask = hist_indices_act >= 0
                    if np.any(valid_mask):
                        valid_indices = np.clip(hist_indices_act[valid_mask], 0, traj_len - 1)
                        hist_actions[valid_mask] = actions_np[valid_indices]
                    hist_actions = torch.from_numpy(hist_actions)

                    fut_acts_np = np.zeros((self.future_len, actions_np.shape[1]), dtype=np.float32)
                    valid_mask_fut = fut_indices < traj_len
                    if np.any(valid_mask_fut):
                        fut_acts_np[valid_mask_fut] = actions_np[fut_indices[valid_mask_fut]]
                    fut_acts = torch.from_numpy(fut_acts_np)

                    instruction = TASK_INSTRUCTION

                    sample = {
                        "proprioception": hist_proprio,
                        "history_actions": hist_actions,
                        "future_actions": fut_acts,
                        "instruction": instruction,
                        "source_info": f"{Path(ep_path).stem}:t{t}",
                    }

                    if spatial_targets_np is not None:
                        sample["spatial_target"] = torch.from_numpy(spatial_targets_np[t].copy())
                    else:
                        sample["spatial_target"] = None

                    sample["action_weight"] = torch.tensor(action_weight_np[t], dtype=torch.float32)

                    if self.load_future_image:
                        if self.future_image_mode == "last":
                            target_idx = traj_len - 1
                        else:
                            target_idx = min(t + self.future_len, traj_len - 1)
                        sample["future_image"] = images_np[target_idx].copy()

                    if self.input_modality == "video":
                        sample["video"] = images_np[hist_indices_obs]
                        if self.view_mode == "multi":
                            if wrist_np is not None:
                                sample["video_wrist"] = wrist_np[hist_indices_obs]
                            if top_np is not None:
                                sample["video_top"] = top_np[hist_indices_obs]
                    elif self.input_modality == "image":
                        sample["image"] = images_np[t].copy()
                        if self.view_mode == "multi":
                            if wrist_np is not None:
                                sample["image_wrist"] = wrist_np[t].copy()
                            if top_np is not None:
                                sample["image_top"] = top_np[t].copy()
                    else:
                        raise ValueError(f"Unknown input_modality: {self.input_modality}")

                    shuffle_buffer.append(sample)
                    if len(shuffle_buffer) >= self.buffer_size:
                        idx = np.random.randint(len(shuffle_buffer))
                        shuffle_buffer[idx], shuffle_buffer[-1] = shuffle_buffer[-1], shuffle_buffer[idx]
                        yield shuffle_buffer.pop()

                del images_np, actions_np, proprio_np, action_weight_np
                if wrist_np is not None:
                    del wrist_np
                if top_np is not None:
                    del top_np
                if spatial_targets_np is not None:
                    del spatial_targets_np
                gc.collect()
