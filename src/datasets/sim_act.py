import os
import gc
import glob
import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import IterableDataset

# Action normalization stats (computed over 9,984 clean episodes, 4,246,613 steps — outliers removed)
# delta_pose(6) + gripper(1)
action_min_sim = [-0.7714075446128845, -2.5631182193756104, -0.5680814385414124,
                  -0.024533260613679886, -0.07352259010076523, -0.051400259137153625, -1.0]
action_max_sim = [1.5511466264724731, 1.4748575687408447, 0.1918737143278122,
                  0.004710988607257605, 0.0015634826850146055, 0.021533237770199776, 1.0]


class SimAct(IterableDataset):
    """
    HDF5-based IterableDataset for custom simulation data.
    Mirrors the DroidAct interface so it can be used as a drop-in replacement
    in train.py.

    Expected HDF5 structure (per episode file):
        action:              (N, 7)  — delta_pose(6) + gripper(1)
        language_instruction: str
        phase:               (N,)
        timestamp:           (N,)
        observations/
            ee_pose:         (N, 7)  — position(3) + orientation(3) + gripper(1)
            qpos:            (N, 6)
            sensor_dist:     (N,)
            images/
                side_camera:  JPEG vlen  (exterior view)
                tool_camera:  JPEG vlen  (wrist view)
                top_camera:   JPEG vlen  (unused by default)

    Camera mapping (DROID-compatible):
        side_camera  → exterior (primary view)
        tool_camera  → wrist    (secondary view)
    """

    def __init__(
        self,
        data_dir,
        dataset_name="sim",
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
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.length = length
        self.history_len = history_len
        self.future_len = future_len
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
        self.action_min = np.array(action_min_sim, dtype=np.float32)
        self.action_max = np.array(action_max_sim, dtype=np.float32)

        # Collect all h5 files
        self.episode_paths = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        if not self.episode_paths:
            raise FileNotFoundError(f"No .h5 files found in {data_dir}")

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

            # --- Proprioception: ee_pose (N, 7) ---
            proprio_np = f["observations"]["ee_pose"][:].astype(np.float32)  # (N, 7)

            # --- Sensor distance (for instruction augmentation & action weight) ---
            sensor_dist_np = None
            proximity_np = None
            if "sensor_dist" in f["observations"]:
                sensor_dist_np = f["observations"]["sensor_dist"][:].astype(np.float32)  # (N,)
                proximity_np = ((sensor_dist_np >= 0) & (sensor_dist_np < 10.0))  # (N,) bool

            # --- Language instruction (base) ---
            base_instruction = "Align the needle with the hollow cylindrical opening and insert it"

            # --- Spatial auxiliary targets (backward compatible) ---
            # Layout (8D): kp_wrist(4) + visibility(2) + dist(1) + phase(1)
            spatial_targets_np = None
            if "keypoints_wrist" in f["observations"]:
                needle_tip = f["observations"]["needle_tip_pos"][:].astype(np.float32)     # (N, 3)
                trocar_entry = f["observations"]["trocar_entry_pos"][:].astype(np.float32) # (N, 3)
                kp_wrist = f["observations"]["keypoints_wrist"][:].astype(np.float32)      # (N, 4)
                kp_vis = f["observations"]["keypoints_visibility"][:].astype(np.float32)   # (N, 2)
                phase_raw = f["phase"][:].astype(np.float32)                                # (N,)

                dist = np.linalg.norm(trocar_entry - needle_tip, axis=-1, keepdims=True)   # (N, 1) mm
                dist_normalized = dist / 100.0  # roughly [0, 1]
                phase_binary = np.clip(phase_raw - 1, 0, 1).reshape(-1, 1)  # 1→0, 2→1

                spatial_targets_np = np.concatenate(
                    [kp_wrist, kp_vis, dist_normalized, phase_binary], axis=-1
                )  # (N, 8)

            # --- Action weight: critical zone = phase 1 (align) + sensor within 30mm ---
            action_weight_np = np.ones(traj_len, dtype=np.float32)
            if sensor_dist_np is not None:
                phase_raw = f["phase"][:].astype(np.float32)
                critical = (phase_raw == 1) & (sensor_dist_np >= 0) & (sensor_dist_np < 30.0)
                action_weight_np[critical] = 5.0  # 5x weight in critical zone

            # --- Images: decode all frames for selected cameras ---
            img_grp = f["observations"]["images"]

            images_np = np.stack(
                [self._decode_jpeg(img_grp[self.cam_exterior][i]) for i in range(traj_len)],
                axis=0,
            )  # (N, H, W, 3) uint8

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

        return traj_len, actions_np, proprio_np, base_instruction, images_np, wrist_np, top_np, spatial_targets_np, action_weight_np, proximity_np

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

        # Shuffle episode order
        np.random.shuffle(episode_paths)

        shuffle_buffer = []

        for ep_path in episode_paths:
            try:
                traj_len, actions_np, proprio_np, base_instruction, images_np, wrist_np, top_np, spatial_targets_np, action_weight_np, proximity_np = self._load_episode(ep_path)
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
                sample_indices = np.arange(traj_len)
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

                # --- Per-timestep instruction with proximity context ---
                if proximity_np is not None and proximity_np[t]:
                    instruction = base_instruction + ". Object detected nearby"
                else:
                    instruction = base_instruction

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

                # Action loss weight (higher in critical zone)
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

        # Flush remaining buffer
        np.random.shuffle(shuffle_buffer)
        for sample in shuffle_buffer:
            yield sample
