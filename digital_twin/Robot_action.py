#!/usr/bin/env python
"""
Real-time VLA Data Collection with 6-DoF Gamepad & Multi-Camera View
- Control: Xbox/PS Controller (Full 6-Axis + D-Pad Support)
- Logging: HDF5 (All Cameras)
- Features: Auto-Recovery, Multi-View, HOME Button, D-Pad Control
"""
import os
import sys
import time
import threading
import logging
import pathlib
import h5py
import numpy as np
from datetime import datetime
from termcolor import colored
import contextlib
import cv2
import queue
import struct
from collections import deque

import pygame
import depthai as dai
import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
# from lerobot.utils.utils import init_logging

# --- Configuration ---
ROBOT_ADDRESS = "192.168.0.100"
DATASET_DIR = "collected_data"

CONTROL_FREQUENCY = 15
# 속도 설정
SCALE_POS = 0.8   # 스틱 이동 속도 (빠름)
SCALE_HAT = 0.3   # 방향키 이동 속도 (정밀 조작용)
SCALE_Z   = 0.5   # 삽입 속도
SCALE_ROT = 0.3   # 회전 속도
DEADZONE = 0.2    # 노이즈 제거

# 초기 위치 (Safe Start Pose)
HOME_JOINTS = (30, -20, 20, 0, 30, 60)

# init_logging()
logger = logging.getLogger(__name__)

# ============================================================
# 1️⃣ Global Clock
# ============================================================
class GlobalClock(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.timestamp = round(time.time(), 3)
        self.running = True
        self.lock = threading.Lock()
    def now(self):
        with self.lock: return self.timestamp
    def run(self):
        while self.running:
            with self.lock: self.timestamp = round(time.time(), 3)
            time.sleep(0.005)
    def stop(self): self.running = False

# ============================================================
# 2️⃣ RtSampler (Real-time Robot State Sampler)
# ============================================================
class RtSampler(threading.Thread):
    def __init__(self, robot, clock, rate_hz=100):
        super().__init__(daemon=True)
        self.robot = robot
        self.dt = 1.0 / float(rate_hz)
        self.clock = clock
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.latest_q = np.zeros(6)
        self.latest_p = np.zeros(6)
    def stop(self): self.stop_evt.set()
    def get_latest_data(self):
        with self.lock: return self.latest_q.copy(), self.latest_p.copy()
    def run(self):
        logger.info("🤖 Starting robot state sampler...")

        # 초기 로봇 상태 읽기
        initial_success = False
        for _ in range(10):  # 최대 10번 시도
            try:
                q = list(self.robot.GetJoints())
                p = list(self.robot.GetPose())
                if q and len(q) >= 6 and p and len(p) >= 6:
                    with self.lock:
                        self.latest_q = np.array(q[:6])
                        self.latest_p = np.array(p[:6])
                    logger.info(f"✅ Initial robot state acquired: q={self.latest_q.round(2)}")
                    initial_success = True
                    break
            except Exception as e:
                logger.debug(f"Initial read attempt failed: {e}")
            time.sleep(0.1)

        if not initial_success:
            logger.warning("⚠️ Failed to get initial robot state! Data will be all zeros until first successful read.")

        next_t = time.time()
        success_count = 0
        fail_count = 0

        while not self.stop_evt.is_set():
            ts_now = self.clock.now()
            q, p = None, None

            # 1. 관절 각도 (Joint Angles) 가져오기
            try:
                q = list(self.robot.GetJoints())
            except Exception as e:
                if fail_count == 0:  # 첫 실패만 로그
                    logger.debug(f"GetJoints failed: {e}")

            # 2. 말단 자세 (Cartesian Pose) 가져오기
            try:
                p = list(self.robot.GetPose())
            except Exception as e:
                if fail_count == 0:  # 첫 실패만 로그
                    logger.debug(f"GetPose failed: {e}")

            # 데이터가 유효하면 메모리에 업데이트
            if q and len(q) >= 6 and p and len(p) >= 6:
                with self.lock:
                    self.latest_q = np.array(q[:6])
                    self.latest_p = np.array(p[:6])
                success_count += 1
            else:
                fail_count += 1
                if fail_count % 100 == 1:  # 100번마다 한번씩만 경고
                    logger.warning(f"⚠️ Invalid robot data (fail count: {fail_count})")

            next_t += self.dt
            if next_t - time.time() > 0: time.sleep(next_t - time.time())

        logger.info(f"📊 RtSampler stats: {success_count} success, {fail_count} failures")

# ============================================================
# OCT & FPI sensor Data
# ============================================================

# 🔥 OPTIMIZED: Sensor settings (100ms window)
SENSOR_ENABLED = True
SENSOR_TEMPORAL_LENGTH = 70  # 100ms at 700Hz
SENSOR_INPUT_CHANNELS = 1026  # 1 force + 1025 A-scan

# Network settings
SENSOR_UDP_PORT = 9999
SENSOR_UDP_IP = "0.0.0.0"
SENSOR_BUFFER_SIZE = 4 * 1024 * 1024

# Sensor packet format
SENSOR_NXZRt = 1025
SENSOR_PACKET_HEADER_FORMAT = '<ddf'  # ts, send_ts, force
SENSOR_PACKET_HEADER_SIZE = struct.calcsize(SENSOR_PACKET_HEADER_FORMAT)
SENSOR_ALINE_FORMAT = f'<{SENSOR_NXZRt}f'
SENSOR_ALINE_SIZE = struct.calcsize(SENSOR_ALINE_FORMAT)
SENSOR_TOTAL_PACKET_SIZE = SENSOR_PACKET_HEADER_SIZE + SENSOR_ALINE_SIZE
SENSOR_CALIBRATION_COUNT = 50

class OCT_FPI_sampler:
    def __init__(self, max_length=70, channels=1026, save_buffer=None):
        self.max_length = max_length
        self.channels = channels
        self.buffer = deque(maxlen=max_length)
        self.save_buffer = save_buffer  # Optional: list to save all data
        self.lock = threading.Lock()
        self.inference_started = False  # 녹화 시작 여부
        self.latest_force = 0.0  # 최신 force 값

    def add_samples(self, samples: list):
        """Add multiple samples (from UDP batch)"""
        with self.lock:
            for sample in samples:
                force = np.array([sample['force']], dtype=np.float32)
                aline = sample['aline'].astype(np.float32)
                combined = np.concatenate([force, aline])  # (1026,)
                self.buffer.append(combined)

                # Update latest force value
                self.latest_force = sample['force']

                # Save to permanent buffer only after inference has started
                if self.save_buffer is not None and self.inference_started:
                    self.save_buffer.append({
                        'timestamp': sample['timestamp'],
                        'send_timestamp': sample['send_timestamp'],
                        'force': sample['force'],
                        'aline': sample['aline']
                    })

    def start_recording(self):
        """녹화 시작 - 이후 수신되는 데이터는 save_buffer에 저장됨"""
        with self.lock:
            self.inference_started = True
            if self.save_buffer is not None:
                self.save_buffer.clear()
        logger.info("🔴 Sensor recording started")

    def stop_recording(self):
        """녹화 종료"""
        with self.lock:
            self.inference_started = False
        logger.info("⏹️ Sensor recording stopped")

    def get_tensor(self):
        """Get current buffer as numpy array (T, C) with padding if needed"""
        with self.lock:
            if len(self.buffer) == 0:
                # Return zeros if no data yet
                return np.zeros((self.max_length, self.channels), dtype=np.float32)

            data = np.array(list(self.buffer), dtype=np.float32)  # (current_len, C)

            # Pad to max_length if needed
            if len(data) < self.max_length:
                pad_length = self.max_length - len(data)
                padding = np.zeros((pad_length, self.channels), dtype=np.float32)
                data = np.concatenate([padding, data], axis=0)

            return data  # (70, 1026)

    def get_status(self):
        """Get sensor status information for display"""
        with self.lock:
            return {
                'buffer_size': len(self.buffer),
                'max_length': self.max_length,
                'latest_force': self.latest_force,
                'recording': self.inference_started
            }

class SensorUDPReceiver(threading.Thread):
    """
    Receives INDIVIDUAL sensor packets (4120 bytes) via UDP.
    Compatible with C++ Sender (Batch Size = 1)
    """
    def __init__(self, sensor_buffer, stop_event):
        super().__init__(daemon=True)
        self.sensor_buffer = sensor_buffer
        self.stop_event = stop_event
        self.clock_offset = None
        self.calibration_samples = []
        self.packet_count = 0
        self.packets_per_second = 0.0
        self.last_rate_update = time.time()
        self.packets_since_last_update = 0
        self.lock = threading.Lock()
        self.last_packet_time = time.time()  # 마지막 패킷 수신 시간

        # C++ 구조체 설정 (DataPacket)
        # 4120 bytes = 8(double) + 8(double) + 4(float) + 4*1025(float array)
        self.PACKET_SIZE = 4120
        self.PACKET_FORMAT = '<ddf1025f' 

    def run(self):
        import socket
        import struct

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            sock.bind((SENSOR_UDP_IP, SENSOR_UDP_PORT))
            sock.settimeout(1.0)
            logger.info(f"✅ Sensor UDP Receiver started on port {SENSOR_UDP_PORT}")
        except Exception as e:
            logger.error(f"❌ Failed to bind UDP socket: {e}")
            return

        while not self.stop_event.is_set():
            try:
                # 1. 패킷 크기를 넉넉하게 잡고 수신
                data, addr = sock.recvfrom(4200) 
            except socket.timeout:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.debug(f"[UDP Sensor] Receive error: {e}")
                continue

            # 2. 크기 검증: 정확히 4120 바이트여야 함
            if len(data) != self.PACKET_SIZE:
                continue

            try:
                # 3. 구조체 언패킹 (헤더 개수 확인 로직 삭제됨)
                unpacked = struct.unpack(self.PACKET_FORMAT, data)
                
                ts = unpacked[0]
                send_ts = unpacked[1]
                force = unpacked[2]
                aline = np.array(unpacked[3:], dtype=np.float32)

                record = {
                    'timestamp': ts,
                    'send_timestamp': send_ts,
                    'force': force,
                    'aline': aline
                }

                # 4. 버퍼에 추가
                self.sensor_buffer.add_samples([record])

                # 통계 업데이트
                current_time = time.time()
                with self.lock:
                    self.last_packet_time = current_time
                self.packet_count += 1
                self.packets_since_last_update += 1

                # 1초마다 수신율 계산
                if current_time - self.last_rate_update >= 1.0:
                    with self.lock:
                        self.packets_per_second = self.packets_since_last_update / (current_time - self.last_rate_update)
                    self.packets_since_last_update = 0
                    self.last_rate_update = current_time

                # Clock calibration (처음 50개만)
                if self.clock_offset is None:
                    recv_time = time.time()
                    self.calibration_samples.append(recv_time - send_ts)
                    if len(self.calibration_samples) >= 50:
                        self.clock_offset = np.mean(self.calibration_samples)
                        logger.info(f"\n✅ Sensor Clock Offset Calibrated: {self.clock_offset * 1000:.1f} ms\n")

            except Exception as e:
                logger.error(f"Unpack failed: {e}")
                continue

        sock.close()
        logger.info("🛑 Sensor UDP Receiver stopped")

    def get_stats(self):
        """Get receiver statistics"""
        with self.lock:
            time_since_last = time.time() - self.last_packet_time
            return {
                'total_packets': self.packet_count,
                'packets_per_second': self.packets_per_second,
                'calibrated': self.clock_offset is not None,
                'is_receiving': time_since_last < 2.0  # 2초 이내에 데이터 받았는지
            }

# ============================================================
# 2.5️⃣ Sensor Visualization
# ============================================================
def visualize_sensor_data(sensor_sampler):
    """
    Create visualization window for sensor data
    - OCT M-mode image (top)
    - FPI force graph (bottom)
    """
    if sensor_sampler is None:
        return None

    with sensor_sampler.lock:
        if len(sensor_sampler.buffer) == 0:
            return None

        # Extract data from circular buffer
        buffer_data = list(sensor_sampler.buffer)

        # Extract force and aline data
        forces = np.array([sample[0] for sample in buffer_data])  # First channel is force
        alines = np.array([sample[1:] for sample in buffer_data])  # Remaining 1025 channels

    # === 1. OCT M-mode Image ===
    # Transpose to get (Depth, Time) for image display
    mmode_img = alines.T  # (1025, T) where T <= 70

    # Normalize to 0-255 for display
    if mmode_img.max() > mmode_img.min():
        mmode_normalized = ((mmode_img - mmode_img.min()) / (mmode_img.max() - mmode_img.min()) * 255).astype(np.uint8)
    else:
        mmode_normalized = np.zeros_like(mmode_img, dtype=np.uint8)

    # Convert to BGR for consistency with OpenCV (grayscale)
    mmode_gray = cv2.cvtColor(mmode_normalized, cv2.COLOR_GRAY2BGR)

    # Resize for better visibility (width x4 to see time progression, height 400)
    mmode_display = cv2.resize(mmode_gray, (mmode_gray.shape[1] * 4, 400))

    # Add label
    cv2.putText(mmode_display, "OCT M-mode (Grayscale)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(mmode_display, f"Depth: {alines.shape[1]} | Frames: {len(buffer_data)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # === 2. FPI Force Graph ===
    graph_width = mmode_display.shape[1]
    graph_height = 200
    graph_img = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

    # Normalize force values for display
    if len(forces) > 1:
        force_min, force_max = forces.min(), forces.max()
        if force_max > force_min:
            forces_norm = (forces - force_min) / (force_max - force_min)
        else:
            forces_norm = np.zeros_like(forces)

        # Draw force graph
        x_step = graph_width / len(forces)
        points = []
        for i, f in enumerate(forces_norm):
            x = int(i * x_step)
            y = int(graph_height - 50 - f * (graph_height - 100))  # Leave margin
            points.append((x, y))

        # Draw line
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(graph_img, points[i], points[i+1], (0, 255, 0), 2)

    # Add label and value
    cv2.putText(graph_img, "FPI Force", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if len(forces) > 0:
        cv2.putText(graph_img, f"Current: {forces[-1]:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(graph_img, f"Min: {forces.min():.3f}  Max: {forces.max():.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Draw axes
    cv2.line(graph_img, (50, graph_height - 50), (graph_width - 20, graph_height - 50), (100, 100, 100), 1)  # X-axis
    cv2.line(graph_img, (50, 50), (50, graph_height - 50), (100, 100, 100), 1)  # Y-axis

    # === 3. Combine vertically ===
    combined = np.vstack([mmode_display, graph_img])

    return combined

# ============================================================
# 3️⃣ Gamepad Controller (Added D-Pad Support & Multi-Mode)
# ============================================================
class GamepadController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.control_mode = 1  # 컨트롤 모드: 1, 2, 3
        self.mode_switch_cooldown = 0  # 모드 전환 쿨다운
        self.smoothing_enabled = False  # 가속도 모드 (X 버튼으로 토글)
        self.smoothing_cooldown = 0  # 토글 쿨다운
        self.current_action = np.zeros(6)  # 현재 속도 (가속도 모드용)
        self.acceleration_rate = 0.2   # 가속률 (빠른 가속)
        self.deceleration_rate = 0.7   # 감속률 (빠른 멈춤)
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logger.info(f"🎮 Connected: {self.joystick.get_name()}")
            logger.info(f"🔧 Control Mode: {self.control_mode} (Press BACK/SELECT to change)")
        else:
            logger.error("❌ No Gamepad found!")

    def get_action(self):
        if not self.joystick: return np.zeros(6), False, False, False
        pygame.event.pump()

        # --- Mode Switching (BACK/SELECT button = button 6) ---
        btn_mode_switch = self.joystick.get_button(6) if self.joystick.get_numbuttons() > 6 else False
        if btn_mode_switch and self.mode_switch_cooldown == 0:
            self.control_mode = (self.control_mode % 3) + 1
            logger.info(colored(f"🔧 Switched to Control Mode {self.control_mode}", "yellow"))
            self.mode_switch_cooldown = 10  # 10 프레임 쿨다운
        if self.mode_switch_cooldown > 0:
            self.mode_switch_cooldown -= 1

        # --- Smoothing Toggle (X button = button 2) ---
        btn_smoothing = self.joystick.get_button(2) if self.joystick.get_numbuttons() > 2 else False
        if btn_smoothing and self.smoothing_cooldown == 0:
            self.smoothing_enabled = not self.smoothing_enabled
            status = "ON" if self.smoothing_enabled else "OFF"
            logger.info(colored(f"🌊 Smoothing Mode: {status}", "cyan"))
            self.smoothing_cooldown = 10  # 10 프레임 쿨다운
        if self.smoothing_cooldown > 0:
            self.smoothing_cooldown -= 1

        # --- 1. Analog Stick Inputs (Raw) ---
        # Left Stick: Move
        y_stick_raw = self.joystick.get_axis(1)
        x_stick_raw = -self.joystick.get_axis(0)

        # Right Stick
        rs_x_raw = self.joystick.get_axis(3)  # 좌우
        rs_y_raw = -self.joystick.get_axis(4) # 상하

        # Apply deadzone
        y_stick = y_stick_raw if abs(y_stick_raw) > DEADZONE else 0.0
        x_stick = x_stick_raw if abs(x_stick_raw) > DEADZONE else 0.0
        rs_x = rs_x_raw if abs(rs_x_raw) > DEADZONE else 0.0
        rs_y = rs_y_raw if abs(rs_y_raw) > DEADZONE else 0.0

        # Apply scaling to movement
        y_stick *= SCALE_POS
        x_stick *= SCALE_POS

        # Triggers: Z-Axis (or rotation depending on mode)
        lt = (self.joystick.get_axis(2) + 1) / 2
        rt = (self.joystick.get_axis(5) + 1) / 2

        # Bumpers
        lb = self.joystick.get_button(4)
        rb = self.joystick.get_button(5)

        # --- 2. D-Pad (Hat) Inputs ---
        # D-Pad는 왼쪽 스틱과 동일한 방향으로 매핑 (스케일만 다름)
        hat_x, hat_y = self.joystick.get_hat(0)
        y_hat = -hat_y * SCALE_HAT   # D-pad UP/DOWN → Y축 (상하, 스틱과 동일)
        x_hat = -hat_x * SCALE_HAT   # D-pad LEFT/RIGHT → X축 (좌우, 스틱과 동일)

        # --- 3. Combine Movement ---
        y = y_stick + y_hat
        x = x_stick + x_hat

        # --- 3.5. Tool Frame Compensation (60° rotation) ---
        # HOME 자세의 J6=60도 회전을 보정하기 위해 -60도 역회전 적용
        angle = np.radians(60)
        x_rotated = x * np.cos(angle) - y * np.sin(angle)
        y_rotated = x * np.sin(angle) + y * np.cos(angle)

        # 보정된 값으로 대체
        x = x_rotated
        y = y_rotated

        # --- 4. Rotation Mapping (Mode-dependent) ---
        if self.control_mode == 1:
            # Mode 1: 기본 (Original)
            # RS: Pitch/Roll, Bumper: Yaw, Trigger: Z
            rx = rs_y * SCALE_ROT  # Pitch (상하)
            ry = rs_x * SCALE_ROT  # Roll (좌우)
            rz = (rb - lb) * SCALE_ROT * 1.5  # Yaw
            z = (rt - lt) * SCALE_Z

        elif self.control_mode == 2:
            # Mode 2: 바늘 삽입 최적화
            # RS: Pitch/Yaw, Bumper: Roll, Trigger: Z
            rx = rs_y * SCALE_ROT  # Pitch (상하)
            rz = rs_x * SCALE_ROT * 1.5  # Yaw (좌우) - 더 민감하게
            ry = (rb - lb) * SCALE_ROT  # Roll
            z = (rt - lt) * SCALE_Z

        else:  # self.control_mode == 3
            # Mode 3: 트리거 회전
            # RS 상하: Pitch only, Trigger: Yaw, Bumper: Roll
            rx = rs_y * SCALE_ROT  # Pitch (상하)
            ry = (rb - lb) * SCALE_ROT  # Roll
            rz = (rt - lt) * SCALE_ROT * 2.0  # Yaw (트리거)
            z = 0  # Z축은 다른 방법으로 제어해야 함 (이 모드의 단점)

        # --- 4.5. Rotation Compensation (60° rotation) ---
        # HOME 자세의 J6=60도 회전을 보정하기 위해 rx, ry도 60도 회전 적용
        # (rz는 Z축 기준이므로 보정 불필요)
        angle = np.radians(60)
        rx_rotated = rx * np.cos(angle) - ry * np.sin(angle)
        ry_rotated = rx * np.sin(angle) + ry * np.cos(angle)
        rx = rx_rotated
        ry = ry_rotated

        target_action = np.array([y, x, z, rx, ry, rz])

        # 가속도 모드가 활성화되어 있으면 smoothing 적용
        if self.smoothing_enabled:
            if np.linalg.norm(target_action) < 0.01:
                # 조이스틱을 거의 안 움직이면 → 빠르게 감속
                self.current_action += (target_action - self.current_action) * self.deceleration_rate
            else:
                # 조이스틱을 움직이면 → 천천히 가속
                self.current_action += (target_action - self.current_action) * self.acceleration_rate
            action = np.where(np.abs(self.current_action) < 0.001, 0, self.current_action)
        else:
            # 즉시 반응 모드
            action = target_action
            self.current_action = target_action  # 모드 전환 시를 위해 동기화

        # Buttons
        btn_rec = self.joystick.get_button(0) # A
        btn_disc = self.joystick.get_button(1) # B
        btn_home = self.joystick.get_button(3) # Y
        btn_exit = self.joystick.get_button(7) if self.joystick.get_numbuttons() > 7 else False # START

        return action, btn_rec, btn_disc, btn_home, btn_exit

# ============================================================
# 4️⃣ Camera & Recorder
# ============================================================
class OAKCameraManager:
    def __init__(self, width=640, height=480, fps=30):
        self.width, self.height = width, height
        self.stack = contextlib.ExitStack()
        self.queues = []
    def initialize_cameras(self):
        infos = dai.Device.getAllAvailableDevices()
        if not infos: raise RuntimeError("No OAK devices")
        for info in infos:
            p = dai.Pipeline()
            c = p.create(dai.node.ColorCamera)
            c.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            c.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            c.setPreviewSize(self.width, self.height)
            c.setInterleaved(False)

            # 카메라 ID가 19로 시작하면 수동 초점 설정
            camera_id = info.getMxId()
            if camera_id.startswith("19"):
                c.initialControl.setManualFocus(101)
                logger.info(f"📷 Camera {camera_id}: Manual focus set to 101")
            else:
                logger.info(f"📷 Camera {camera_id}: Auto focus")

            out = p.create(dai.node.XLinkOut)
            out.setStreamName("rgb")
            c.preview.link(out.input)
            d = self.stack.enter_context(dai.Device(p, info, dai.UsbSpeed.SUPER))
            self.queues.append(d.getOutputQueue("rgb", 4, False))
        return len(self.queues)
    def get_frames(self):
        frames = {}
        for i, q in enumerate(self.queues):
            f = q.tryGet()
            if f: frames[f"camera{i+1}"] = f.getCvFrame()
        return frames
    def close(self): self.stack.close()

class VLARecorder:
    def __init__(self, output_dir, clock, sensor_buffer=None):
        self.out = pathlib.Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.buffer = []
        self.sensor_buffer = sensor_buffer  # OCT_FPI_sampler 인스턴스
        self.recording = False
        self.is_saving = False # 현재 저장 중인지 확인하는 플래그

    def start(self):
        if self.is_saving:
            logger.warning("⚠️ Still saving previous episode! Wait a moment.")
            return
        self.buffer = []
        self.recording = True

        # 센서 녹화 시작
        if self.sensor_buffer is not None:
            self.sensor_buffer.start_recording()

        logger.info("🎥 STARTED Recording")

    def add(self, frames, q, p, action):
        if not self.recording: return
        # 메모리 절약을 위해 여기서 변환하지 않고 저장할 때 변환할 수도 있지만,
        # CPU 부하 분산을 위해 수집 시 변환 유지 (단, 메모리 넉넉한 경우)
        rgb = {k: cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for k, v in frames.items()}
        self.buffer.append({"ts": self.clock.now(), "imgs": rgb, "q": q, "p": p, "act": action})

    def save_async(self):
        """백그라운드 스레드에서 저장을 수행 (메인 루프 멈춤 방지)"""
        if not self.buffer: return

        # 1. 현재 버퍼를 임시 변수에 넘기고, 메인 버퍼는 즉시 비움 (다음 녹화 준비)
        buffer_snapshot = self.buffer
        self.buffer = []
        self.recording = False

        # 센서 녹화 종료 및 데이터 복사
        sensor_data_snapshot = None
        if self.sensor_buffer is not None:
            self.sensor_buffer.stop_recording()
            with self.sensor_buffer.lock:
                if self.sensor_buffer.save_buffer:
                    sensor_data_snapshot = list(self.sensor_buffer.save_buffer)

        self.is_saving = True # 저장 시작 플래그 ON

        # 2. 저장 작업 정의 (별도 스레드에서 실행될 함수)
        def worker(data, sensor_data, filename):
            try:
                start_time = time.time()
                logger.info(f"💾 Saving {len(data)} steps to disk... (Background)")

                with h5py.File(filename, 'w') as f:
                    obs = f.create_group("observations")
                    img_grp = obs.create_group("images")

                    # 첫 번째 프레임으로 키 확인
                    first = data[0]["imgs"]

                    # ✨ 이미지 저장 (JPEG 압축 - 5~10배 용량 감소)
                    # Quality 95: 거의 무손실 수준, VLA 학습에 최적
                    jpeg_quality = 95
                    for k in first.keys():
                        jpeg_list = []
                        encode_failures = 0
                        for idx, x in enumerate(data):
                            # 이미지 유효성 검사
                            img = x["imgs"][k]
                            if img is None or img.size == 0:
                                logger.warning(f"Empty image at step {idx} for {k}")
                                # 검은 이미지로 대체 (640x480 기본)
                                img = np.zeros((480, 640, 3), dtype=np.uint8)

                            # JPEG 인코딩 (BGR로 변환 필요 - imencode는 BGR 기대)
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            success, jpeg_buf = cv2.imencode('.jpg', img_bgr,
                                                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                            if success and jpeg_buf is not None and len(jpeg_buf) > 0:
                                # numpy array를 1D로 flatten (vlen 저장 위해)
                                jpeg_list.append(jpeg_buf.flatten())
                            else:
                                encode_failures += 1
                                logger.warning(f"JPEG encoding failed at step {idx} for {k}")
                                # 최소 크기 JPEG 생성 (검은 이미지)
                                black_img = np.zeros((480, 640, 3), dtype=np.uint8)
                                _, fallback_buf = cv2.imencode('.jpg', black_img,
                                                              [cv2.IMWRITE_JPEG_QUALITY, 50])
                                jpeg_list.append(fallback_buf.flatten())

                        if encode_failures > 0:
                            logger.warning(f"⚠️ {k}: {encode_failures}/{len(data)} frames failed encoding")

                        # 가변 길이 바이너리 데이터 저장 (HDF5 vlen dtype)
                        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                        dset = img_grp.create_dataset(k, (len(jpeg_list),), dtype=dt)
                        for i, jpeg_data in enumerate(jpeg_list):
                            dset[i] = jpeg_data

                    # ✨ 나머지 데이터 저장 (float32로 변환 - 50% 용량 감소)
                    # shuffle=True: 압축률 향상 (비슷한 값들을 그룹화)
                    obs.create_dataset("qpos",
                                      data=np.stack([x["q"] for x in data]).astype(np.float32),
                                      compression="gzip", compression_opts=4, shuffle=True)
                    obs.create_dataset("ee_pose",
                                      data=np.stack([x["p"] for x in data]).astype(np.float32),
                                      compression="gzip", compression_opts=4, shuffle=True)
                    f.create_dataset("action",
                                    data=np.stack([x["act"] for x in data]).astype(np.float32),
                                    compression="gzip", compression_opts=4, shuffle=True)
                    f.create_dataset("timestamp",
                                    data=np.stack([x["ts"] for x in data]).astype(np.float32),
                                    compression="gzip", compression_opts=4, shuffle=True)

                    # ✨ 센서 데이터 저장 (OCT + FPI) - float16으로 50% 용량 절감
                    if sensor_data is not None and len(sensor_data) > 0:
                        sensor_grp = obs.create_group("sensor")
                        # Force 데이터 (1D array) - float16
                        forces = np.array([s['force'] for s in sensor_data], dtype=np.float16)
                        sensor_grp.create_dataset("force",
                                                 data=forces,
                                                 compression="gzip", compression_opts=4)

                        # A-line 데이터 (2D array: num_samples x 1025) - float16
                        # M-mode 이미지 생성용이므로 시간/채널 해상도 유지, 타입만 최적화
                        alines = np.stack([s['aline'] for s in sensor_data]).astype(np.float16)
                        sensor_grp.create_dataset("aline",
                                                 data=alines,
                                                 compression="gzip", compression_opts=4, shuffle=True)

                        # 센서 타임스탬프 (float32 유지 - 타임스탬프 정밀도 중요)
                        sensor_ts = np.array([s['timestamp'] for s in sensor_data], dtype=np.float32)
                        sensor_grp.create_dataset("timestamp",
                                                 data=sensor_ts,
                                                 compression="gzip", compression_opts=4)

                        logger.info(f"📊 Saved {len(sensor_data)} sensor samples (float16, ~50% compressed)")

                duration = time.time() - start_time
                file_size_mb = filename.stat().st_size / (1024 * 1024)
                logger.info(colored(f"✅ Save Complete: {filename} ({duration:.1f}s, {file_size_mb:.1f} MB)", "green"))

            except Exception as e:
                logger.error(f"❌ Save Failed: {e}")

            finally:
                self.is_saving = False # 저장 완료 플래그 OFF

        # 3. 파일명 생성 및 스레드 시작
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.out / f"episode_{timestamp}.h5"

        t = threading.Thread(target=worker, args=(buffer_snapshot, sensor_data_snapshot, fname))
        t.start()

        logger.info("⏳ Saving started in background... You can move the robot.")

    def discard(self):
        self.buffer = []
        self.recording = False

        # 센서 녹화도 종료
        if self.sensor_buffer is not None:
            self.sensor_buffer.stop_recording()

        logger.warning("🗑️ DISCARDED")

# ============================================================
# 5️⃣ Main
# ============================================================
def main():
    clock = GlobalClock()
    clock.start()
    gp = GamepadController()
    if not gp.joystick: return
    cam = OAKCameraManager()

    # 센서 시스템 초기화 (옵션)
    sensor_sampler = None
    sensor_receiver = None
    sensor_stop_event = None

    if SENSOR_ENABLED:
        logger.info("🔬 Initializing sensor system...")
        sensor_save_buffer = []  # 녹화된 센서 데이터 저장용
        sensor_sampler = OCT_FPI_sampler(
            max_length=SENSOR_TEMPORAL_LENGTH,
            channels=SENSOR_INPUT_CHANNELS,
            save_buffer=sensor_save_buffer
        )
        sensor_stop_event = threading.Event()
        sensor_receiver = SensorUDPReceiver(sensor_sampler, sensor_stop_event)
        sensor_receiver.start()
        logger.info("✅ Sensor system initialized")

    rec = VLARecorder(DATASET_DIR, clock, sensor_buffer=sensor_sampler)

    try:
        robot = mdr.Robot()
        logger.info(f"🔌 Connecting to robot at {ROBOT_ADDRESS}...")
        robot.Connect(address=ROBOT_ADDRESS)

        if not robot.IsConnected():
            logger.error(f"❌ Failed to connect to robot at {ROBOT_ADDRESS}")
            return

        logger.info("✅ Connected! Activating and homing...")
        robot.ActivateAndHome()
        robot.SetRealTimeMonitoring(1)

        logger.info("🏠 Moving to start pose...")
        robot.MoveJoints(*HOME_JOINTS)
        robot.WaitIdle()
        logger.info("✅ Ready!")

        sampler = RtSampler(robot, clock, 100)
        sampler.start()

        logger.info("📷 Initializing cameras...")
        num_cameras = cam.initialize_cameras()
        logger.info(f"✅ {num_cameras} camera(s) initialized")

        logger.info(colored("\n=== CONTROLS ===", "cyan"))
        logger.info(" [LS / D-Pad] Move X/Y (Axis-locked for precise movement)")
        logger.info(" [X]          Toggle Smoothing Mode (Acceleration ON/OFF)")
        logger.info(" [BACK/SELECT] Switch Control Mode (1/2/3)")
        logger.info(" [A/B/Y]      Rec / Discard / Home")
        logger.info(" [START]      Exit Program")
        logger.info(colored("\n=== CONTROL MODES ===", "cyan"))
        logger.info(" Mode 1 (Default):  RS=Pitch/Roll, LB/RB=Yaw, LT/RT=Z")
        logger.info(" Mode 2 (Needle):   RS=Pitch/Yaw,  LB/RB=Roll, LT/RT=Z")
        logger.info(" Mode 3 (Trigger):  RS=Pitch only, LB/RB=Roll, LT/RT=Yaw")

        while True:
            t0 = time.time()
            
            # 0. Safety Check
            try:
                if robot.GetStatusRobot().error_status:
                    logger.warning("⚠️ Error! Auto-Reset...")
                    robot.ResetError()
                    time.sleep(0.1); robot.ResumeMotion(); time.sleep(0.5)
                    continue
            except Exception as e:
                logger.debug(f"Safety check failed: {e}")

            # 1. Inputs
            frames = cam.get_frames()
            q, p = sampler.get_latest_data()
            act_raw, rec_btn, disc_btn, home_btn, exit_btn = gp.get_action()

            # --- EXIT BUTTON ---
            if exit_btn:
                logger.info(colored("🛑 EXIT button pressed. Shutting down...", "red"))
                break

            # --- HOME BUTTON ---
            if home_btn:
                logger.info(colored("🏠 GOING HOME...", "yellow"))
                try:
                    if robot.GetStatusRobot().error_status:
                        robot.ResetError(); robot.ResumeMotion()
                    robot.MoveJoints(*HOME_JOINTS)
                    robot.WaitIdle()
                except Exception as e: logger.error(f"Home Failed: {e}")
                continue 

            # 2. Move Robot
            if np.any(np.abs(act_raw) > 0.001):
                try:
                    robot.MoveLinRelTrf(*[float(x) for x in act_raw])
                except Exception as e:
                    logger.debug(f"Move command failed: {e}")

            # 3. Recorder
            if rec_btn:
                if not rec.recording: 
                    rec.start()
                    time.sleep(0.5)
                else: 
                    # [수정됨] 비동기 저장 호출
                    rec.save_async() 
                    time.sleep(0.5)
                    
            if disc_btn and rec.recording: rec.discard(); time.sleep(0.5)
            
            rec.add(frames, q, p, act_raw)

            # 4. GUI
            if frames:
                sorted_keys = sorted(frames.keys())
                img_list = []
                for key in sorted_keys:
                    img = frames[key].copy()
                    cv2.putText(img, key, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    img_list.append(img)

                # Combine all camera views horizontally
                if img_list:
                    try:
                        # Ensure all images have the same height for hstack
                        if len(img_list) > 1:
                            heights = [img.shape[0] for img in img_list]
                            if len(set(heights)) > 1:
                                # Resize all to same height if different
                                target_h = min(heights)
                                resized = []
                                for img in img_list:
                                    if img.shape[0] != target_h:
                                        aspect = img.shape[1] / img.shape[0]
                                        target_w = int(target_h * aspect)
                                        resized.append(cv2.resize(img, (target_w, target_h)))
                                    else:
                                        resized.append(img)
                                combined_view = np.hstack(resized)
                            else:
                                combined_view = np.hstack(img_list)
                        else:
                            combined_view = img_list[0]

                        # Status text
                        if rec.recording:
                            txt = f"REC ({len(rec.buffer)})"
                            col = (0, 0, 255) # 빨강
                        elif rec.is_saving:
                            txt = "SAVING..." # 저장 중일 때 표시
                            col = (0, 255, 255) # 노랑
                        else:
                            txt = "IDLE"
                            col = (0, 255, 0) # 초록

                        cv2.putText(combined_view, txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)

                        # Control mode display
                        mode_names = {1: "Mode1: Default", 2: "Mode2: Needle", 3: "Mode3: Trigger"}
                        mode_txt = mode_names.get(gp.control_mode, "Unknown")
                        cv2.putText(combined_view, mode_txt, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # Smoothing mode display
                        smooth_txt = "Smoothing: ON" if gp.smoothing_enabled else "Smoothing: OFF"
                        smooth_col = (0, 255, 255) if gp.smoothing_enabled else (128, 128, 128)
                        cv2.putText(combined_view, smooth_txt, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, smooth_col, 2)

                        # Sensor status display
                        if SENSOR_ENABLED and sensor_sampler and sensor_receiver:
                            sensor_status = sensor_sampler.get_status()
                            receiver_stats = sensor_receiver.get_stats()

                            # Connection status
                            if not receiver_stats['is_receiving']:
                                conn_txt = "Sensor: NO DATA"
                                conn_col = (0, 0, 255)  # Red
                            elif receiver_stats['calibrated']:
                                conn_txt = "Sensor: CONNECTED"
                                conn_col = (0, 255, 0)  # Green
                            else:
                                conn_txt = "Sensor: CALIBRATING..."
                                conn_col = (0, 165, 255)  # Orange
                            cv2.putText(combined_view, conn_txt, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_col, 2)

                            # Packet rate
                            rate_txt = f"Rate: {receiver_stats['packets_per_second']:.0f} pkt/s"
                            cv2.putText(combined_view, rate_txt, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            # Buffer status
                            buffer_txt = f"Buffer: {sensor_status['buffer_size']}/{sensor_status['max_length']}"
                            cv2.putText(combined_view, buffer_txt, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                            # Force value
                            force_txt = f"Force: {sensor_status['latest_force']:.3f}"
                            cv2.putText(combined_view, force_txt, (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        cv2.imshow("Multi-View Dashboard", combined_view)
                    except Exception as e:
                        logger.error(f"GUI display error: {e}")

            # 5. Sensor Visualization (Separate Window)
            if SENSOR_ENABLED and sensor_sampler:
                try:
                    sensor_vis = visualize_sensor_data(sensor_sampler)
                    if sensor_vis is not None:
                        cv2.imshow("Sensor Data (OCT + FPI)", sensor_vis)
                except Exception as e:
                    logger.debug(f"Sensor visualization error: {e}")

            if cv2.waitKey(1) == ord('q'): break
            
            el = time.time() - t0
            if el < 1/CONTROL_FREQUENCY: time.sleep(1/CONTROL_FREQUENCY - el)

    except KeyboardInterrupt: logger.info("Stopped.")
    except Exception as e: logger.error(e, exc_info=True)
    finally:
        # 센서 수신기 종료
        if sensor_stop_event is not None:
            sensor_stop_event.set()
        if sensor_receiver is not None:
            sensor_receiver.join(timeout=2.0)

        if 'sampler' in locals(): sampler.stop(); sampler.join()
        if 'robot' in locals() and robot.IsConnected(): robot.DeactivateRobot(); robot.Disconnect()
        cam.close(); clock.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()