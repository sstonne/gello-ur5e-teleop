"""GELLO joint mapping exactly as the official gello_software computes it.

Source: github.com/wuphilipp/gello_software (MIT, Philipp Wu), commit 1b3fec5.
What is reproduced here, formula for formula:

  scripts/gello_get_offset.py      offsets: per joint, the multiple of pi/2 in
                                   [-8pi, 8pi] that best puts sign*(raw-offset)
                                   on the known calibration pose; gripper
                                   open = raw - 0.2 deg, close = raw - 42 deg
  gello/robots/dynamixel.py        joint = (raw - offset) * sign, raw = ticks /
                                   2048 * pi (signed 32-bit present position);
                                   gripper = clip((raw - open) / (close - open));
                                   start_joints: offsets shifted by whole turns
                                   to land nearest the start pose; smoothing
                                   alpha 0.99
  experiments/run_env.py           start check: refuse when any joint is more
                                   than 0.8 rad from the follower; then approach
                                   at most 0.05 rad per step for 25 steps

Not taken from upstream: its DynamixelDriver kills whatever process holds the
port and silently falls back to a fake driver that reads all zeros when the
port fails, which is not acceptable next to a real arm. Reading is done here
instead (Reader), with the same signed 32-bit present-position read, plus
incremental unwrapping so a motor in single-turn position mode cannot jump a
full turn while it is being followed.

The calibration lives in gello_config.json next to this file (written by
gello_calibrate.py). It keeps the raw readings taken at the calibration pose,
so offsets for other joint signs are recomputed from the same pose without
posing GELLO again.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().with_name('gello_config.json')
TICKS_TO_RAD = np.pi / 2048.0          # driver.py: _j / 2048.0 * np.pi
UR_START_JOINTS = (0., -1.57, 1.57, -1.57, -1.57, 0.)   # README, UR command example
UR_JOINT_SIGNS = (1, 1, -1, 1, 1, 1)                    # README, "Joint Signs Reference"
MAX_START_DELTA = 0.8                  # run_env.py: max_joint_delta
APPROACH_STEP = 0.05                   # run_env.py: max_delta per step
APPROACH_STEPS = 25                    # run_env.py: range(25)
SMOOTHING_ALPHA = 0.99                 # dynamixel.py: self._alpha


def find_offsets(raw_rad, start_joints, joint_signs):
    """gello_get_offset.py get_config(): brute force over multiples of pi/2."""
    offsets, errors = [], []
    for i, (raw, start, sign) in enumerate(zip(raw_rad, start_joints, joint_signs)):
        best_offset, best_error = 0., 1e6
        for offset in np.linspace(-8 * np.pi, 8 * np.pi, 8 * 4 + 1):  # intervals of pi/2
            error = abs(sign * (raw - offset) - start)
            if error < best_error:
                best_error, best_offset = error, offset
        offsets.append(float(best_offset))
        errors.append(float(best_error))
    return offsets, errors


def gripper_open_close(raw_rad):
    """gello_get_offset.py: printed open/close degrees, from the open gripper."""
    deg = float(np.rad2deg(raw_rad))
    return deg - 0.2, deg - 42.


@dataclass
class GelloConfig:
    joint_signs: list
    start_joints: list
    calib_raw_ticks: list              # IDs 1..7 at the calibration pose
    joint_offsets: list = field(default_factory=list)
    calib_errors_rad: list = field(default_factory=list)
    gripper_open_deg: float = 0.
    gripper_close_deg: float = 0.
    joint_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    gripper_id: int = 7
    port: str = ''
    calibrated_at: str = ''

    @classmethod
    def calibrate(cls, raw_ticks, start_joints=UR_START_JOINTS, joint_signs=UR_JOINT_SIGNS, port=''):
        cfg = cls(joint_signs=[int(s) for s in joint_signs], start_joints=[float(v) for v in start_joints],
                  calib_raw_ticks=[int(v) for v in raw_ticks], port=port,
                  calibrated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        cfg.recompute()
        return cfg

    def recompute(self):
        raw = np.asarray(self.calib_raw_ticks, float) * TICKS_TO_RAD
        self.joint_offsets, self.calib_errors_rad = find_offsets(raw[:6], self.start_joints, self.joint_signs)
        self.gripper_open_deg, self.gripper_close_deg = gripper_open_close(raw[6])

    def with_signs(self, joint_signs):
        """Same calibration pose, other signs: offsets recomputed, not re-posed."""
        other = GelloConfig(**asdict(self))
        other.joint_signs = [int(s) for s in joint_signs]
        other.recompute()
        return other

    def save(self, path=CONFIG):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n')

    @classmethod
    def load(cls, path=CONFIG):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f'{path} not found: put GELLO in the calibration pose and run '
                f'ur5e_test/control/gello_calibrate.py first')
        return cls(**json.loads(path.read_text()))


class Mapping:
    """DynamixelRobot.get_joint_state() for one calibration."""

    def __init__(self, cfg: GelloConfig, alpha=SMOOTHING_ALPHA):
        self.cfg = cfg
        self.offsets = np.asarray(cfg.joint_offsets, float)
        self.signs = np.asarray(cfg.joint_signs, float)
        self.open = np.deg2rad(cfg.gripper_open_deg)
        self.close = np.deg2rad(cfg.gripper_close_deg)
        self.alpha = alpha
        self.last = None

    def raw_joints(self, ticks):
        return (np.asarray(ticks[:6], float) * TICKS_TO_RAD - self.offsets) * self.signs

    def wrap_to(self, ticks, start_joints):
        """DynamixelRobot.__init__(start_joints=...): whole-turn offset shift."""
        current = self.raw_joints(ticks)
        self.offsets = self.offsets + 2 * np.pi * np.round(
            (current - np.asarray(start_joints, float)) / (2 * np.pi)) * self.signs
        self.last = None

    def gripper(self, ticks):
        g = (float(ticks[6]) * TICKS_TO_RAD - self.open) / (self.close - self.open)
        return min(max(0., g), 1.)

    def __call__(self, ticks):
        """Six joint angles (rad) and gripper closure 0 (open) .. 1 (closed)."""
        pos = np.r_[self.raw_joints(ticks), self.gripper(ticks)]
        if self.last is not None:
            pos = self.last * (1 - self.alpha) + pos * self.alpha
        self.last = pos
        return pos[:6].copy(), float(pos[6])


def start_check(leader, follower, max_joint_delta=MAX_START_DELTA):
    """run_env.py: joints too far apart to start. Empty list = may start."""
    leader, follower = np.asarray(leader, float), np.asarray(follower, float)
    deltas = np.abs(leader - follower)
    return [(i + 1, float(deltas[i]), float(leader[i]), float(follower[i]))
            for i in np.flatnonzero(deltas > max_joint_delta)]


def format_start_check(bad):
    return '\n'.join(f'  joint {i}: delta {np.degrees(d):6.1f} deg  GELLO {np.degrees(l):7.1f}  '
                     f'robot {np.degrees(f):7.1f}' for i, d, l, f in bad)


def approach(current, target, step=APPROACH_STEP):
    """run_env.py: move at most `step` rad on the worst joint."""
    delta = np.asarray(target, float) - np.asarray(current, float)
    worst = np.abs(delta).max()
    if worst > step:
        delta = delta / worst * step
    return np.asarray(current, float) + delta


class Reader:
    """Background read of present position (signed 32-bit, as upstream), IDs 1..7.

    Positions are unwrapped against the previous read, so a single-turn motor
    crossing 0/4095 keeps counting instead of jumping by 4096 ticks; a motor in
    extended mode reads the same either way. Torque is switched off so the arm
    moves by hand. A read that moves any motor more than `spike_ticks` within
    1/30 s is treated as a glitch and dropped.
    """

    def __init__(self, port='/dev/ttyUSB0', baudrate=57600, ids=range(1, 8), spike_ticks=400):
        import dynamixel_sdk as dxl
        self.dxl, self.ids, self.spike = dxl, list(ids), spike_ticks
        self.port = dxl.PortHandler(port)
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.ticks, self.raw, self.stamp, self.error = None, None, 0., 'waiting for GELLO'
        if not self.port.openPort():
            raise RuntimeError(f'cannot open {port}')
        try:
            if not self.port.setBaudRate(baudrate):
                raise RuntimeError(f'cannot set baud rate {baudrate}')
            packet = dxl.PacketHandler(2.0)
            for motor in self.ids:
                for _ in range(3):
                    self.port.clearPort()
                    _, result, error = packet.ping(self.port, motor)
                    if result == dxl.COMM_SUCCESS and not error:
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError(f'GELLO ID {motor} did not respond')
            for motor in self.ids:
                result, error = packet.write1ByteTxRx(self.port, motor, 64, 0)   # torque off
                if result != dxl.COMM_SUCCESS or error:
                    raise RuntimeError(f'cannot switch torque off on ID {motor}')
            self.group = dxl.GroupSyncRead(self.port, packet, 132, 4)
            for motor in self.ids:
                if not self.group.addParam(motor):
                    raise RuntimeError(f'cannot register ID {motor}')
        except Exception:
            self.port.closePort()
            raise
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while not self.stop.is_set():
                result = self.group.txRxPacket()
                if result == self.dxl.COMM_SUCCESS and all(self.group.isAvailable(i, 132, 4) for i in self.ids):
                    raw = np.array([self.group.getData(i, 132, 4) for i in self.ids], dtype=np.int64)
                    raw[raw > 0x7FFFFFFF] -= 0x100000000          # two's complement, as upstream
                    now = time.monotonic()
                    with self.lock:
                        if self.raw is None:
                            self.ticks, self.raw = raw.copy(), raw.copy()
                        else:
                            change = (raw - self.raw + 2048) % 4096 - 2048
                            limit = self.spike * max(1., (now - self.stamp) * 30)
                            if np.any(np.abs(change) > limit):
                                self.error = f'encoder spike on ID {self.ids[int(np.argmax(np.abs(change)))]}'
                                self.stop.wait(1 / 60)
                                continue
                            self.ticks = self.ticks + change
                            self.raw = raw
                        self.stamp, self.error = now, ''
                else:
                    with self.lock:
                        self.error = 'incomplete GELLO packet'
                self.stop.wait(1 / 100)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.error = str(exc)
        finally:
            self.port.closePort()

    def get(self):
        """(ticks or None, monotonic stamp, error string)."""
        with self.lock:
            return (None if self.ticks is None else self.ticks.copy()), self.stamp, self.error

    def wait(self, timeout=3.):
        deadline = time.monotonic() + timeout
        while self.get()[0] is None and time.monotonic() < deadline:
            time.sleep(.02)
        ticks, _, error = self.get()
        if ticks is None:
            raise RuntimeError(f'GELLO: {error}')
        return ticks

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
