#!/usr/bin/env python3
"""GELLO(Dynamixel, ID 1~7)로 실제 UR5e + Robotiq 그리퍼를 관절각 미러링으로 조종한다.

gello_ur10e_robotiq_sim.py(Isaac Sim)에서 검증한 것과 같은 방식 - GELLO는 UR의
관절 구조를 그대로 축소 복제한 "joint twin"이라, Cartesian FK/IK 없이 관절각을
그대로 미러링한다. 이 스크립트는 시뮬 대신 실제 UR5e 로봇에 RTDE로 명령을 보낸다.

매핑 두 가지 (--mapping):
  official         공식 gello_software와 같은 절대 매핑 (gello_official.py).
                   gello_calibrate.py로 만든 gello_config.json의 오프셋/부호로
                   GELLO 관절각 = (모터각 - 오프셋) x 부호. ENTER 때 어느 관절이든
                   로봇과 0.8 rad(약 46도) 넘게 다르면 공식 run_env.py처럼 시작을
                   거부하고 관절별 차이를 출력한다. 아니면 로봇이 GELLO 자세로
                   (속도 상한 안에서) 다가간 뒤 따라간다.
  relative (기본)  예전 방식. ENTER 순간 GELLO 자세 = 로봇 현재 자세로 기준을 잡고
                   변화량만 따라간다. 부호는 --joint-sign (기본 1,-1,1,-1,1,1).
로봇 제어는 두 매핑 모두 아래 speedJ 방식이다 (공식은 servoJ - 안전장치 때문에 유지).

사용법 (ENTER로 연결/해제 토글):
  1. 실행하면 대기 상태로 시작 - 이때는 로봇에 어떤 명령도 안 나감
  2. GELLO를 지금 실제 UR5e가 서있는 자세랑 최대한 비슷하게 손으로 맞춰놓기
  3. ENTER - official: 차이 확인 후 추종 시작 / relative: 그 순간을 기준으로 latch
  4. 다시 ENTER - 정지(그 자리에서 speedStop), 로봇은 마지막 자세에서 안전하게 멈춤
  5. Ctrl+C - 완전 종료

안전장치:
  - 관절 목표값에 도달하기 위해 속도 상한 + 가속 램프를 건 speedJ만 사용 (moveJ 없음)
  - 매 틱 목표-현재 관절각 오차에 비례한 속도로 접근 (P 제어), 오차가 커도 상한 이내로만 출력
  - GELLO 통신이 --watchdog-timeout 이상 끊기면 자동으로 정지하고 연결 해제 상태로 전환
  - 기준(latch) 대비 --max-joint-delta-deg 넘게 벗어나는 목표는 클램프 (하드웨어 글리치 방어)
  - --dry-run: 실제 speedJ/그리퍼 명령 없이 로그만 출력 (첫 테스트는 반드시 이걸로)

실행 전 로봇 주변 충돌 가능성을 확인하고, 티치펜던트 비상정지에 손이 닿는 상태에서
사용할 것. 기본 --max-joint-speed(0.3 rad/s)는 느리고 보수적인 값이니 처음엔 그대로
쓰는 걸 권장한다.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import dynamixel_sdk as dxl

import gello_official as G

ARM_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_ID = 7
ALL_IDS = ARM_IDS + [GRIPPER_ID]

ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
TICKS_PER_REV = 4096
RAD_PER_TICK = (2.0 * np.pi) / TICKS_PER_REV

CONTROL_HZ = 30.0
CONTROL_DT = 1.0 / CONTROL_HZ

# 한 제어 주기(1/30초)에 허용하는 leader 틱 변화. 400틱은 35도로 너무 커서
# USB 잡음이나 튀는 값이 그대로 로봇 명령이 됐다. 120틱은 약 10.5도다.
MAX_STEP_TICKS_DEFAULT = 120
MAX_STEP_TICKS_GRIP = 200
FILTER_ALPHA_DEFAULT = 0.7
FILTER_ALPHA_GRIP = 0.6
NOISY_IDS: set[int] = set()  # 접촉 불안정 재발하면 여기 축 ID 추가 (예: {4})


def step_limit_for(dxl_id: int) -> int:
    return MAX_STEP_TICKS_GRIP if dxl_id == GRIPPER_ID else MAX_STEP_TICKS_DEFAULT


def alpha_for(dxl_id: int) -> float:
    return FILTER_ALPHA_GRIP if dxl_id == GRIPPER_ID else FILTER_ALPHA_DEFAULT


def wrapped_tick_delta(cur: int, ref: int) -> int:
    """cur-ref를 (-2048, 2048] 범위로 감아서(최단 경로) 반환한다."""
    d = cur - ref
    return ((d + TICKS_PER_REV // 2) % TICKS_PER_REV) - TICKS_PER_REV // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--robot-ip", default="172.28.60.1")
    parser.add_argument("--gripper-port", type=int, default=63352)
    parser.add_argument("--mapping", choices=("official", "relative"), default="relative",
                        help="official: 공식 GELLO 절대 매핑(gello_config.json 필요), "
                             "relative: 예전 ENTER 기준 상대 매핑")
    parser.add_argument("--gello-config", default=str(G.CONFIG),
                        help="official 매핑의 보정 파일 (gello_calibrate.py가 만든다)")
    parser.add_argument("--joint-sign", default=None,
                        help="팔 관절(ID 1~6) 방향 부호, 쉼표구분 6개. official: 보정 파일의 "
                             "부호 대신 쓴다(오프셋은 보정 자세에서 다시 계산). "
                             "relative: 기본 1,-1,1,-1,1,1 (예전 UR5e 실측값)")
    parser.add_argument("--grip-sign", type=float, default=-1.0)
    parser.add_argument("--grip-ticks-to-full", type=float, default=570.0,
                        help="GELLO 그리퍼 모터가 이만큼 틱 움직이면 그리퍼가 완전히 닫힘")
    parser.add_argument("--max-joint-speed", type=float, default=0.3,
                        help="관절 속도 상한 rad/s. 기본은 보수적인 값이다. "
                             "부호/매핑이 틀리면 로봇이 빠르게 잘못된 방향으로 "
                             "가므로, 새 GELLO는 이 값부터 확인하고 올릴 것")
    parser.add_argument("--joint-accel", type=float, default=1.0, help="rad/s^2")
    parser.add_argument("--stop-decel", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=1.2,
                        help="관절 오차 -> 속도 P 게인 (1/s)")
    parser.add_argument("--max-joint-delta-deg", type=float, default=60.0,
                        help="기준 자세 대비 이 각도(deg) 넘게 벗어나는 목표는 클램프 "
                             "(하드웨어 글리치로 인한 큰 목표값 방어)")
    parser.add_argument("--watchdog-timeout", type=float, default=0.3,
                        help="GELLO 데이터가 이 시간(초) 이상 안 갱신되면 자동 정지+연결해제")
    parser.add_argument("--no-gripper", action="store_true",
                        help="팔만 조종하고 그리퍼는 건드리지 않는다. GELLO 그리퍼 "
                             "모터를 움직여도 무시한다 (session 2/3처럼 그리퍼를 "
                             "쓰지 않는 촬영용)")
    parser.add_argument("--view", nargs="?", const="2", default=None, metavar="CAM",
                        help="카메라 하나를 실시간 창으로 띄움(저장 안 함). CAM = 0/1/2 또는 시리얼, "
                             "기본 2(로봇을 마주보는 카메라). --record와 같이 쓰면 무시됨(카메라 점유 충돌); "
                             "--record --no-cameras와는 같이 쓸 수 있음(기록은 팔/그리퍼만, 창은 보기용)")
    parser.add_argument("--record", action="store_true",
                        help="연결(ENTER)~해제(ENTER) 구간을 시연으로 기록: 팔/그리퍼 30Hz + "
                             "카메라 3대 PhysTwin 형식 촬영 (sysid_routing/demo/real_gello/)")
    parser.add_argument("--record-seconds", type=float, default=120.0,
                        help="--record 한 세션의 최대 영상 길이(초). 초당 약 140MB")
    parser.add_argument("--no-cameras", action="store_true", help="--record에서 카메라 촬영은 생략")
    parser.add_argument("--out", default=None, help="--record 저장 폴더 (기본 sysid_routing/demo/real_gello)")
    parser.add_argument("--no-preview", action="store_true", help="--record에서 카메라 미리보기 창을 띄우지 않음")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 speedJ/그리퍼 명령을 보내지 않고 로그만 출력 (첫 테스트 필수)")
    return parser.parse_args()


class RobotiqGripper:
    def __init__(self, host: str, port: int, dry_run: bool):
        self.dry_run = dry_run
        self.available = False
        self.sock: socket.socket | None = None
        if dry_run:
            print("[gripper] dry-run: 소켓 연결 생략")
            return
        try:
            self.sock = socket.create_connection((host, port), timeout=2.0)
            self.sock.settimeout(1.0)
            if self._get("ACT") != 1:
                print("[gripper] WARNING: 그리퍼가 활성화(ACT)되지 않음. "
                      "UR 티치펜던트에서 먼저 activate 해줘. 그리퍼 제어는 비활성화됨.")
                return
            self._set("GTO", 1)
            self.available = True
            print("[gripper] Robotiq 그리퍼 연결 및 활성화 확인됨.")
        except Exception as exc:  # noqa: BLE001
            print(f"[gripper] WARNING: 연결 실패 ({exc}). 그리퍼 제어 없이 진행.")

    def _cmd(self, line: str) -> str:
        assert self.sock is not None
        self.sock.sendall((line + "\n").encode("ascii"))
        return self.sock.recv(128).decode("ascii").strip()

    def _get(self, var: str) -> int:
        return int(self._cmd(f"GET {var}").split()[1])

    def _set(self, var: str, value: int) -> None:
        self._cmd(f"SET {var} {value}")

    def wait_until_stopped(self, timeout: float = 5.0, poll=None):
        """그리퍼가 실제로 멈출 때까지 기다리고 (OBJ 코드, 위치)를 돌려준다.

        고정 시간만 쉬고 넘어가면 다 닫히기 전에 로봇이 출발해 물체를 놓치거나
        끌 수 있다.

        멈춤 판정은 OBJ가 아니라 **위치가 더 이상 변하지 않는 것**으로 한다.
        명령 직후에는 이전 동작의 OBJ가 그대로 남아 있어서 시작하자마자 멈춘
        것으로 오판하기 때문이다. 실측에서 큐브를 제대로 물었는데도 OBJ가
        3(물체 없음)으로 나왔으므로 OBJ만으로는 물림 여부도 판단하지 않는다.

        poll을 주면 기다리는 동안 주기적으로 불러준다 (미리보기 갱신용).
        """
        if self.dry_run or not self.available:
            return None, None
        deadline = time.time() + timeout
        last = None
        stable = 0
        moved = False
        code = position = None
        while time.time() < deadline:
            try:
                position = self._get("POS")
                code = self._get("OBJ")
            except Exception as exc:  # noqa: BLE001
                print(f"[gripper] 상태 읽기 실패: {exc}")
                return None, None
            if last is not None and abs(position - last) <= 1:
                stable += 1
            else:
                if last is not None:
                    moved = True
                stable = 0
            last = position
            # 움직였다가 멈췄거나, 애초에 움직일 필요가 없었거나.
            if stable >= (4 if moved else 12) and code != 0:
                return code, position
            if poll is not None:
                poll()
            else:
                time.sleep(0.02)
        print("[gripper] 경고: 시간 안에 멈추지 않았습니다.")
        return code, position

    def set_position_pct(self, pct: float) -> None:
        pct = float(np.clip(pct, 0.0, 100.0))
        pos_0_255 = int(pct / 100.0 * 255.0)
        self.requested_position = pos_0_255
        if self.dry_run:
            print(f"[dry-run gripper] SET POS {pos_0_255}")
            return
        if not self.available:
            return
        try:
            self._set("POS", pos_0_255)
        except Exception as exc:  # noqa: BLE001
            print(f"[gripper] 명령 실패: {exc}")


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    enabled: bool = False
    shutdown: threading.Event = field(default_factory=threading.Event)
    gripper_target_pct: float = 0.0
    n_sent: int = 0
    n_glitch: int = 0
    # official mapping: the main thread asks for the start check before it
    # starts a recording, so a refused start leaves nothing running
    check_request: bool = False
    check_ok: bool | None = None


def gripper_worker(state: SharedState, args: argparse.Namespace) -> None:
    gripper = RobotiqGripper(args.robot_ip, args.gripper_port, args.dry_run)
    last_sent: float | None = None
    while not state.shutdown.is_set():
        with state.lock:
            enabled = state.enabled
            target = state.gripper_target_pct
        if not enabled:
            # 아직 GELLO 연결(ENTER) 전이면 그리퍼는 절대 건드리지 않는다 - state의
            # gripper_target_pct 기본값(0%)을 실제 목표로 착각해서 시작하자마자
            # 그리퍼가 열려버리는 걸 방지.
            last_sent = None
            time.sleep(0.05)
            continue
        if last_sent is None or abs(target - last_sent) > 1.5:
            gripper.set_position_pct(target)
            last_sent = target
        time.sleep(0.05)


def approach_vector(current: np.ndarray, target: np.ndarray, max_change: float) -> np.ndarray:
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if norm <= max_change or norm < 1e-12:
        return target.copy()
    return current + delta * (max_change / norm)


def read_all_positions(port_handler, packet_handler, group_read) -> dict[int, int]:
    group_read.txRxPacket()
    positions = {}
    for dxl_id in ALL_IDS:
        if group_read.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
            raw = group_read.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if raw >= 2**31:
                raw -= 2**32
            positions[dxl_id] = raw
    return positions


def control_worker(state: SharedState, args: argparse.Namespace) -> None:
    port_handler = dxl.PortHandler(args.port)
    if not port_handler.openPort() or not port_handler.setBaudRate(args.baudrate):
        print(f"[error] failed to open {args.port} @ {args.baudrate}")
        state.shutdown.set()
        return
    packet_handler = dxl.PacketHandler(2.0)
    for dxl_id in ALL_IDS:
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    group_read = dxl.GroupSyncRead(port_handler, packet_handler,
                                    ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for dxl_id in ALL_IDS:
        if not group_read.addParam(dxl_id):
            print(f"[error] GroupSyncRead addParam failed for id={dxl_id}")
            state.shutdown.set()
            return

    rtde_c = rtde_r = None
    if not args.dry_run:
        import rtde_control
        import rtde_receive
        rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
        rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        print(f"[ur5e] connected to {args.robot_ip}")
    else:
        print("[ur5e] dry-run: 실제 연결 생략")

    official = args.mapping == "official"
    joint_sign = np.array([float(x) for x in (args.joint_sign or "1,-1,1,-1,1,1").split(",")],
                          dtype=np.float64)
    assert joint_sign.shape == (6,), "--joint-sign은 값 6개여야 함"
    max_delta = np.radians(args.max_joint_delta_deg)
    gello = G.Mapping(args.gello_cfg) if official else None
    # official: continuous ticks per motor, unwrapped read to read, so a motor
    # crossing 0/4095 does not jump a full turn
    cont_ticks: dict[int, int] = {}
    dry_r = None
    if official and args.dry_run:
        # read-only connection, so the start check can be tried in a dry run
        try:
            import rtde_receive
            dry_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
            print(f"[ur5e] dry-run: 관절각만 읽음 ({args.robot_ip})")
        except Exception as exc:  # noqa: BLE001
            print(f"[ur5e] dry-run: 로봇 관절각을 못 읽음({exc}) - 시작 확인은 GELLO 자세를 로봇 자세로 가정")

    last_good: dict[int, int] = {}
    filtered_delta: dict[int, float] = {}
    ref_ticks: dict[int, int] | None = None
    ur5e_ref_joints: np.ndarray | None = None
    was_enabled = False
    current_qdot = np.zeros(6, dtype=np.float64)
    speed_mode = False
    last_update_time = 0.0
    n_read_fail = 0
    last_fail_print = 0.0
    prev_positions: dict[int, int] = {}

    try:
        while not state.shutdown.is_set():
            loop_start = time.monotonic()
            positions = read_all_positions(port_handler, packet_handler, group_read)

            with state.lock:
                enabled = state.enabled

            if len(positions) < len(ALL_IDS):
                n_read_fail += 1
                if loop_start - last_fail_print > 1.0:
                    missing = set(ALL_IDS) - set(positions)
                    print(f"[leader] read fail (missing ids={missing}, total={n_read_fail})")
                    last_fail_print = loop_start
                time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - loop_start)))
                continue

            for dxl_id in ALL_IDS:
                step_limit = step_limit_for(dxl_id)
                prev = last_good.get(dxl_id)
                if prev is not None and abs(wrapped_tick_delta(positions[dxl_id], prev)) > step_limit:
                    with state.lock:
                        state.n_glitch += 1
                    positions[dxl_id] = prev
                else:
                    last_good[dxl_id] = positions[dxl_id]
            for dxl_id in ALL_IDS:
                if dxl_id not in cont_ticks:
                    cont_ticks[dxl_id] = positions[dxl_id]
                else:
                    cont_ticks[dxl_id] += wrapped_tick_delta(positions[dxl_id], prev_positions[dxl_id])
            prev_positions = dict(positions)
            ticks7 = np.array([cont_ticks[i] for i in ALL_IDS], dtype=np.float64)

            # --- watchdog: 오래 신호 없으면 강제 정지+해제 ---
            watchdog_tripped = enabled and was_enabled and (loop_start - last_update_time) > args.watchdog_timeout
            if watchdog_tripped:
                print("\n[bridge] WATCHDOG: leader data stalled, forcing disable")
                with state.lock:
                    state.enabled = False
                enabled = False

            def start_check():
                if rtde_r is not None:
                    robot_q = np.array(rtde_r.getActualQ())
                elif dry_r is not None:
                    robot_q = np.array(dry_r.getActualQ())
                else:
                    robot_q = gello.raw_joints(ticks7)
                gello.wrap_to(ticks7, robot_q)
                leader_q, _ = gello(ticks7)
                return robot_q, leader_q, G.start_check(leader_q, robot_q)

            with state.lock:
                check_request = state.check_request
            if official and check_request and not enabled:
                _, _, bad = start_check()
                if bad:
                    print("\n[bridge] 시작 거부 - GELLO와 로봇이 0.8 rad(약 46도) 넘게 다른 관절:\n"
                          + G.format_start_check(bad)
                          + "\n  GELLO를 로봇 자세에 더 가깝게 맞추고 다시 ENTER")
                with state.lock:
                    state.check_request, state.check_ok = False, not bad

            if enabled and not was_enabled and official:
                robot_q, leader_q, bad = start_check()
                if bad:
                    # run_env.py: refuse to start, list the joints that are too far apart
                    print("\n[bridge] 시작 거부 - GELLO와 로봇이 0.8 rad(약 46도) 넘게 다른 관절:\n"
                          + G.format_start_check(bad)
                          + "\n  GELLO를 로봇 자세에 더 가깝게 맞추고 다시 ENTER")
                    with state.lock:
                        state.enabled = False
                    enabled = False
                else:
                    ur5e_ref_joints = robot_q
                    print(f"\n[bridge] CONNECTED (official) - 로봇이 GELLO 자세로 다가간 뒤 따라갑니다. "
                          f"차이(deg)={np.round(np.degrees(leader_q - robot_q), 1)}")
            elif enabled and not was_enabled:
                ref_ticks = dict(positions)
                filtered_delta = {dxl_id: 0.0 for dxl_id in ALL_IDS}
                if args.dry_run:
                    ur5e_ref_joints = np.zeros(6)
                else:
                    ur5e_ref_joints = np.array(rtde_r.getActualQ())
                print(f"\n[bridge] CONNECTED - GELLO 기준 latch, UR5e 기준 관절각(deg)="
                      f"{np.round(np.degrees(ur5e_ref_joints), 1)}")
            elif not enabled and was_enabled:
                current_qdot[:] = 0.0
                if rtde_c is not None and speed_mode:
                    rtde_c.speedStop(args.stop_decel)
                speed_mode = False
                print("\n[bridge] DISCONNECTED - 정지")
            was_enabled = enabled

            if enabled and ur5e_ref_joints is not None and (official or ref_ticks is not None):
                last_update_time = loop_start
                if official:
                    leader_q, closure = gello(ticks7)
                    # same glitch guard as relative: never more than max_delta from the connect pose
                    target_joint = np.clip(leader_q, ur5e_ref_joints - max_delta, ur5e_ref_joints + max_delta)
                    grip_pct = closure * 100.0
                else:
                    for dxl_id in ALL_IDS:
                        alpha = alpha_for(dxl_id)
                        raw_delta = wrapped_tick_delta(positions[dxl_id], ref_ticks[dxl_id])
                        prev_f = filtered_delta.get(dxl_id, raw_delta)
                        filtered_delta[dxl_id] = alpha * raw_delta + (1.0 - alpha) * prev_f

                    dq_arm = np.array(
                        [filtered_delta[dxl_id] * RAD_PER_TICK * joint_sign[i]
                         for i, dxl_id in enumerate(ARM_IDS)]
                    )
                    dq_arm = np.clip(dq_arm, -max_delta, max_delta)
                    target_joint = ur5e_ref_joints + dq_arm

                    grip_delta_ticks = filtered_delta[GRIPPER_ID] * args.grip_sign
                    grip_pct = float(np.clip(grip_delta_ticks / args.grip_ticks_to_full, 0.0, 1.0)) * 100.0

                if args.dry_run:
                    actual_joint = ur5e_ref_joints + current_qdot * 0.0  # dry-run: no real feedback
                    if dry_r is not None:
                        actual_joint = np.array(dry_r.getActualQ())
                else:
                    actual_joint = np.array(rtde_r.getActualQ())

                joint_err = target_joint - actual_joint
                desired_qdot = np.clip(joint_err * args.kp, -args.max_joint_speed, args.max_joint_speed)
                current_qdot = approach_vector(current_qdot, desired_qdot, args.joint_accel * CONTROL_DT)

                with state.lock:
                    state.gripper_target_pct = grip_pct
                    state.n_sent += 1
                    n_sent_now = state.n_sent
                    n_glitch_now = state.n_glitch

                if args.dry_run:
                    if np.linalg.norm(current_qdot) > 1e-6:
                        print(f"[dry-run] speedJ={np.round(current_qdot, 3)} "
                              f"target(deg)={np.round(np.degrees(target_joint), 1)} grip={grip_pct:.0f}%")
                else:
                    if not rtde_c.isProgramRunning():
                        # 보호정지가 걸렸거나 펜던트에서 프로그램이 멈추면 여기 온다.
                        # 계속 speedJ를 보내봐야 같은 에러만 반복 출력된다.
                        print("\n[ur5e] RTDE 제어 스크립트가 멈췄습니다. "
                              "보호정지(protective stop)나 비상정지일 수 있습니다.\n"
                              "       티치펜던트에서 원인을 해제하고 이 프로그램을 "
                              "다시 실행하세요.")
                        current_qdot[:] = 0.0
                        speed_mode = False
                        with state.lock:
                            state.enabled = False
                        state.shutdown.set()
                        break
                    rtde_c.speedJ(current_qdot.tolist(), args.joint_accel, CONTROL_DT * 2.0)
                    speed_mode = True

                if n_sent_now % (int(CONTROL_HZ) * 5) == 0:
                    print(f"[status] sent={n_sent_now} glitches={n_glitch_now} "
                          f"target(deg)={np.round(np.degrees(target_joint), 1)} grip={grip_pct:.0f}%")

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, CONTROL_DT - elapsed))
    except Exception as exc:  # noqa: BLE001
        print(f"\n[CONTROL ERROR] {type(exc).__name__}: {exc}", flush=True)
        with state.lock:
            state.enabled = False
    finally:
        if rtde_c is not None:
            try:
                rtde_c.speedStop(args.stop_decel)
            except Exception:
                pass
            try:
                rtde_c.stopScript()
            except Exception:
                pass
            try:
                rtde_c.disconnect()
            except Exception:
                pass
        for receiver in (rtde_r, dry_r):
            if receiver is not None:
                try:
                    receiver.disconnect()
                except Exception:
                    pass
        port_handler.closePort()


def main() -> int:
    args = parse_args()
    if args.mapping == "official":
        try:
            args.gello_cfg = G.GelloConfig.load(args.gello_config)
        except FileNotFoundError as exc:
            print(f"[error] {exc}\n        (예전 방식으로 쓰려면 --mapping relative)")
            return 1
        if args.joint_sign:
            args.gello_cfg = args.gello_cfg.with_signs([int(float(v)) for v in args.joint_sign.split(",")])
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN 모드 - 실제 로봇에 명령 안 보냄")
        print("=" * 60)
    else:
        print("=" * 60)
        print("실제 UR5e 로봇을 움직입니다. 주변 충돌 가능성을 확인하고")
        print("티치펜던트 비상정지에 손이 닿는 상태에서 진행하세요.")
        print("=" * 60)

    state = SharedState()
    worker = threading.Thread(target=control_worker, args=(state, args),
                               name="gello-ur5e-control", daemon=True)
    gripper_thread = None
    if not args.no_gripper:
        gripper_thread = threading.Thread(target=gripper_worker, args=(state, args),
                                           name="gello-ur5e-gripper", daemon=True)
    worker.start()
    if gripper_thread is not None:
        gripper_thread.start()

    if args.no_gripper:
        print("\n[gripper] --no-gripper: 그리퍼에 연결하지 않고 명령도 보내지 않습니다. "
              "지금 잡혀 있는 상태 그대로 유지됩니다.")
    if args.mapping == "official":
        c = args.gello_cfg
        print(f"\n[mapping] official (공식 GELLO)  부호={c.joint_signs}  보정={c.calibrated_at}  "
              f"오프셋/(pi/2)={[int(np.round(x / (np.pi / 2))) for x in c.joint_offsets]}")
    else:
        print(f"\n[mapping] relative  [joint-sign] {args.joint_sign or '1,-1,1,-1,1,1'}  [grip-sign] {args.grip_sign}")
    print(f"[limits] max_joint_speed={args.max_joint_speed} rad/s, "
          f"max_joint_delta={args.max_joint_delta_deg} deg")
    print("ENTER = 연결/해제 토글, Ctrl+C = 종료")
    print("(연결하는 순간 GELLO 자세 = UR5e 현재 관절각으로 매핑 확정됨 - 먼저 GELLO를 "
          "UR5e 실제 자세랑 맞춰두고 ENTER)")

    viewer = None; viewer_stop = None
    if args.view is not None and args.record and not args.no_cameras:
        print("[view] --record가 카메라 3대를 모두 쓰므로 --view는 무시합니다 (녹화 미리보기 창이 대신 뜹니다).")
    elif args.view is not None:
        import subprocess, tempfile
        from pathlib import Path as _Path
        root = _Path(__file__).resolve().parents[2]
        py = _Path.home() / "anaconda3/envs/realsense/bin/python"
        viewer_stop = _Path(tempfile.gettempdir()) / f"gello_view_stop_{int(time.time())}"
        viewer = subprocess.Popen([str(py), str(root / "ur5e_real2sim/view_camera.py"), "--cam", args.view,
                                   "--stop-file", str(viewer_stop)])
        print(f"[view] cam {args.view} 실시간 창을 띄웁니다 (저장 안 함).")
    recorder = None
    if args.record:
        from demo_recorder import DemoRecorder
        recorder = DemoRecorder(args.robot_ip, state, args.gripper_port,
                                max_seconds=args.record_seconds, cameras=not args.no_cameras,
                                preview=not args.no_preview, out=args.out)
        print("[record] ON: 지금 카메라 미리보기 창이 뜹니다(저장 안 함). 연결 ENTER에서 녹화가 시작된 뒤 "
              "GELLO가 연결되고, 해제 ENTER에서 저장됩니다.")
        recorder.prepare()
    try:
        while not state.shutdown.is_set():
            input()
            if state.shutdown.is_set():
                break
            with state.lock:
                enabling = not state.enabled
            if enabling and args.mapping == "official":
                # run_env.py's start check, before anything (recording) starts
                with state.lock:
                    state.check_request, state.check_ok = True, None
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with state.lock:
                        ok = state.check_ok
                    if ok is not None:
                        break
                    time.sleep(0.02)
                with state.lock:
                    state.check_request = False
                if ok is None:
                    print("[toggle] GELLO 값을 읽지 못해 시작 확인을 못 했습니다 - 연결 안 함")
                    continue
                if not ok:
                    continue
            if enabling and recorder is not None:
                recorder.start()              # cameras first; the arm is released after
            with state.lock:
                state.enabled = enabling
                status = "연결" if state.enabled else "해제"
            print(f"[toggle] -> {status}")
            if not enabling and recorder is not None:
                recorder.stop()
    except (KeyboardInterrupt, EOFError):
        print("\n[bridge] stopped by user")
    finally:
        with state.lock:
            state.enabled = False
        if recorder is not None:
            recorder.stop(final=True)
        if viewer is not None:
            try:
                viewer_stop.write_text("stop\n"); viewer.wait(timeout=5)
            except Exception:
                viewer.kill()
            try: viewer_stop.unlink()
            except Exception: pass
        state.shutdown.set()
        worker.join(timeout=3.0)
        if gripper_thread is not None:
            gripper_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
