# GELLO로 UR5e 조종하기 (시뮬레이터 + 실제 로봇)

GELLO(Dynamixel XL330 7개, ID 1~6 팔 / ID 7 그리퍼)로 UR5e + Robotiq 2F-85를
관절각 그대로 따라 움직이게 한다.

- **시뮬레이터**: 실제 책상을 스캔해 만든 장면(RealTable)에서 로프를 집고 옮겨 볼 수 있다.
  가우시안 스플래팅 렌더링 + 스프링-질량 로프 물리. **NVIDIA GPU(CUDA) 필요.**
- **실제 로봇**: RTDE로 UR5e를, URCap 소켓으로 Robotiq 그리퍼를 움직인다. GPU 필요 없음.

조종 방식은 기본이 **relative**다. 시작 키를 누른 순간 GELLO 자세 = 로봇 자세로 보고,
그 뒤 GELLO가 움직인 만큼만 따라간다. 그래서 시작 전에 GELLO를 화면/실제 로봇 자세와
비슷하게 맞춰 두면 된다. (공식 gello_software처럼 보정해서 절대 각도로 쓰는 방식은
`--mapping official`, 아래 "공식 방식" 참고.)

## 폴더

```
run_sim.sh                     시뮬레이터 실행
setup.sh                       시뮬레이터 설치 (real2sim-eval/.venv)
sysid_routing/code/gello_routing.py   시뮬레이터 GELLO 조종 본체
sysid_routing/scene/           장면 설정 (routing_measured.yaml)
ur5e_real2sim/                 장면 데이터 (책상 스캔, 로프 스캔, 로프 물성, 클립)
real2sim-eval/                 시뮬레이터 (Kaifeng Zhang et al., MIT, 아래 출처)
ur5e_test/control/
  gello_ur5e_real_teleop.py    실제 UR5e 조종
  gello_wire_check.py          선 연결 확인 (USB + 모터 1~7, 1초마다)
  gello_monitor.py             모터 각도 실시간 보기 (관절 방향 확인)
  gello_calibrate.py           공식 방식 보정 (--mapping official 쓸 때만)
  gello_official.py            공식 gello_software 매핑 계산식
  stop_robot.py                실제 로봇 RTDE 제어 강제 정지
  GELLO_SETUP.md               하드웨어, 모터 ID 설정, 전원(5V) 주의
```

## 1. GELLO 연결 확인 (공통)

```bash
sudo usermod -aG dialout "$USER"      # 시리얼 포트 권한, 한 번만. 이후 로그아웃/로그인
pip install dynamixel-sdk             # 시뮬레이터를 설치했다면 .venv에 이미 있음
python ur5e_test/control/gello_wire_check.py
```

`USB OK | 모터 1~7 전부 OK`가 나와야 한다.

| 출력 | 확인할 곳 |
|---|---|
| `USB 없음` | USB 케이블, U2D2. 허브 말고 PC 뒷면 포트에 바로 꽂기 |
| `모터 응답 없음` | 모터 5V 전원, U2D2 - Power Hub 3핀 케이블 |
| `응답 [1, 2, 3], 없음 [4, ...]` | 마지막으로 응답한 모터와 그다음 모터 사이 케이블 |

## 2. 시뮬레이터

### 설치 (처음 한 번)

검증 환경: Ubuntu, NVIDIA RTX 3090, CUDA Toolkit 12.8, Python 3.11.

```bash
sudo apt install build-essential libglm-dev libgl1 libegl1 libglib2.0-0
# CUDA 12.8 Toolkit(nvcc)과 Python 3.11이 있어야 한다. 예: conda create -n gello python=3.11 && conda activate gello
bash setup.sh
```

`setup.sh`가 `real2sim-eval/.venv`를 만들고 torch 2.7.1(cu128), warp, 렌더러
(`third-party/diff-gaussian-rasterization-w-depth`, CUDA 빌드)를 설치한다.

### 실행

```bash
bash run_sim.sh                        # GELLO 연결된 상태
bash run_sim.sh --fake                 # GELLO 없이 화면만 확인 (팔이 저절로 조금씩 움직임)
bash run_sim.sh --substep-scale 0.5    # 더 빠름 (로프 물리를 거칠게 계산, 연습용)
```

창이 뜨면:

| 키 | 동작 |
|---|---|
| SPACE | 따라가기 시작 / 멈춤 (시작 전에 GELLO를 화면 속 팔 자세와 비슷하게 맞출 것) |
| O / C | 멈춘 상태에서 GELLO 그리퍼를 완전히 벌리고 O, 완전히 쥐고 C (그리퍼 범위 저장) |
| 0 | 장면 리셋 |
| ESC | 종료 |

- 관절 방향 부호 기본값은 `--joint-sign 1,1,-1,1,1,1` (공식 GELLO의 UR 값).
  어떤 관절이 반대로 움직이면 그 자리 부호를 바꿔서 실행한다.
- 창이 안 보이면 화면이 연결된 GPU 번호로: `CUDA_VISIBLE_DEVICES=0 bash run_sim.sh`
- 속도(RTX 3090): 약 15~17 Hz, `--substep-scale 0.5`면 약 20 Hz. 로프 물리가 대부분의 시간을 쓴다.

## 3. 실제 UR5e

### 설치

```bash
pip install numpy dynamixel-sdk ur-rtde
```

로봇 준비: 티치펜던트에서 Remote Control 모드, Robotiq URCap 설치 + 그리퍼 활성화,
PC와 로봇이 같은 네트워크 (기본 로봇 IP `172.28.60.1`, 다르면 `--robot-ip`).

### 실행

```bash
cd ur5e_test/control
python gello_ur5e_real_teleop.py --dry-run          # 처음엔 반드시: 로봇에 명령 안 보내고 로그만
python gello_ur5e_real_teleop.py                    # 실제 조종
python gello_ur5e_real_teleop.py --robot-ip 192.168.0.10 --port /dev/ttyUSB1
python gello_ur5e_real_teleop.py --no-gripper       # 팔만
python stop_robot.py                                # 비상시: 스크립트가 죽었는데 로봇 제어가 남아 있을 때
```

조작: 실행 → GELLO를 **실제 로봇 자세와 비슷하게** 맞춤 → **ENTER** (추종 시작) →
다시 **ENTER** (정지) → **Ctrl+C** (종료).

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--joint-sign` | `1,-1,1,-1,1,1` | 관절 1~6 방향 부호 (우리 GELLO + UR5e에서 맞춘 값) |
| `--max-joint-speed` | `0.3` rad/s | 관절 속도 상한. 처음엔 그대로 둘 것 |
| `--max-joint-delta-deg` | `60` | 시작 자세에서 이 이상 벗어난 목표는 잘라냄 |
| `--watchdog-timeout` | `0.3` s | GELLO 신호가 끊기면 자동 정지 |

`--record`, `--view`는 우리 연구실 카메라 코드가 필요해서 이 저장소에서는 동작하지 않는다.

**안전**: 로봇 주변을 비우고 티치펜던트 비상정지에 손이 닿게 둔다. 새 GELLO는 `--dry-run`과
낮은 속도로 관절 하나씩 방향을 먼저 확인한다. 시뮬레이터와 실제 로봇의 부호 기본값이 서로
다르니, 실제 로봇은 반드시 dry-run부터.

## 공식 방식 (선택)

공식 [gello_software](https://github.com/wuphilipp/gello_software)처럼 기준 자세에서 한 번
보정한 뒤 절대 관절각으로 쓰려면:

```bash
python ur5e_test/control/gello_calibrate.py        # GELLO를 0,-90,90,-90,-90,0도에 놓고, 그리퍼 벌린 채 Enter
bash run_sim.sh --mapping official
python ur5e_test/control/gello_ur5e_real_teleop.py --mapping official --dry-run
```

시작할 때 GELLO와 로봇이 0.8 rad(약 46도) 넘게 다른 관절이 있으면 공식 코드처럼 시작을 거부한다.

## 출처

- 시뮬레이터 `real2sim-eval/`: Kaifeng Zhang et al., *Real-to-Sim Robot Policy Evaluation with
  Gaussian Splatting Simulation of Soft-Body Interactions*, https://github.com/kywind/real2sim-eval
  (MIT, `real2sim-eval/LICENSE`). 이 저장소에는 UR5e 지원과 몇 가지 수정이 들어간 사본이 있다.
- `ur5e_test/control/gello_official.py`의 보정/매핑 계산식: https://github.com/wuphilipp/gello_software (MIT).
