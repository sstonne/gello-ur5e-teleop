# GELLO 리더암 연결/셋업 정리

GELLO는 U2D2 보드에 물린 Dynamixel(XL330) 모터 7개로 이루어진 리더암이다.
UR 계열 로봇의 관절 구조를 그대로 축소 복제한 "joint twin" 설계라서,
관절각을 그대로 미러링하는 방식으로 UR5e/UR10e를 조종한다 (Cartesian FK/IK 불필요).

- ID 1~6: 팔 관절 6축 (model 1200 = XL330-M288)
- ID 7: 그리퍼 (model 1190 = XL330-M077, 감속비 다름)

## 1. 하드웨어

- U2D2 보드(USB-Dynamixel 변환기) + Dynamixel Power Hub 보드
- **전원은 반드시 5V(정격, 4A 권장)** — U2D2/Power Hub는 입력 전압을 그대로
  모터에 통과시키는 구조라, 12V 같은 걸 잘못 물리면 XL330 정격(5V, 허용범위
  3.7~6.0V)을 크게 초과해서 모터 보호회로가 걸리거나 손상 위험이 있다.
  **전원 입력은 하나만** 연결할 것 (배럴잭/스크류터미널 동시 연결 금지 -
  ROBOTIS 공식 경고사항).
- USB로 컴퓨터에 연결하면 `/dev/ttyUSB0` 같은 시리얼 포트로 잡힌다
  (`ls /dev/serial/by-id/`로 확인).
- 커넥터 접촉불량이 흔한 고장 원인이다 - 스캔이 불안정하면 케이블을 다른 걸로
  바꿔보거나 커넥터를 끝까지 눌러서 고정할 것.

## 2. 모터 ID 설정 (필수, 최초 1회)

Dynamixel 모터는 공장 출고 시 전부 같은 ID(보통 1)를 가지고 있다. 여러 개를
데이지체인으로 한 버스에 연결하면 ID가 겹쳐서 통신 자체가 안 되거나(무응답)
불안정하게(가끔만 응답) 동작한다. **반드시 모터를 하나씩만 연결한 상태에서
ID를 순서대로(베이스=1 ~ 그리퍼=7) 겹치지 않게 설정**해야 한다.

절차 (모터마다 반복):
1. 그 모터 하나만 U2D2에 연결 (나머지는 케이블 뽑아둠)
2. 아래 스캔 명령으로 현재 ID 확인 (보통 기본값 1)
3. 원하는 ID로 덮어쓰기
4. 확인 핑으로 검증

```bash
# 현재 연결된 모터들 스캔 (baudrate는 항상 57600, protocol 2.0)
sg dialout -c "python -c \"
import dynamixel_sdk as dxl
port = dxl.PortHandler('/dev/ttyUSB0')
port.openPort()
port.setBaudRate(57600)
ph = dxl.PacketHandler(2.0)
found = []
for dxl_id in range(0, 253):
    model, comm, err = ph.ping(port, dxl_id)
    if comm == dxl.COMM_SUCCESS:
        found.append((dxl_id, model))
print(found)
port.closePort()
\""
```

```bash
# ID 변경 (아래 예시는 1 -> 4로 변경. NEW_ID만 바꿔가며 반복)
sg dialout -c "python -c \"
import dynamixel_sdk as dxl
ADDR_TORQUE_ENABLE = 64
ADDR_ID = 7
OLD_ID = 1
NEW_ID = 4
port = dxl.PortHandler('/dev/ttyUSB0')
port.openPort()
port.setBaudRate(57600)
ph = dxl.PacketHandler(2.0)
ph.write1ByteTxRx(port, OLD_ID, ADDR_TORQUE_ENABLE, 0)  # EEPROM 쓰기 전 torque off
result, err = ph.write1ByteTxRx(port, OLD_ID, ADDR_ID, NEW_ID)
print('write:', ph.getTxRxResult(result))
model, comm, err = ph.ping(port, NEW_ID)
print('verify:', ph.getTxRxResult(comm), model if comm==dxl.COMM_SUCCESS else '')
port.closePort()
\""
```

ID는 모터 내부에 영구 저장되므로(EEPROM), 한 번 설정해두면 전원을 껐다 켜도
유지된다. 다 설정한 뒤 전체 체인을 연결하고 스캔하면 `[(1,1200),(2,1200),
(3,1200),(4,1200),(5,1200),(6,1200),(7,1190)]`처럼 7개가 한 번에 잡혀야 한다.
3번 이상 연속으로 스캔해서 매번 안정적으로 다 잡히는지 확인할 것 - 한 번이라도
빠지면 그 사이 커넥터를 의심.

## 3. 스크립트 목록 (`ur5e_test/control/`)

모두 `ur5e-eye-to-hand` conda 환경에서 `dynamixel_sdk`로 GELLO를 직접 읽는다
(`lerobot`은 안 씀 - SO-101/Feetech용이라 이 하드웨어와 프로토콜이 다름).

| 스크립트 | 용도 |
|---|---|
| `gello_udp_publisher.py` | GELLO 관절 델타를 UDP로 브로드캐스트 (Isaac Sim 등 다른 파이썬 환경과 연동할 때) |
| `gello_ur10e_robotiq_sim.py` | Isaac Sim에서 UR10e+Robotiq를 GELLO로 조종 (raw Isaac Sim 빌드의 python.sh로 실행) |
| `gello_ur5e_real_teleop.py` | 실제 UR5e+Robotiq를 GELLO로 조종 (터미널, ENTER로 연결/해제 토글) |
| `gello_ur5e_real_teleop_gui.py` | 위와 동일 + GUI (연결 상태 표시, 카메라 미리보기, 그리퍼 on/off 선택) |
| `run_gello_ur5e_teleop_gui.sh` | 바탕화면 아이콘에서 GUI 버전을 실행하는 래퍼 스크립트 |

### 실행 예시

```bash
conda activate ur5e-eye-to-hand

# 실제 UR5e, 터미널 버전 (안전 확인 후 실제로 움직임)
sg dialout -c "python ur5e_test/control/gello_ur5e_real_teleop.py"

# 실제 UR5e, GUI 버전 (한글 렌더링 때문에 시스템 python3로 실행해야 함 - 아래 4번 참고)
sg dialout -c "/usr/bin/python3 ur5e_test/control/gello_ur5e_real_teleop_gui.py"

# GELLO 포트가 다르면(예: 두 번째 GELLO)
... --port /dev/ttyUSB1
```

사용 흐름은 공통이다: 실행 → GELLO를 로봇(또는 시뮬 로봇)의 현재/홈 자세랑
최대한 맞춰놓기 → ENTER(또는 GUI 버튼) → 그 순간 자세가 기준(0)으로 latch되고
추종 시작 → 다시 ENTER로 정지.

### 관절 방향/그리퍼 보정값 (이미 확인된 기본값)

- `--joint-sign 1,1,-1,1,1,1` (팔꿈치만 반대 방향)
- `--grip-sign -1`
- `--grip-ticks-to-full 570` (그리퍼 전체 스트로크에 해당하는 GELLO 틱 수)

다른 GELLO 유닛이나 조립 방향이 다르면 값이 달라질 수 있으니, 실제로 GELLO를
움직여보면서 로그(`dq(deg)`, `grip_delta_ticks`)를 보고 재조정할 것.

## 4. GUI 버전 특이사항: 반드시 시스템 python3로 실행

`ur5e-eye-to-hand` conda 환경의 Tk는 **Xft(폰트 렌더링 라이브러리) 없이
빌드되어 있어서 한글이 전부 깨진다** (글꼴 이름 문제가 아니라 그 conda 환경의
Tk 자체가 CJK 폰트를 하나도 못 그림 - `ldd libtk8.6.so`로 확인 가능).

시스템 python3(`/usr/bin/python3`)의 Tk는 정상이라, GUI 버전은 그쪽으로
실행해야 한다. 필요한 패키지는 시스템 python3에 `--user`로 설치해뒀다:

```bash
/usr/bin/python3 -m pip install --user dynamixel-sdk ur-rtde pyrealsense2
```

(pip이 없으면 먼저 `curl -sS https://bootstrap.pypa.io/get-pip.py | /usr/bin/python3 - --user`)

## 5. 바탕화면 아이콘

`~/Desktop/GELLO-UR5e-Teleop.desktop`을 더블클릭하면 터미널이 열리면서
`run_gello_ur5e_teleop_gui.sh`가 실행된다. 실행 로그는 `/tmp/gello_ur5e_teleop_launch.log`에도
남는다. 더블클릭해도 안 열리면 아이콘 우클릭 → "실행 허용"(Allow Launching)을
한 번 해줘야 할 수 있다 (Nautilus가 처음 보는 실행파일을 막아두는 보안 기능).

## 6. 자주 겪은 문제 / 원인

- **모터 몇 개가 계속 무응답이거나 가끔만 잡힘** → 대부분 ID 충돌(2장 참고) 또는
  커넥터 접촉불량. 모터 하나만 남기고 스캔했을 때도 불안정하면 케이블/커넥터
  문제, 안정적이면 ID 충돌 쪽을 의심.
- **raw 관절값이 갑자기 수백 도씩 튐** → XL330 Present Position은 단일회전
  0~4095 절대값이라, 기준점 대비 ±180°를 넘어가면 델타 계산이 wrap되면서
  불연속으로 보일 수 있다 (정상 동작, 실사용에서 관절을 180° 넘게 돌릴 일은
  거의 없어서 문제 안 됨). 이거랑 별개로 진짜 노이즈성 글리치라면 접촉불량을
  의심할 것.
- **시작하자마자 그리퍼가 갑자기 열림** → GELLO 연결(ENTER) 전에 그리퍼
  워커 스레드가 기본값(0%)을 그대로 내보내던 버그, 이미 수정됨 (연결 전에는
  그리퍼에 어떤 명령도 안 나감).
