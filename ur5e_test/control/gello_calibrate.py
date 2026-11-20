#!/usr/bin/env python3
"""GELLO 보정 - 공식 gello_software의 scripts/gello_get_offset.py와 같은 계산.

1. GELLO를 기준 자세에 놓는다. UR 기본 기준 자세(공식 README)는
       0, -90, 90, -90, -90, 0 도   (= 0 -1.57 1.57 -1.57 -1.57 0 rad)
   그리퍼는 벌린 상태로 둔다.
2. Enter. 관절마다 오프셋(pi/2의 배수)을 공식 방식으로 찾고, 그리퍼 열림/닫힘
   각도도 공식 식(열림 = 지금 - 0.2도, 닫힘 = 지금 - 42도)으로 정한다.
3. gello_config.json에 저장하고, 실시간으로 관절 각도를 보여준다. 관절을 하나씩
   움직여 보며 방향이 맞는지 확인한다(시뮬레이터에서 보는 게 가장 안전하다).

부호를 바꿀 때는 다시 자세를 잡을 필요 없이:
    python gello_calibrate.py --resign 1,1,-1,1,1,1
(보정 자세에서 읽은 원시값이 저장돼 있어서 오프셋만 다시 계산한다.)

    python gello_calibrate.py                      # 보정 + 확인
    python gello_calibrate.py --watch              # 저장된 보정으로 확인만
    python gello_calibrate.py --joint-signs 1,1,-1,1,1,1 --start-joints-deg 0,-90,90,-90,-90,0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gello_official as G  # noqa: E402


def floats(text, n):
    values = [float(v) for v in text.replace(' ', '').split(',')]
    if len(values) != n:
        raise argparse.ArgumentTypeError(f'값 {n}개가 필요합니다: {text}')
    return values


def watch(reader, cfg):
    mapping = G.Mapping(cfg)
    print('\n관절을 하나씩 움직여 보세요. 기준 자세에서의 변화(도)와 현재 각도(도)입니다. Ctrl+C로 끝.')
    print('            ' + ''.join(f'  J{i}     ' for i in range(1, 7)) + ' gripper')
    try:
        while True:
            ticks, _, error = reader.get()
            if ticks is None or error:
                print(f'\r  [{error}]' + ' ' * 60, end='', flush=True)
            else:
                q, g = mapping(ticks)
                deg = np.degrees(q)
                moved = deg - np.degrees(cfg.start_joints)
                print('\r  angle ' + ''.join(f'{v:8.1f} ' for v in deg)
                      + f'  {g:4.2f}   moved ' + ''.join(f'{v:+6.1f}' for v in moved),
                      end='', flush=True)
            time.sleep(.1)
    except KeyboardInterrupt:
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--baudrate', type=int, default=57600)
    ap.add_argument('--joint-signs', default=','.join(str(s) for s in G.UR_JOINT_SIGNS),
                    help='공식 UR 기본값 1,1,-1,1,1,1')
    ap.add_argument('--start-joints-deg', default='0,-90,90,-90,-90,0',
                    help='GELLO를 놓은 기준 자세(도). 공식 UR 예시 = 0,-90,90,-90,-90,0')
    ap.add_argument('--config', type=Path, default=G.CONFIG)
    ap.add_argument('--watch', action='store_true', help='저장된 보정으로 실시간 확인만')
    ap.add_argument('--resign', default=None,
                    help='저장된 보정 자세 그대로, 부호만 바꿔 오프셋 다시 계산 (GELLO 연결 불필요)')
    a = ap.parse_args()

    if a.resign:
        cfg = G.GelloConfig.load(a.config).with_signs([int(v) for v in floats(a.resign, 6)])
        cfg.save(a.config)
        print(f'부호 {cfg.joint_signs}로 오프셋 다시 계산: {np.round(cfg.joint_offsets, 3).tolist()}')
        print(f'저장: {a.config}')
        return

    reader = G.Reader(a.port, a.baudrate)
    try:
        reader.wait()
        if a.watch:
            watch(reader, G.GelloConfig.load(a.config))
            return
        signs = [int(v) for v in floats(a.joint_signs, 6)]
        start = np.radians(floats(a.start_joints_deg, 6))
        print(f'\n기준 자세(도): {np.round(np.degrees(start), 1).tolist()}   부호: {signs}')
        input('GELLO를 이 자세로 놓고, 그리퍼는 벌린 채로 Enter... ')
        samples = []
        for _ in range(20):                       # upstream warms up 10 reads; average 20
            ticks, _, error = reader.get()
            if ticks is not None and not error:
                samples.append(ticks)
            time.sleep(.02)
        if len(samples) < 5:
            raise SystemExit(f'GELLO 값을 읽지 못했습니다: {reader.get()[2]}')
        raw = np.round(np.mean(samples, axis=0)).astype(int)
        cfg = G.GelloConfig.calibrate(raw, start, signs, port=a.port)

        print('\nbest offsets               : ', [f'{x:.3f}' for x in cfg.joint_offsets])
        print('best offsets function of pi: [' + ', '.join(
            f'{int(np.round(x / (np.pi / 2)))}*np.pi/2' for x in cfg.joint_offsets) + ' ]')
        print(f'gripper open (degrees)       {cfg.gripper_open_deg:.2f}')
        print(f'gripper close (degrees)      {cfg.gripper_close_deg:.2f}')
        err = np.degrees(cfg.calib_errors_rad)
        print('기준 자세와의 남은 차이(도): ' + ', '.join(f'J{i + 1} {e:.1f}' for i, e in enumerate(err)))
        if np.any(err > 20):
            print('  주의: 20도 넘게 남은 관절이 있습니다. 자세가 기준과 다르거나, 그 관절의 '
                  '모터 혼(horn)이 90도 단위로 조립되지 않았을 수 있습니다.')
        cfg.save(a.config)
        print(f'저장: {a.config}')
        watch(reader, cfg)
    finally:
        reader.close()


if __name__ == '__main__':
    main()
