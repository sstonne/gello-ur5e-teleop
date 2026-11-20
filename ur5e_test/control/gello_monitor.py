#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GELLO 모터만 읽어서 보여준다. 로봇에는 연결하지 않는다.

새 GELLO를 처음 붙였을 때 이것부터 돌린다. 관절을 하나씩 움직이며
어느 모터 ID가 반응하는지, 어느 방향으로 도는지 눈으로 확인한 뒤
--joint-sign을 정한다.

로봇을 전혀 건드리지 않으므로 안전하다.

    python ur5e_test/control/gello_monitor.py
    python ur5e_test/control/gello_monitor.py --port /dev/ttyUSB1 --baudrate 1000000
"""

from __future__ import annotations

import argparse
import time

import dynamixel_sdk as dxl
import numpy as np

ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
TICKS_PER_REV = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_REV


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--ids", nargs="+", type=int,
                        help="읽을 모터 ID들. 생략하면 0~20을 훑어 찾는다")
    parser.add_argument("--hz", type=float, default=10.0)
    return parser.parse_args()


def wrapped(delta):
    return ((delta + TICKS_PER_REV // 2) % TICKS_PER_REV) - TICKS_PER_REV // 2


def scan(port_handler, packet_handler):
    found = []
    for dxl_id in range(0, 21):
        model, comm, _ = packet_handler.ping(port_handler, dxl_id)
        if comm == dxl.COMM_SUCCESS:
            found.append((dxl_id, model))
    return found


def main() -> int:
    args = parse_args()
    port_handler = dxl.PortHandler(args.port)
    packet_handler = dxl.PacketHandler(2.0)
    if not port_handler.openPort():
        raise SystemExit(f"포트를 열 수 없습니다: {args.port}")
    if not port_handler.setBaudRate(args.baudrate):
        raise SystemExit(f"baudrate 설정 실패: {args.baudrate}")

    if args.ids:
        ids = list(args.ids)
        print(f"지정된 ID: {ids}")
    else:
        print(f"{args.port} @ {args.baudrate} 에서 모터를 찾는 중...")
        found = scan(port_handler, packet_handler)
        if not found:
            port_handler.closePort()
            raise SystemExit(
                "모터를 찾지 못했습니다.\n"
                "  - baudrate가 다를 수 있습니다 (--baudrate 1000000 등)\n"
                "  - USB 연결이 불안정하면 읽기가 실패합니다\n"
                "  - GELLO 모터 전원이 들어와 있는지 확인하세요")
        ids = [i for i, _ in found]
        print("찾은 모터:")
        for dxl_id, model in found:
            print(f"  ID {dxl_id:2d}  model {model}")

    # 손으로 움직여야 하므로 토크를 끈다.
    for dxl_id in ids:
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)

    group = dxl.GroupSyncRead(port_handler, packet_handler,
                              ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for dxl_id in ids:
        group.addParam(dxl_id)

    print("\n관절을 하나씩 천천히 움직여 보세요. 어느 ID가 반응하는지 보입니다.")
    print("기준을 지금 자세로 다시 잡으려면 Enter, 끝내려면 Ctrl+C.\n")
    header = "  ".join(f"ID{d:<2d}" for d in ids)
    print(f"{'':6s}{header}")

    reference = None
    period = 1.0 / max(1.0, args.hz)
    try:
        while True:
            start = time.monotonic()
            if group.txRxPacket() != dxl.COMM_SUCCESS:
                print("읽기 실패 (USB 연결 확인)", end="\r")
                time.sleep(period)
                continue
            positions = {}
            for dxl_id in ids:
                if group.isAvailable(dxl_id, ADDR_PRESENT_POSITION,
                                     LEN_PRESENT_POSITION):
                    positions[dxl_id] = group.getData(
                        dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if len(positions) < len(ids):
                missing = sorted(set(ids) - set(positions))
                print(f"일부 모터 응답 없음: {missing}   ", end="\r")
                time.sleep(period)
                continue
            if reference is None:
                reference = dict(positions)
            cells = []
            for dxl_id in ids:
                deg = wrapped(positions[dxl_id] - reference[dxl_id]) * DEG_PER_TICK
                mark = "*" if abs(deg) > 3.0 else " "
                cells.append(f"{deg:+7.1f}{mark}")
            print("기준대비 " + " ".join(cells) + "   ", end="\r", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - start)))
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        port_handler.closePort()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
