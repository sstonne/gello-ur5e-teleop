#!/usr/bin/env python3
"""GELLO 선 연결 확인 - 1초마다 USB(U2D2)와 모터 ID 1~7 응답을 출력한다.

케이블을 만지면서 실행해 두면 언제 연결되는지 바로 보인다. 읽기(ping)만 하고
모터 설정은 건드리지 않는다. Ctrl+C로 끝.

    python ur5e_test/control/gello_wire_check.py
    python ur5e_test/control/gello_wire_check.py --port /dev/ttyUSB1
"""
import argparse
import os
import time

import dynamixel_sdk as dxl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--baudrate', type=int, default=57600)
    a = ap.parse_args()
    try:
        while True:
            stamp = time.strftime('%X')
            if not os.path.exists(a.port):
                print(stamp, f'USB 없음 - {a.port} 안 잡힘 (USB 케이블 / U2D2)', flush=True)
                time.sleep(1)
                continue
            port, packet = dxl.PortHandler(a.port), dxl.PacketHandler(2.0)
            try:
                if not (port.openPort() and port.setBaudRate(a.baudrate)):
                    print(stamp, f'{a.port} 열기 실패 (권한: dialout 그룹?)', flush=True)
                else:
                    ids = [i for i in range(1, 8) if packet.ping(port, i)[1] == dxl.COMM_SUCCESS]
                    missing = [i for i in range(1, 8) if i not in ids]
                    if not ids:
                        state = '모터 응답 없음 (모터 5V 전원 / U2D2-Power Hub 3핀 케이블)'
                    elif missing:
                        state = f'응답 {ids}, 없음 {missing} (마지막 응답 ID 다음 케이블 확인)'
                    else:
                        state = '모터 1~7 전부 OK'
                    print(stamp, 'USB OK |', state, flush=True)
            except Exception as exc:  # noqa: BLE001
                print(stamp, f'USB 끊김: {exc}', flush=True)
            finally:
                port.closePort()
            time.sleep(1)
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
