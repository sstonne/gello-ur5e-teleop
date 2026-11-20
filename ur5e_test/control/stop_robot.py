#!/usr/bin/env python3
"""RTDE로 실행 중인 UR5e 직선 제어를 감속 정지한다."""

import argparse

import rtde_control

DEFAULT_ROBOT_IP = "172.28.60.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--decel", type=float, default=2.0,
                        help="stopL 감속도 (m/s^2, 기본 2.0)")
    args = parser.parse_args()

    print(f"Connecting to UR5e at {args.robot_ip}...")
    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    try:
        print("Sending stopL and stopScript...")
        rtde_c.stopL(args.decel)
        rtde_c.stopScript()
        print("Stop command sent.")
    finally:
        rtde_c.disconnect()


if __name__ == "__main__":
    main()
