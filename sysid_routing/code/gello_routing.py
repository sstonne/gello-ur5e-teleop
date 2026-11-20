"""GELLO drives the simulated UR5e in the RealTable rope scene, joint for joint.

GELLO is a scaled copy of the UR arm, so its joint angles go to the simulator
unchanged: the UR5e environment accepts [six radians, closure] directly and
limits the joint speed itself. The scene is make_env's: identified rope, floor
kernels and clips.

Two mappings:
  official            the official gello_software mapping, via
                      ur5e_test/control/gello_official.py: absolute joint angles
                      from the calibration in gello_config.json
                      (gello_calibrate.py). SPACE refuses to start while any
                      joint is more than 0.8 rad from the simulated arm, as
                      run_env.py does; otherwise the arm approaches the GELLO
                      pose at 0.05 rad per step for 25 steps, then follows.
  relative (default)  the earlier latch: GELLO and the arm are taken to agree at
                      the SPACE press, and only changes are followed.

Speed. Each step used to cost about 80 ms, and half of it was not physics: the
Gaussian appearance of the rope and the arm links was redeformed after every
step, drawn or not. Here that update runs only on frames that are drawn, from
the particle positions at the last drawn frame (rope_routing_env.py does the
same). Physics state is not touched by it on this joint-command path.
--substep-scale below 1 buys more speed by integrating more coarsely; use it to
practise, not for anything that will be replayed or identified.

    SPACE follow/pause   O/C record gripper open/closed (paused, relative only)
    0 reset the scene    ESC quit
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_realtable as C  # noqa: E402
sys.path.insert(0, str(C.WORKSPACE / 'ur5e_test' / 'control'))
import gello_official as G  # noqa: E402


class DeferredAppearance:
    """Skip the renderer's appearance update until a frame is actually drawn.

    The LBS deformation moves the splats by the particle motion since the last
    appearance update, so the particles at the last drawn frame are kept as the
    reference and restored for the one update made before drawing.
    """
    def __init__(self, renderer):
        self.renderer = renderer
        self.full = renderer.update_rendervar
        self.reference = renderer.state['x'].clone()
        renderer.update_rendervar = lambda *args, **kwargs: None

    def reset(self):
        self.renderer.relations = self.renderer.weights = None
        self.reference = self.renderer.state['x'].clone()

    def flush(self):
        import torch
        r = self.renderer
        current = r.state['x']
        # the commanded joints, so the arm's splats need no inverse kinematics
        q = torch.tensor(np.r_[r.qpos_curr_xarm, 0.], dtype=torch.float32, device=current.device)[None]
        try:
            r.state['x'] = self.reference
            self.full(current.clone(), qpos_now=q)
        finally:
            r.state['x'] = current
        self.reference = current.clone()


def fast_link_transform(robot):
    """Same result as UR5eRobotiq.transform_scene, without its per-frame overhead.

    The shipped version rebuilds each link's boolean mask and converts every
    splat's rotation to a matrix and back on each call (about 27 ms a frame).
    Link membership is fixed, so the indices are kept, and each link's rotation
    is composed onto the splats' quaternions directly.
    """
    import torch
    from scipy.spatial.transform import Rotation
    cache = {}

    def transform(params, labels, q, openness):
        key = (labels.data_ptr(), labels.shape[0])
        if key not in cache:
            cache.clear()
            cache[key] = [torch.nonzero(labels == i).squeeze(1) for i in range(len(robot.names))]
        fk = robot.model.fk(np.degrees(q), float(np.clip(openness, 0, 1)) * 85)
        xyz, quat = params['means3D'], params['rotations']
        for ids, name in zip(cache[key], robot.names):
            if ids.numel() == 0:
                continue
            M = fk[name] @ np.linalg.inv(robot.scene_reference[name])
            T = torch.tensor(M, device=labels.device, dtype=torch.float32)
            xyz[ids] = xyz[ids] @ T[:3, :3].T + T[:3, 3]
            # one 3x3 matrix: scipy on the CPU, not kornia's batched GPU path
            w1, v1 = torch.tensor(Rotation.from_matrix(M[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                  device=labels.device, dtype=torch.float32).split([1, 3])
            p = quat[ids]
            w2, v2 = p[:, :1], p[:, 1:]
            quat[ids] = torch.cat([w1 * w2 - v2 @ v1[:, None],
                                   w1 * v2 + w2 * v1 + torch.cross(v1.expand_as(v2), v2, dim=1)], dim=1)
        return params
    return transform


def settle(env, seed, steps=30):
    """Reset, then hold the arm while the rope settles, as make_env does."""
    import kornia
    import torch
    robot = env.reset(seed=seed)[0]['robot']
    rotation = kornia.geometry.conversions.quaternion_to_rotation_matrix(robot['eef_quat'])
    hold = torch.cat([robot['eef_xyz'], rotation.reshape(1, 9), robot['eef_gripper']], dim=1)
    hold = hold.detach().cpu().numpy().copy()
    for _ in range(steps):
        C.step(env, hold)


def view(obs, which):
    import cv2
    frames = [obs['image_list'][i] for i in which if isinstance(i, int) and i < len(obs['image_list'])]
    if 'wrist' in which:
        frames.append(obs['image_wrist_list'][0])
    ims = [(f[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for f in frames]
    return cv2.cvtColor(np.hstack(ims), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', default='routing_measured.yaml',
                    help='file in sysid_routing/scene, or a path')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--baudrate', type=int, default=57600)
    # set by watching this GELLO drive the simulated arm; differs from the
    # real-arm signs in gello_ur5e_real_teleop.py (1,-1,1,-1,1,1)
    ap.add_argument('--mapping', choices=('official', 'relative'), default='relative')
    ap.add_argument('--gello-config', type=Path, default=G.CONFIG,
                    help='official mapping: calibration written by gello_calibrate.py')
    ap.add_argument('--joint-sign', default=None,
                    help='official: overrides the calibrated signs (offsets recomputed from the '
                         'calibration pose); relative: default 1,1,-1,1,1,1')
    ap.add_argument('--grip-sign', type=float, default=-1)
    ap.add_argument('--grip-ticks-to-full', type=float, default=570)
    ap.add_argument('--filter-alpha', type=float, default=.75)
    ap.add_argument('--max-joint-speed', type=float, default=None,
                    help='rad/s cap in the simulator (scene default when omitted)')
    ap.add_argument('--substep-scale', type=float, default=1.,
                    help='below 1 integrates more coarsely and runs faster; practice only')
    ap.add_argument('--render-every', type=int, default=2,
                    help='draw every Nth step; the steps between skip all appearance work')
    ap.add_argument('--views', default='0,wrist',
                    help='comma list of fixed-camera indices and/or "wrist"')
    ap.add_argument('--fake', action='store_true',
                    help='no GELLO: small sinusoidal joint motion, for testing the loop')
    ap.add_argument('--steps', type=int, default=0, help='stop after N steps (0: run until ESC)')
    ap.add_argument('--headless', action='store_true', help='no window; print the loop rate')
    ap.add_argument('--snapshot', type=Path, default=None, help='write the last drawn frame here')
    ap.add_argument('--full-appearance', action='store_true',
                    help='update the appearance every step as the simulator ships (for comparison)')
    a = ap.parse_args()
    if a.render_every < 1:
        ap.error('--render-every must be at least 1')
    which = [int(v) if v.isdigit() else v for v in a.views.split(',')]

    sys.path.insert(0, str(C.REPO))
    from experiments.gello_ur5e import JointMapper, Reader, tick_delta
    official = a.mapping == 'official' and not a.fake
    mapper = JointMapper([float(v) for v in (a.joint_sign or '1,1,-1,1,1,1').split(',')], a.grip_sign,
                         a.grip_ticks_to_full, a.filter_alpha)
    gello = None
    if official:
        cfg_g = G.GelloConfig.load(a.gello_config)
        if a.joint_sign:
            cfg_g = cfg_g.with_signs([int(float(v)) for v in a.joint_sign.split(',')])
        gello = G.Mapping(cfg_g)
        print(f'official mapping: signs {cfg_g.joint_signs}, offsets '
              f'{np.round(cfg_g.joint_offsets, 3).tolist()} (calibrated {cfg_g.calibrated_at})', flush=True)
    reader = None
    if not a.fake:
        reader = (G.Reader if official else Reader)(a.port, a.baudrate)
        deadline = time.monotonic() + 3
        while reader.get()[0] is None and time.monotonic() < deadline:
            time.sleep(.02)
        ticks, _, error = reader.get()
        if ticks is None:
            reader.close()
            raise SystemExit(f'GELLO: {error}')
        print('GELLO IDs 1-7 connected:', ticks.tolist(), flush=True)
    approach_left = 0

    import cv2
    import torch
    scene = Path(a.scene)
    if not scene.is_absolute() and not scene.exists():
        scene = C.HERE.parent / 'scene' / a.scene
    env, _, cfg = C.make_env(seed=a.seed, config=scene.resolve(), substep_scale=a.substep_scale)
    sim = env.unwrapped
    r = sim.renderer
    if a.max_joint_speed is not None:
        cfg.env.robot.max_joint_speed_rad_s = a.max_joint_speed
        sim.cfg.env.robot.max_joint_speed_rad_s = a.max_joint_speed
    appearance = None if a.full_appearance else DeferredAppearance(r)
    if not a.full_appearance:
        r.robot.transform_scene = fast_link_transform(r.robot)
    fps = float(cfg.physics.fps)
    calibration = C.HERE.parent / 'scene' / 'gello_gripper.json'
    if calibration.exists():
        saved = json.loads(calibration.read_text())
        if saved.get('port') == a.port:
            mapper.endpoints = saved['endpoints']

    window = 'GELLO - RealTable rope'
    if not a.headless:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    active = bool(a.fake)
    status = 'PAUSED: match GELLO to the arm on screen, SPACE to follow'
    q_home = r.qpos_curr_xarm.copy()
    if a.fake:
        mapper.latch(np.zeros(7), q_home, 1 - float(r.get_state()['eef_gripper'][0, 0]))
    image, step, hz, t_run = None, 0, fps, time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            state = r.get_state()
            q = r.qpos_curr_xarm.copy()
            closure = 1 - float(state['eef_gripper'][0, 0])
            fresh = True
            if a.fake:
                q = q_home + .15 * np.sin(step / 20) * np.array([1, .5, -.5, .5, 1, 1])
            else:
                ticks, stamp, error = reader.get()
                fresh = ticks is not None and t0 - stamp < .25 and not error
                if active and not fresh:
                    active, status = False, 'INPUT LOST - paused; SPACE to reconnect'
                elif active:
                    try:
                        if official:
                            q, closure = gello(ticks)
                            if approach_left > 0:       # run_env.py: 25 steps of <= 0.05 rad
                                q = G.approach(r.qpos_curr_xarm, q)
                                approach_left -= 1
                        else:
                            q, closure = mapper.update(ticks, continuous=True)
                        lim = r.robot.limits
                        if np.any(q < lim[:, 0]) or np.any(q > lim[:, 1]):
                            raise ValueError('Joint limit reached')
                    except ValueError as exc:
                        active, status = False, f'{exc} - SPACE to reconnect'
                        q, closure = r.qpos_curr_xarm.copy(), 1 - float(state['eef_gripper'][0, 0])
            action = torch.as_tensor(np.r_[q, closure], dtype=torch.float32, device=state['x'].device)[None]
            sim.step({'action': action, 'do_velocity_control': True})

            if step % a.render_every == 0 or image is None:
                if appearance is not None:
                    appearance.flush()
                image = view(sim.get_obs(), which)
            step += 1
            if not a.headless:
                opening = float(r.get_state()['eef_gripper'][0, 0]) * 85
                text = ('FOLLOWING' if active else status) + f' | gripper {opening:.0f} mm | {hz:.1f}/{fps:.0f} Hz'
                if a.substep_scale != 1:
                    text += f' | substeps x{a.substep_scale:g} (practice)'
                bar = np.full((76 if official else 52, image.shape[1], 3), 20, np.uint8)
                cv2.putText(bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (80, 230, 100) if active else (0, 210, 255), 1, cv2.LINE_AA)
                cv2.putText(bar, 'SPACE follow/pause | O/C gripper open/closed (paused) | 0 reset | ESC quit',
                            (10, 42), cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1, cv2.LINE_AA)
                if official and fresh and not active:
                    # where GELLO is against the arm, so the two can be brought together
                    lead = gello.raw_joints(ticks)
                    lead = lead + 2 * np.pi * np.round((r.qpos_curr_xarm - lead) / (2 * np.pi))
                    diff = np.degrees(lead - r.qpos_curr_xarm)
                    cv2.putText(bar, 'GELLO - arm (deg): ' + '  '.join(f'J{i + 1} {d:+.0f}' for i, d in enumerate(diff))
                                + f'   | start needs all within {np.degrees(G.MAX_START_DELTA):.0f}',
                                (10, 66), cv2.FONT_HERSHEY_SIMPLEX, .45,
                                (80, 230, 100) if np.all(np.abs(diff) <= np.degrees(G.MAX_START_DELTA)) else (0, 140, 255),
                                1, cv2.LINE_AA)
                cv2.imshow(window, np.vstack([bar, image]))
                key = cv2.waitKey(1) & 255
                if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == 32 and not a.fake:
                    active = fresh and not active
                    if active and official:
                        gello.wrap_to(ticks, r.qpos_curr_xarm)
                        bad = G.start_check(gello(ticks)[0], r.qpos_curr_xarm)
                        if bad:                         # run_env.py refuses to start
                            active = False
                            status = 'Too far from the arm - ' + ', '.join(
                                f'J{i} {np.degrees(d):.0f} deg' for i, d, _, _ in bad)
                            print('start refused (more than 0.8 rad apart):\n' + G.format_start_check(bad), flush=True)
                        else:
                            approach_left = G.APPROACH_STEPS
                            status = 'FOLLOWING'
                    elif active:
                        s = r.get_state()
                        mapper.latch(ticks, r.qpos_curr_xarm, 1 - float(s['eef_gripper'][0, 0]))
                        status = 'FOLLOWING'
                    else:
                        status = 'PAUSED: SPACE to follow' if fresh else 'No fresh input; check USB'
                elif key == ord('0'):
                    active = False
                    settle(env, a.seed)
                    if appearance is not None:
                        appearance.reset()
                    image, status = None, 'Scene reset; match GELLO, SPACE to follow'
                elif key in (ord('o'), ord('c')) and not active and fresh and not a.fake and not official:
                    mapper.endpoints['open' if key == ord('o') else 'close'] = int(ticks[6])
                    if len(mapper.endpoints) == 2:
                        span = abs(float(tick_delta(mapper.endpoints['close'], mapper.endpoints['open'])))
                        if span < 50:
                            mapper.endpoints.clear()
                            status = 'Calibration too small; record O and C again'
                        else:
                            calibration.write_text(json.dumps(dict(port=a.port, endpoints=mapper.endpoints), indent=2) + '\n')
                            status = 'Gripper calibrated; SPACE to follow'
                    else:
                        status = 'Endpoint saved; record the other with O/C'
            if a.steps and step >= a.steps:
                break
            time.sleep(max(0., 1 / fps - (time.monotonic() - t0)))
            hz = .9 * hz + .1 / (time.monotonic() - t0)
        if a.snapshot is not None:
            cv2.imwrite(str(a.snapshot), image)
        print(f'{step} steps, {step / (time.monotonic() - t_run):.1f} Hz average '
              f'(simulated time runs at {fps:.0f} Hz)', flush=True)
    finally:
        if reader is not None:
            reader.close()
        if not a.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
