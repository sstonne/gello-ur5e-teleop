"""GELLO joint-space input for the saved Desk UR5e + Robotiq simulator.

Dynamixel IDs, baud and signs follow ur3_test/control/gello_udp_publisher.py.
No physical follower robot connection is created.
"""
from pathlib import Path
import argparse
import json
import sys
import threading
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tick_delta(now, before):
    return (np.asarray(now, dtype=float)-before+2048) % 4096-2048


class EncoderStream:
    """Validate and unwrap at serial-read cadence, not slow rendering cadence."""
    def __init__(self):
        self.raw = None
        self.position = None
        self.stamp = None

    def feed(self, raw, stamp):
        raw = np.asarray(raw,dtype=float)
        if self.raw is None:
            self.raw, self.position, self.stamp = raw.copy(), raw.copy(), stamp
            return self.position.copy()
        elapsed = stamp-self.stamp
        change = tick_delta(raw,self.raw)
        limit = min(2047.,400*max(1.,elapsed/(1/30)))
        if np.any(np.abs(change)>limit):
            axis = int(np.argmax(np.abs(change)))+1
            raise ValueError(f'Encoder spike ID {axis}: {abs(change[axis-1]):.0f} ticks / {elapsed:.3f}s')
        self.position += change
        self.raw, self.stamp = raw.copy(), stamp
        return self.position.copy()


class JointMapper:
    def __init__(self, signs, grip_sign=-1., grip_span=570., filter_alpha=.75):
        self.signs = np.asarray(signs, dtype=float)
        if self.signs.shape != (6,) or not np.isin(self.signs, [-1, 1]).all():
            raise ValueError('joint-sign requires six +1/-1 values')
        if grip_span <= 0 or grip_sign not in (-1, 1):
            raise ValueError('Invalid gripper sign/span')
        self.grip_sign, self.grip_span = grip_sign, grip_span
        if not 0 < filter_alpha <= 1:
            raise ValueError("filter-alpha must be in (0, 1]")
        self.filter_alpha = filter_alpha
        self.endpoints = {}

    def latch(self, ticks, q, closure):
        self.previous = np.array(ticks, dtype=float)
        self.delta = np.zeros(7)
        self.filtered = np.zeros(7)
        self.q_ref = np.array(q, dtype=float)
        self.closure_ref = float(closure)

    def update(self, ticks, continuous=False):
        ticks = np.array(ticks, dtype=float)
        change = ticks-self.previous if continuous else tick_delta(ticks, self.previous)
        if not continuous and np.any(np.abs(change) > 400):
            raise ValueError('Encoder jump; pause and reconnect')
        self.previous = ticks
        self.delta += change  # Incremental unwrapping, including >180 degree motion.
        self.filtered += self.filter_alpha*(self.delta-self.filtered)
        q = self.q_ref+self.filtered[:6]*(2*np.pi/4096)*self.signs
        closure = self.closure_ref+self.filtered[6]*self.grip_sign/self.grip_span
        if 'open' in self.endpoints and 'close' in self.endpoints:
            span = float(tick_delta(self.endpoints['close'], self.endpoints['open']))
            if abs(span) < 50:
                raise ValueError('Gripper calibration endpoints are too close')
            closure = float(tick_delta(ticks[6], self.endpoints['open']))/span
        return q, float(np.clip(closure, 0, 1))


class Reader:
    def __init__(self, port, baudrate):
        import dynamixel_sdk as dxl
        self.dxl = dxl
        self.port = dxl.PortHandler(port)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.ticks = None
        self.stamp = 0.
        self.error = 'Waiting for IDs 1-7'
        self.encoder_stream = EncoderStream()
        if not self.port.openPort():
            raise RuntimeError(f'Cannot open {port}')
        try:
            if not self.port.setBaudRate(baudrate):
                raise RuntimeError('Cannot set baud rate')
            packet = dxl.PacketHandler(2.)
            for motor in range(1,8):
                for attempt in range(3):
                    self.port.clearPort()
                    _, result, error = packet.ping(self.port, motor)
                    if result == dxl.COMM_SUCCESS and not error:
                        break
                    time.sleep(.05)
                if result != dxl.COMM_SUCCESS or error:
                    raise RuntimeError(f'GELLO ID {motor} did not respond: {result}, {error}')
            # Leader joints must be movable by hand; never enable torque.
            for motor in range(1,8):
                result,error = packet.write1ByteTxRx(self.port,motor,64,0)
                if result != dxl.COMM_SUCCESS or error:
                    raise RuntimeError(f'Cannot disable GELLO torque: ID {motor}')
            self.group = dxl.GroupSyncRead(self.port,packet,132,4)
            for motor in range(1,8):
                if not self.group.addParam(motor):
                    raise RuntimeError(f'Cannot register ID {motor}')
        except Exception:
            self.port.closePort()
            raise
        self.thread = threading.Thread(target=self.loop,daemon=True)
        self.thread.start()

    def loop(self):
        try:
            while not self.stop.is_set():
                result = self.group.txRxPacket()
                if result == self.dxl.COMM_SUCCESS and all(self.group.isAvailable(i,132,4) for i in range(1,8)):
                    values = np.array([self.group.getData(i,132,4) % 4096 for i in range(1,8)])
                    stamp = time.monotonic()
                    try:
                        values = self.encoder_stream.feed(values,stamp)
                    except ValueError as exc:
                        with self.lock:
                            self.error = str(exc)
                        self.stop.wait(1/60)
                        continue
                    with self.lock:
                        self.ticks, self.stamp, self.error = values, stamp, ''
                else:
                    with self.lock:
                        self.error = 'Incomplete GELLO packet'
                self.stop.wait(1/60)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            self.port.closePort()

    def get(self):
        with self.lock:
            return None if self.ticks is None else self.ticks.copy(), self.stamp, self.error

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--project',default=str(root/'ur5e_real2sim/projects/Desk/original_sim/simulator.yaml'))
    parser.add_argument('--port',default='/dev/ttyUSB0')
    parser.add_argument('--baudrate',type=int,default=57600)
    parser.add_argument('--joint-sign',default='1,1,-1,1,1,1')
    parser.add_argument('--grip-sign',type=float,default=-1)
    parser.add_argument('--grip-ticks-to-full',type=float,default=570)
    parser.add_argument('--max-joint-speed',type=float,default=2.,help='Simulation joint speed cap in rad/s; midpoint between original 1 and fast 3')
    parser.add_argument('--filter-alpha',type=float,default=.75,help='Input response: midpoint between original .6 and fast .9')
    parser.add_argument('--physics-dt',type=float,default=5e-5,help='Physics substep seconds; keep original stable default')
    parser.add_argument('--camera-scale',type=float,default=.5,help='Preview resolution factor; physics and assets are unchanged')
    parser.add_argument('--render-every',type=int,default=2,help='Render every N physics frames to reduce teleop latency')
    parser.add_argument('--probe',action='store_true',help='Read seven encoders and exit without launching simulator')
    parser.add_argument('--smoke-test',action='store_true',help='Read real encoders, exercise small simulated joint targets and partial gripper openings, then exit')
    args = parser.parse_args()
    mapper = JointMapper([float(x) for x in args.joint_sign.split(',')],args.grip_sign,args.grip_ticks_to_full,args.filter_alpha)
    if not np.isfinite(args.max_joint_speed) or args.max_joint_speed <= 0 or args.render_every < 1 or not 0 < args.camera_scale <= 1 or not 0 < args.physics_dt <= 5e-5:
        parser.error('Require positive speed, render-every >= 1 and 0 < physics-dt <= 5e-5')
    reader = Reader(args.port,args.baudrate)
    try:
        deadline = time.monotonic()+3
        while reader.get()[0] is None and time.monotonic()<deadline:
            time.sleep(.02)
        ticks, stamp, error = reader.get()
        if ticks is None:
            raise RuntimeError(error)
        print('GELLO IDs 1-7 connected:',ticks.tolist(),flush=True)
        if args.probe:
            return
        import cv2
        import torch
        from omegaconf import OmegaConf
        from sim.envs.env import BaseEnv
        OmegaConf.register_new_resolver('eval',eval,replace=True)
        cfg = OmegaConf.load(Path(args.project).resolve())
        if cfg.env.robot.type != 'ur5e':
            raise ValueError('This entry point requires UR5e')
        cfg.env.robot.max_joint_speed_rad_s = args.max_joint_speed
        cfg.physics.dt = args.physics_dt
        for camera in cfg.env.cameras:
            old_w, old_h = int(camera.w), int(camera.h)
            camera.w, camera.h = max(64,round(old_w*args.camera_scale)), max(64,round(old_h*args.camera_scale))
            intr = np.asarray(camera.intr,dtype=float).reshape(3,3)
            intr[0] *= camera.w/old_w
            intr[1] *= camera.h/old_h
            camera.intr = intr.reshape(-1).tolist()
        env = BaseEnv(cfg,cfg.exp_root)
        env.reset(seed=0)
        active = False
        status = 'PAUSED: align GELLO; SPACE to follow'
        window = 'GELLO - UR5e + Robotiq 2F-85'
        cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        calibration = Path(args.project).resolve().with_name('gello_gripper.json')
        if calibration.exists():
            saved = json.loads(calibration.read_text())
            if saved.get('port') == args.port:
                mapper.endpoints = saved['endpoints']
        frames = 0
        last_xyz = env.renderer.get_state()['eef_xyz'][0].cpu().numpy().copy()
        last_vel = np.zeros(3)
        measured_hz = float(cfg.physics.fps)
        run_start = time.monotonic()
        peak_speed = peak_accel = 0.
        image = None
        while True:
            start = time.monotonic()
            ticks, stamp, error = reader.get()
            fresh = ticks is not None and start-stamp < .25 and not error
            state = env.renderer.get_state()
            q = env.renderer.qpos_curr_xarm.copy()
            closure = 1-float(state['eef_gripper'][0,0])
            if active:
                if not fresh:
                    active = False
                    status = 'INPUT LOST - paused; SPACE to reconnect'
                else:
                    try:
                        q,closure = mapper.update(ticks,continuous=True)
                        if np.any(q < env.renderer.robot.limits[:,0]) or np.any(q > env.renderer.robot.limits[:,1]):
                            raise ValueError('Joint limit reached')
                    except ValueError as exc:
                        active = False
                        status = str(exc)+' - SPACE to reconnect'
                        q = env.renderer.qpos_curr_xarm.copy()
                        closure = 1-float(state['eef_gripper'][0,0])
            if args.smoke_test:
                q = env.renderer.robot.initial_q + .02*np.sin(frames/10)*np.ones(6)
                closure = [.25,.5,.75][frames//10]
            action = torch.as_tensor(np.r_[q,closure],dtype=torch.float32,device=state['x'].device)[None]
            env.step({'action':action,'do_velocity_control':True})
            assert torch.isfinite(env.renderer.get_state()['x']).all(), 'Nonfinite simulation state'
            xyz = env.renderer.get_state()['eef_xyz'][0].cpu().numpy()
            velocity = (xyz-last_xyz)*cfg.physics.fps
            acceleration = (velocity-last_vel)*cfg.physics.fps
            speed, accel = float(np.linalg.norm(velocity)), float(np.linalg.norm(acceleration))
            peak_speed, peak_accel = max(peak_speed,speed), max(peak_accel,accel)
            last_xyz, last_vel = xyz.copy(), velocity.copy()
            if image is None or frames % args.render_every == 0:
                obs = env.get_obs()
                images = obs['image_list'][:1]+obs['image_wrist_list'][:1]
                image = np.concatenate([(im[:3].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype('uint8') for im in images],axis=1)
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            opening = float(env.renderer.get_state()['eef_gripper'][0,0])*85
            if args.smoke_test and frames%10 == 9:
                np.testing.assert_allclose(opening,85*(1-closure),atol=.1)
                np.testing.assert_allclose(env.renderer.qpos_curr_xarm,q,atol=1e-4)
                print(f'PASS: simulated six joints + gripper opening {opening:.2f} mm',flush=True)
            text = ('FOLLOWING' if active else status)+f' | gripper {opening:.1f} mm'
            display_image = np.concatenate([np.full((80,image.shape[1],3),20,dtype=np.uint8),image],axis=0)
            cv2.putText(display_image,text,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(80,230,100) if active else (0,210,255),1,cv2.LINE_AA)
            cv2.putText(display_image,'SPACE: follow/pause | O: record open, C: record closed (paused) | ESC: exit',(12,47),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
            cv2.putText(display_image,f'Tool: {speed:.2f} m/s | accel {accel:.1f} m/s2 (sim time) | {measured_hz:.1f}/{cfg.physics.fps} Hz | cap {args.max_joint_speed:.1f} rad/s',(12,71),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
            display_scale = min(1.5,1280/display_image.shape[1])
            display_image = cv2.resize(display_image,None,fx=display_scale,fy=display_scale,interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window,display_image)
            key = cv2.waitKey(1)&255
            if key == 27 or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                break
            if key == 32:
                active = not active if fresh else False
                if active:
                    current = env.renderer.get_state()
                    mapper.latch(ticks,env.renderer.qpos_curr_xarm,1-float(current['eef_gripper'][0,0]))
                    status = 'FOLLOWING'
                else:
                    status = 'PAUSED: SPACE to follow' if fresh else 'No fresh input; check USB'
            if not active and fresh and key in (ord('o'),ord('c')):
                mapper.endpoints['open' if key==ord('o') else 'close'] = int(ticks[6])
                if len(mapper.endpoints)==2:
                    span = abs(float(tick_delta(mapper.endpoints['close'],mapper.endpoints['open'])))
                    if span < 50:
                        mapper.endpoints.clear()
                        status = 'Calibration too small; record O and C again'
                    else:
                        calibration.write_text(json.dumps(dict(port=args.port,endpoints=mapper.endpoints),indent=2)+'\n')
                        status = 'Gripper calibrated; SPACE to follow'
                else:
                    status = 'Endpoint saved; record other endpoint with O/C'
            frames += 1
            if args.smoke_test and frames>=30:
                out = Path(args.project).resolve().parent/'gello_smoke.png'
                cv2.imwrite(str(out),display_image)
                print(f'Motion telemetry: peak {peak_speed:.3f} m/s, {peak_accel:.3f} m/s2; loop {frames/(time.monotonic()-run_start):.1f} Hz',flush=True)
                print('PASS: seven real encoders + simulated continuous joint/gripper targets; screenshot:',out,flush=True)
                break
            time.sleep(max(0,1/cfg.physics.fps-(time.monotonic()-start)))
            measured_hz = .9*measured_hz+.1/(time.monotonic()-start)
        env.close()
        cv2.destroyAllWindows()
    finally:
        reader.close()

if __name__ == '__main__':
    main()
