"""RealTable rope routing under seongbeen's absolute-action interface.

Adapted from rope_routing_nominal/action_correction/mass5/code/common.py.
The simulator, mass/stiffness scaling and 13-component absolute command are
unchanged. Scene-specific constants that were hardcoded for the upstream clip
placement are read from the project YAML instead.
"""
from pathlib import Path
import hashlib
import os
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('R2S_WORKSPACE', str(Path(__file__).resolve().parents[2]))).resolve()
REPO = Path(os.environ.get('REAL2SIM_EVAL', str(WORKSPACE / 'real2sim-eval'))).resolve()
PROJECT = Path(os.environ.get('ROUTING_PROJECT',
    str(WORKSPACE / 'log/rl/registered_shared_routing/routing_project.yaml'))).resolve()


def make_env(mass_scale=1.0, seed=0, config=None, stiffness_scale=1.0,
             gripper_friction_scale=1.0, theta=None, floor_friction=None,
             settle_steps=30, substep_scale=1.0, clip_friction=None):
    import torch
    from omegaconf import OmegaConf
    if not np.isfinite(mass_scale) or mass_scale <= 0:
        raise ValueError('mass_scale must be finite and positive')
    # GRIPPER_FRICTION_SCALE scales finger-rope friction for one session. With the
    # sticky grasp holding the rope, finger friction only drags the rope sideways
    # when the wrist turns; the real rope twists instead, which springs cannot.
    if 'GRIPPER_FRICTION_SCALE' in os.environ:
        gripper_friction_scale = float(os.environ['GRIPPER_FRICTION_SCALE'])
    # STIFFNESS_SCALE scales the rope's spring stiffness for one session.
    if 'STIFFNESS_SCALE' in os.environ:
        stiffness_scale = float(os.environ['STIFFNESS_SCALE'])
    # MASS_SCALE scales every particle's mass for one session.
    if 'MASS_SCALE' in os.environ:
        mass_scale = float(os.environ['MASS_SCALE'])
    if not np.isfinite([stiffness_scale, gripper_friction_scale]).all() or min(stiffness_scale, gripper_friction_scale) <= 0:
        raise ValueError('Physics scales must be finite and positive')
    os.environ.setdefault('R2S_WORKSPACE', str(WORKSPACE))
    os.chdir(REPO)
    for path in (str(REPO), str(HERE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import gymnasium as gym
    import sim.envs  # noqa: F401  (registers BaseEnv-v0)
    import sim.physics.phystwin as pt

    # FLOOR_SPRING_MASS_DIR selects which copy of the kernels to load, so the
    # separating-contact guard can be compared against the unmodified upstream.
    kernels = os.environ.get('FLOOR_SPRING_MASS_DIR')
    if kernels:
        sys.path.insert(0, str(Path(kernels).resolve()))
    from floor_spring_mass import SpringMassSystemWarp
    original_simulator = pt.SpringMassSystemWarp
    clip_value = float(os.environ['CLIP_FRICTION_ONLY']) if 'CLIP_FRICTION_ONLY' in os.environ else clip_friction

    class ScaledMassSystem(SpringMassSystemWarp):
        def __init__(self, *args, **kwargs):
            cfg_arg = kwargs.get('phystwin_cfg', args[0] if args else None)
            if theta is not None:
                cfg_arg.drag_damping = theta['drag_damping']
                cfg_arg.dashpot_damping = theta['dashpot_damping']
                cfg_arg.floor_elasticity_only = theta['ground_elasticity']
            kwargs['init_masses'] = kwargs['init_masses'] * mass_scale
            # Upstream supplies log spring stiffness and linear contact friction.
            if stiffness_scale != 1.0:
                kwargs['init_spring_Y'] = kwargs['init_spring_Y'] + np.log(stiffness_scale)
            if gripper_friction_scale != 1.0:
                kwargs['collide_eef_fric'] = kwargs['collide_eef_fric'] * gripper_friction_scale
            # The clip is a static mesh and shares the one fitted friction with
            # the floor; FLOOR_FRICTION_ONLY moves the floor alone. This moves
            # the clip alone.
            if clip_value is not None:
                kwargs['collide_fric'] = type(kwargs['collide_fric'])(clip_value) if not hasattr(kwargs['collide_fric'], 'shape') else kwargs['collide_fric'] * 0 + clip_value
            super().__init__(*args, **kwargs)

    pt.SpringMassSystemWarp = ScaledMassSystem
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    cfg = OmegaConf.load(str(config or PROJECT))
    # Nominal floor friction is the value identified from recordings 04/05/06;
    # perturbed conditions pass an explicit value instead.
    fitted_friction = float(cfg.physics.collide_fric)
    cfg.physics.floor_friction_only = float(
        os.environ['FLOOR_FRICTION_ONLY'] if 'FLOOR_FRICTION_ONLY' in os.environ
        else (fitted_friction if floor_friction is None else floor_friction))
    # STICKY_GRASP=1/0 overrides env.robot.sticky_grasp for one session, the same
    # way FLOOR_FRICTION_ONLY overrides the floor, so neither needs a second scene.
    if 'STICKY_GRASP' in os.environ:
        cfg.env.robot.sticky_grasp = os.environ['STICKY_GRASP'] == '1'
    # GRASP_RELEASE_MM=<mm>: a latched grasp survives the pads opening up to this
    # much past the width at which they met the rope (default 0: any widening lets go).
    if 'GRASP_RELEASE_MM' in os.environ:
        cfg.env.robot.grasp_release_margin_mm = float(os.environ['GRASP_RELEASE_MM'])
    # STICKY_MODE=point|hinge: how the pinned stretch may turn (see _apply_sticky_grasp).
    if 'STICKY_MODE' in os.environ:
        cfg.env.robot.sticky_grasp_mode = os.environ['STICKY_MODE']
    substep_multiplier = int(os.environ.get('SUBSTEP_MULTIPLIER', '1'))
    if substep_multiplier < 1:
        raise ValueError('SUBSTEP_MULTIPLIER must be a positive integer')
    # substep_scale below 1 trades integration resolution for wall-clock speed.
    # Use it to search for a teacher schedule, never to produce a demonstration:
    # replay and identification re-run at the recorded resolution.
    scale = float(os.environ.get('SUBSTEP_SCALE', substep_scale))
    if not 0 < scale <= 8:
        raise ValueError('substep_scale must be in (0, 8]')
    base = int(cfg.physics.num_substeps) * substep_multiplier
    cfg.physics.num_substeps = max(1, int(round(base * scale)))
    cfg.physics.dt = float(cfg.physics.dt) / substep_multiplier / scale
    cfg.seed = seed
    OmegaConf.resolve(cfg)
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        env = gym.make(cfg.env_name, max_episode_steps=10**9, cfg=cfg,
                       obs_mode=cfg.obs_mode, exp_root=cfg.exp_root,
                       local_rank=0, randomize=False)
        obs, _ = env.reset(seed=seed)
    finally:
        pt.SpringMassSystemWarp = original_simulator
    dm = env.unwrapped.physics.dynamics_module
    masses = dm.simulator.wp_masses.numpy()
    if not np.allclose(masses, mass_scale):
        raise RuntimeError(f'Mass readback differs from requested {mass_scale}')
    readback = dict(floor=float(dm.simulator.wp_ground_fric.numpy()[0]),
                    static_mesh=float(dm.simulator.wp_collide_fric.numpy()[0]),
                    gripper=float(dm.simulator.wp_collide_eef_fric.numpy()[0]))
    print('FRICTION_READBACK', readback, flush=True)
    expected_gripper = float(cfg.physics.collide_eef_fric) * gripper_friction_scale
    if not np.isclose(readback['floor'], cfg.physics.floor_friction_only) \
       or not np.isclose(readback['static_mesh'], fitted_friction if clip_value is None else clip_value) \
       or not np.isclose(readback['gripper'], expected_gripper):
        raise RuntimeError('Unexpected friction intervention')
    # Clips are the meshes named clip, clip1, ...; the routing predicate needs one.
    # A scene without any (an object scene, e.g. the T-block with no static
    # mesh at all) is fine for teleop, replay and mirroring.
    if not any(str(mm.name).startswith('clip') for mm in cfg.gs.meshes or []):
        print('NOTE: no clip in this scene; routing counts will be 0', flush=True)
    # Let the reconstructed rope settle before recording, exactly as replay does.
    import kornia
    robot = obs['robot']
    rotation = kornia.geometry.conversions.quaternion_to_rotation_matrix(robot['eef_quat'])
    hold = torch.cat([robot['eef_xyz'], rotation.reshape(1, 9), robot['eef_gripper']], dim=1)
    hold = hold.detach().cpu().numpy().copy()
    for _ in range(settle_steps):
        step(env, hold)
    obs = env.unwrapped.get_obs()
    # Upstream env.reset() calls renderer.set_all_cameras(), which appends to
    # the camera lists without clearing them: every reset adds another copy of
    # each fixed camera, the display grows a pair of views per press of the
    # reset key, and every frame renders them all. Clear the lists first. Done
    # here rather than in the upstream file so the simulator stays as shipped.
    renderer = env.unwrapped.renderer
    original = renderer.set_all_cameras

    def set_all_cameras_fresh():
        renderer.cameras = []
        renderer.wrist_cameras = []
        return original()

    renderer.set_all_cameras = set_all_cameras_fresh
    return env, obs, cfg


def clip_meshes(cfg):
    """The clips, in scene order: meshes named clip, clip1, ... Other static objects are not clips."""
    return [mm for mm in cfg.gs.meshes or [] if str(mm.name).startswith('clip')]


def clip_pose(cfg, index=0):
    """Pose of one clip; index 0 is the one nearest the rope when the scene was measured."""
    return np.asarray(clip_meshes(cfg)[index].pose, dtype=float).reshape(4, 4)


def clip_poses(cfg):
    return [np.asarray(mm.pose, dtype=float).reshape(4, 4) for mm in clip_meshes(cfg)]


def finger_forces(env):
    """Opposed squeeze load per finger, read the same way the grasp latch does.

    Recomputed here rather than stored by the simulator so the shared physics
    files stay untouched. Mirrors PhysTwinDynamics._step_ur5e_mesh.
    """
    from scipy.spatial.transform import Rotation
    dm = env.unwrapped.physics.dynamics_module
    forces = dm.simulator.collision_forces.numpy()
    mesh_map = dm.simulator.mesh_map.numpy()
    quat = env.unwrapped.renderer.get_state()['eef_quat'].detach().cpu().numpy().reshape(4)
    squeeze_axis = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()[:, 1]
    norms = np.array([max(0., sign * (forces[mesh_map == i].sum(axis=0) @ squeeze_axis))
                      for i, sign in ((0, 1.), (1, -1.))], dtype=np.float32)
    threshold = float(env.unwrapped.cfg.env.robot.get('contact_stop_threshold', 100.0))
    return norms, bool(np.all(norms > threshold))


def state(env):
    r = env.unwrapped.renderer
    s = r.get_state()
    dm = env.unwrapped.physics.dynamics_module
    arr = lambda x: np.asarray(x.detach().cpu().numpy().copy(), dtype=np.float32)
    forces, bilateral = finger_forces(env)
    # renderer 'v' is computed as (x_pred - x0) * fps where x0 aliases the warp
    # buffer step() writes in place, so it is identically zero. Read the
    # simulator's own velocity state, as the upstream common.py does.
    return dict(x=arr(s['x']), v=arr(dm.current_velocities), eef=arr(s['eef_xyz']),
                eef_quat=arr(s['eef_quat']), gripper=arr(s['eef_gripper']),
                qpos=np.asarray(r.qpos_curr_xarm, dtype=np.float32).copy(),
                latch=np.bool_(dm.grasped),
                force_left=np.float32(forces[0]), force_right=np.float32(forces[1]),
                bilateral_contact=np.bool_(bilateral),
                opening=np.float32(dm.current_openness if dm.current_openness is not None else 1.0))


def rope_bins(env, cfg, count=24):
    """Ordered particle groups along the rope, as RopeRoutingEnv builds them."""
    x = env.unwrapped.renderer.get_state()['x'].detach().cpu().numpy()
    direction = np.asarray(cfg.gs.object.pose, dtype=float).reshape(4, 4)[:3, 0]
    return np.array_split(np.argsort(x @ direction), count)


def rope_centerline(env, bins):
    x = env.unwrapped.renderer.get_state()['x'].detach().cpu().numpy()
    return np.array([x[b].mean(0) for b in bins])


def channel_geometry(cfg, index=0):
    """Half width, wall height and clip scale of one clip's channel, in metres.

    The stock clip is 35 mm wide with 30 mm walls. clip_scale on the mesh
    records a resized print; the window scales with it.
    """
    scale = float(clip_meshes(cfg)[index].get('clip_scale', 1.))
    return .0175 * scale, .030 * scale, scale


def channel_crossing(centerline, clip_pose, half_width, wall_height, radius=.0125):
    """Continuous rope segment crosses both ends of the open channel.

    Same predicate as rope_routing_env.channel_crossing, with the 35 mm x 30 mm
    window as arguments so a resized clip can be scored. Consecutive rows are
    treated as connected rope, so this needs the ordered centerline.
    """
    local = (np.asarray(centerline) - clip_pose[:3, 3]) @ clip_pose[:3, :3]
    clearance = half_width - radius
    hits = []
    for plane in (-half_width, half_width):
        found = []
        for i, (a, b) in enumerate(zip(local[:-1], local[1:])):
            dy = b[1] - a[1]
            if abs(dy) < 1e-9:
                continue
            t = (plane - a[1]) / dy
            if 0 <= t <= 1:
                p = a + t * (b - a)
                if abs(p[0]) <= clearance and radius - .003 <= p[2] <= wall_height:
                    found.append(i + t)
        hits.append(found)
    if not hits[0] or not hits[1]:
        return False
    for lo in hits[0]:
        for hi in hits[1]:
            low, high = sorted([lo, hi])
            inside = local[int(np.ceil(low)):int(np.floor(high)) + 1]
            if len(inside) and np.any((np.abs(inside[:, 0]) > clearance)
                                      | (inside[:, 2] < radius - .003)
                                      | (inside[:, 2] > wall_height)):
                continue
            return True
    return False


def in_clip(env, cfg, bins=None):
    """Rope centerline passes through the clip channel (RealTable criterion)."""
    if bins is None:
        bins = rope_bins(env, cfg)
    half_width, wall_height, _ = channel_geometry(cfg)
    return bool(channel_crossing(rope_centerline(env, bins), clip_pose(cfg),
                                 half_width, wall_height))


def images(env):
    """Side and wrist RGB frames, BGR for OpenCV."""
    import cv2
    obs = env.unwrapped.get_obs()
    def frame(tensor):
        rgb = (tensor[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return frame(obs['image_list'][0]), frame(obs['image_wrist_list'][0])


def step(env, action):
    import torch
    command = torch.as_tensor(action, dtype=torch.float32, device='cuda:0').reshape(1, 13)
    effective = env.unwrapped.renderer.mimic_velocity_control(command.clone())
    saved = effective.detach().cpu().numpy().copy()
    env.step({'action': effective, 'do_velocity_control': False})
    return saved


def fingerprint():
    paths = ['sim/physics/phystwin.py', 'sim/physics/spring_mass_warp.py',
             'sim/renderer/gs_renderer.py',
             'experiments/utils/calculate_success_rope.py']
    result = {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in paths}
    result['floor_spring_mass.py'] = hashlib.sha256((HERE / 'floor_spring_mass.py').read_bytes()).hexdigest()
    return result
