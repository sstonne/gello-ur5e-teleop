"""UR5e/2F-85 adapter for the original GSRenderer and PhysTwin simulator.

All transforms use URDF base_link and tool0 (flange outward Z), in metres.
Dynamic meshes include both fingers with rubber pads and the palm housing,
evaluated with the same URDF FK as the rendered robot.
"""
import json
from pathlib import Path
import numpy as np
import torch
import kornia
import open3d as o3d
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .ur5e_model import Robot, JOINTS, origin


class UR5eRobotiq:
    sapien_eef_idx = 'tool0'

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = Robot(cfg.env.urdf.ik_urdf_path)
        self.initial_q = np.radians(cfg.env.robot.init_joints_deg)
        self.reference = self.model.fk(np.degrees(self.initial_q), 85)
        self.names = list(self.model.meshes)
        self.scene_reference = None
        self.finger_names = ['left_inner_finger', 'right_inner_finger']
        self.finger_meshes = []
        for side, name in zip(['left', 'right'], self.finger_names):
            pad = self.model.meshes[f'{side}_inner_finger_pad'].copy()
            pad.apply_transform(np.linalg.inv(self.reference[name]) @ self.reference[f'{side}_inner_finger_pad'])
            # Convex hull is a compact collision proxy; visual splats use full meshes.
            mesh = trimesh.util.concatenate([self.model.meshes[name], pad]).convex_hull
            self.finger_meshes.append(mesh)
        # The palm must also be solid; otherwise rope can pass through the
        # visible gripper housing once it slides above the finger pads.
        self.finger_names.append('robotiq_arg2f_base_link')
        self.finger_meshes.append(self.model.meshes['robotiq_arg2f_base_link'].convex_hull)
        self.collision_splits = np.cumsum([len(m.vertices) for m in self.finger_meshes])[:-1]
        self.finger_split = int(self.collision_splits[0])
        self.limits = np.array([[float(self.model.joints[n].find('limit').get(k))
                                for k in ('lower', 'upper')] for n in JOINTS])
        self.chain = []
        child_to_joint = {j.find('child').get('link'): (n, j) for n, j in self.model.joints.items()}
        child = 'tool0'
        while child in child_to_joint:
            name, joint = child_to_joint[child]
            axis = joint.find('axis')
            self.chain.append((JOINTS.index(name) if name in JOINTS else None,
                               origin(joint), np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else None))
            child = joint.find('parent').get('link')
        self.chain.reverse()

    def tool_pose(self, q):
        T = np.eye(4)
        for idx, fixed, axis in self.chain:
            T = T @ fixed
            if idx is not None:
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis * q[idx]).as_matrix()
                T = T @ motion
        return T

    def compute_fk_sapien_links(self, qpos, link_idx):
        fk = self.model.fk(np.degrees(np.asarray(qpos)[:6]), 85)
        return [fk[name] for name in link_idx]

    def solve_ik(self, initial, target):
        initial = np.clip(np.asarray(initial), self.limits[:, 0], self.limits[:, 1])
        def error(q):
            T = self.tool_pose(q)
            return np.r_[T[:3, 3] - target[:3, 3],
                         .3 * Rotation.from_matrix(T[:3, :3] @ target[:3, :3].T).as_rotvec()]
        result = least_squares(error, initial, bounds=(self.limits[:, 0], self.limits[:, 1]),
                               max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        e = error(result.x)
        return result.x, bool(np.linalg.norm(e[:3]) < .001 and np.linalg.norm(e[3:]) < .003)

    def compute_ik_sapien(self, initial_qpos, cartesian, verbose=False):
        target = np.eye(4)
        target[:3, 3] = cartesian[:3]
        target[:3, :3] = Rotation.from_euler('xyz', cartesian[3:]).as_matrix()
        q, ok = self.solve_ik(initial_qpos, target)
        return q if ok else np.asarray(initial_qpos).copy()

    def finger_local_points(self, openness):
        fk = self.model.fk(np.degrees(self.initial_q), float(np.clip(openness, 0, 1)) * 85)
        inv_tool = np.linalg.inv(fk['tool0'])
        points = []
        for name, mesh in zip(self.finger_names, self.finger_meshes):
            T = inv_tool @ fk[name]
            points.append(mesh.vertices @ T[:3, :3].T + T[:3, 3])
        return np.concatenate(points)

    def get_gripper_meshes(self, openness):
        pts = self.finger_local_points(openness)
        T = self.tool_pose(self.initial_q)
        pts = pts @ T[:3, :3].T + T[:3, 3]
        meshes = []
        for vertices, source in zip(np.split(pts, self.collision_splits), self.finger_meshes):
            m = o3d.geometry.TriangleMesh()
            m.vertices = o3d.utility.Vector3dVector(vertices)
            m.triangles = o3d.utility.Vector3iVector(source.faces)
            meshes.append(m)
        return meshes

    def prepare_scene(self, params, labels):
        scene = self.cfg.gs.scene
        if scene.get('robot_manifest'):
            manifest_path = Path(scene.robot_manifest)
            manifest = json.loads(manifest_path.read_text())
            self.names = manifest['link_names']
            with np.load(manifest_path.parent / 'scan_link_transforms.npz') as data:
                self.scene_reference = {name: data[name].copy() for name in self.names}
            if any(name not in self.model.links for name in self.names):
                raise ValueError('Scan link names do not match UR5e/Robotiq URDF')
            if len(labels) != len(params['means3D']):
                raise ValueError('Scene and robot labels have different lengths')
            valid = labels != -2
            return {k: v[valid] for k, v in params.items()}, labels[valid]

        # Installed xArm scan provides the demo background only. Replace its robot.
        valid = labels == -1
        params = {k: v[valid] for k, v in params.items()}
        self.names = list(self.model.meshes)
        self.scene_reference = self.reference
        samples, colors, labels_np, splat_scales, splat_rotations = [], [], [], [], []
        light = np.array([.3, -.4, .85]); light /= np.linalg.norm(light)
        for i, name in enumerate(self.names):
            mesh = self.model.meshes[name]
            # Cover the surface at a metric density. Small gripper parts need
            # much smaller splats than arm tubes; spherical 3 mm splats inflated
            # fingers and left random holes in the large arm links.
            spacing = .0008 if name not in ('base_link', 'shoulder_link', 'upper_arm_link',
                                            'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link') else .002
            count = int(np.clip(np.ceil(mesh.area / spacing**2), 1200, 90000))
            p, faces = trimesh.sample.sample_surface(mesh, count, seed=17+i)
            T = self.reference[name]
            samples.append(p @ T[:3, :3].T + T[:3, 3])
            normals = mesh.face_normals[faces] @ T[:3, :3].T
            bary = trimesh.triangles.points_to_barycentric(mesh.triangles[faces], p)
            shading_normals = np.einsum('ni,nij->nj', bary, mesh.vertex_normals[mesh.faces[faces]]) @ T[:3, :3].T
            shading_normals /= np.maximum(np.linalg.norm(shading_normals, axis=1, keepdims=True), 1e-12)
            shade = .4 + .6*np.maximum(shading_normals @ light, 0)
            # Thin ellipsoids lie tangent to the actual triangles. Their normal
            # thickness is 0.1 mm, retaining openings and the URDF silhouette.
            axes = np.tile([0., 0., 1.], (len(p), 1))
            axes[np.abs(normals[:, 2]) > .9] = [1., 0., 0.]
            tangent = np.cross(axes, normals)
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
            bitangent = np.cross(normals, tangent)
            frames = np.stack([tangent, bitangent, normals], axis=2)
            splat_rotations.append(Rotation.from_matrix(frames).as_quat()[:, [3, 0, 1, 2]])
            radius = np.sqrt(mesh.area/count)*1.1
            splat_scales.append(np.tile([radius, radius, .0001], (len(p), 1)))
            color = np.asarray(mesh.visual.face_colors[faces, :3])/255
            colors.append(shade[:, None]*color)
            labels_np.append(np.full(len(p), i))
        pts, rgb, ids = np.concatenate(samples), np.concatenate(colors), np.concatenate(labels_np)
        n = len(pts)
        device = labels.device
        shs = torch.zeros((n, 16, 3), device=device)
        shs[:, 0] = torch.tensor((rgb-.5)/.28209479177387814, device=device)
        robot_params = {
            'means3D': torch.tensor(pts, dtype=torch.float32, device=device),
            'shs': shs, 'scales': torch.tensor(np.concatenate(splat_scales), dtype=torch.float32, device=device),
            'rotations': torch.tensor(np.concatenate(splat_rotations), dtype=torch.float32, device=device),
            'opacities': torch.full((n, 1), .98, device=device),
            'means2D': torch.zeros((n, 3), device=device),
        }
        return {k: torch.cat([v, robot_params[k]]) for k, v in params.items()}, torch.cat([
            labels[valid], torch.tensor(ids, device=device, dtype=labels.dtype)])

    def transform_scene(self, params, labels, q, openness):
        fk = self.model.fk(np.degrees(q), float(np.clip(openness, 0, 1))*85)
        for idx, name in enumerate(self.names):
            mask = labels == idx
            T = torch.tensor(fk[name] @ np.linalg.inv(self.scene_reference[name]),
                             device=labels.device, dtype=torch.float32)
            params['means3D'][mask] = params['means3D'][mask] @ T[:3, :3].T + T[:3, 3]
            R = kornia.geometry.conversions.quaternion_to_rotation_matrix(params['rotations'][mask])
            params['rotations'][mask] = kornia.geometry.conversions.rotation_matrix_to_quaternion(T[:3, :3] @ R)
        return params

    def project_action(self, renderer, action, velocity_control=True):
        """Send the same achievable pose to both rendering and contact physics."""
        if action.shape != (1, 13):
            if action.shape != (1, 7):
                raise ValueError('UR5e action: [xyz, rotation matrix, openness] or [6 radians, closure]')
            q = action[0, :6].detach().cpu().numpy()
            if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
                raise ValueError('Joint target exceeds UR5e limits')
            T = self.tool_pose(q)
            out = torch.tensor(np.r_[T[:3, 3], T[:3, :3].ravel(), 1-float(action[0, -1])],
                               dtype=action.dtype, device=action.device)[None]
        else:
            out = action.clone()
            T = np.eye(4)
            T[:3, 3] = action[0, :3].detach().cpu().numpy()
            T[:3, :3] = action[0, 3:12].detach().cpu().numpy().reshape(3, 3)
            q, ok = self.solve_ik(renderer.qpos_curr_xarm, T)
            if not ok:
                q = renderer.qpos_curr_xarm.copy()
        if velocity_control:
            delta = q-renderer.qpos_curr_xarm
            limit = float(self.cfg.env.robot.max_joint_speed_rad_s)/self.cfg.physics.fps
            q = renderer.qpos_curr_xarm + delta * min(1., limit/max(np.max(np.abs(delta)), 1e-12))
        reached = self.tool_pose(q)
        out[0, :3] = torch.as_tensor(reached[:3, 3], device=out.device)
        out[0, 3:12] = torch.as_tensor(reached[:3, :3].ravel(), device=out.device)
        out[0, 12] = out[0, 12].clamp(0, 1)
        if velocity_control:
            current_opening = float(renderer.grippers[0, 13])
            speed = 2.0/self.cfg.physics.fps
            out[0, 12] = out[0, 12].clamp(max(0., current_opening-speed), min(1., current_opening+speed))
        renderer.qpos_curr_xarm = q.copy()
        return out
