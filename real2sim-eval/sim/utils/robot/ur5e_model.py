"""Pinned upstream URDF generation and name-based FK, including Robotiq mimic joints."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
def finite_vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected {size} finite values')
    return a

SOURCES = {
    'universal_robot': ('https://github.com/ros-industrial/universal_robot.git',
                        '39ad110d8f2e8f66856a201cca88aa7a7025e3eb'),
    'robotiq': ('https://github.com/ros-industrial-attic/robotiq.git',
                '45196f6558fe8ba9d89bc8a105396c68c3e7e892'),
}


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def origin(node):
    o = node.find('origin')
    if o is None:
        return np.eye(4)
    return transform(np.fromstring(o.get('xyz', '0 0 0'), sep=' '),
                     np.fromstring(o.get('rpy', '0 0 0'), sep=' '))


def opening_to_angle(mm):
    """Parallel grasp linkage approximation from the 2F-85 geometry; 85=open."""
    if not np.isfinite(mm) or not 0 <= mm <= 85:
        raise ValueError('그리퍼 벌림은 0~85 mm입니다.')
    # Normalize the upstream linkage's end stops to the nominal 0..85 mm UI.
    lo = .0376 * np.cos(.8) - .043 * np.sin(.8)
    y = lo + (mm / 85.) * (.0376 - lo)
    return float(np.clip(np.arccos(y / np.hypot(.0376, .043)) - np.arctan2(.043, .0376), 0, .8))


def angle_to_opening(angle):
    lo = .0376 * np.cos(.8) - .043 * np.sin(.8)
    return float(np.clip(85 * ((.0376 * np.cos(angle) - .043 * np.sin(angle)) - lo) / (.0376 - lo), 0, 85))


class Robot:
    def __init__(self, path):
        self.path = Path(path)
        root = ET.parse(path).getroot()
        self.links = {l.get('name'): l for l in root.findall('link')}
        self.joints = {j.get('name'): j for j in root.findall('joint')}
        children = {j.find('child').get('link') for j in self.joints.values()}
        roots = set(self.links) - children
        if len(roots) != 1:
            raise ValueError('URDF는 하나의 루트 링크를 가져야 합니다.')
        self.root = roots.pop()
        self.meshes = {}
        for name, link in self.links.items():
            parts = []
            for visual in link.findall('visual'):
                geom = visual.find('geometry/mesh')
                box = visual.find('geometry/box')
                cylinder = visual.find('geometry/cylinder')
                sphere = visual.find('geometry/sphere')
                if geom is not None:
                    mesh_path = self.path.parent / geom.get('filename')
                    colored_path = mesh_path.with_suffix('.ply')
                    if mesh_path.suffix.lower() == '.stl' and colored_path.exists():
                        mesh_path = colored_path
                    m = trimesh.load(mesh_path, force='mesh')
                    m.apply_scale(np.fromstring(geom.get('scale', '1 1 1'), sep=' '))
                elif box is not None:
                    m = trimesh.creation.box(extents=np.fromstring(box.get('size'), sep=' '))
                elif cylinder is not None:
                    m = trimesh.creation.cylinder(radius=float(cylinder.get('radius')), height=float(cylinder.get('length')))
                elif sphere is not None:
                    m = trimesh.creation.icosphere(radius=float(sphere.get('radius')))
                else:
                    continue
                m.apply_transform(origin(visual))
                color = visual.find('material/color')
                rgba = np.fromstring(color.get('rgba'), sep=' ') if color is not None else [.6, .7, .8, 1]
                # Embedded mesh colors take precedence over the fallback URDF color.
                if not m.visual.defined:
                    m.visual.vertex_colors = (np.asarray(rgba) * 255).astype(np.uint8)
                parts.append(m)
            if parts:
                self.meshes[name] = trimesh.util.concatenate(parts)
        if not all(j in self.joints for j in JOINTS + ['finger_joint']):
            raise ValueError('UR5e 또는 2F-85 관절이 없습니다.')

    def configuration(self, q_deg, opening_mm):
        q = finite_vector(q_deg, 6, '관절각')
        values = dict(zip(JOINTS, np.radians(q)))
        values['finger_joint'] = opening_to_angle(opening_mm)
        def resolve(name, visiting):
            if name in values:
                return values[name]
            if name in visiting:
                raise ValueError('URDF mimic 관절이 순환합니다.')
            joint = self.joints[name]
            mimic = joint.find('mimic')
            if mimic is None:
                return 0.0
            values[name] = resolve(mimic.get('joint'), visiting | {name}) * float(mimic.get('multiplier', '1')) + float(mimic.get('offset', '0'))
            return values[name]
        for name in self.joints:
            resolve(name, set())
        return values

    def fk(self, q_deg=None, opening_mm=85, values=None):
        values = self.configuration(q_deg, opening_mm) if values is None else values
        result = {self.root: np.eye(4)}
        todo = dict(self.joints)
        while todo:
            previous = len(todo)
            for name, j in list(todo.items()):
                parent, child = j.find('parent').get('link'), j.find('child').get('link')
                if parent not in result:
                    continue
                motion = np.eye(4)
                if j.get('type') in ('revolute', 'continuous', 'prismatic'):
                    axis_node = j.find('axis')
                    axis = np.fromstring(axis_node.get('xyz', '1 0 0'), sep=' ') if axis_node is not None else np.array([1., 0, 0])
                    axis /= np.linalg.norm(axis)
                    if j.get('type') == 'prismatic':
                        motion[:3, 3] = axis * values.get(name, 0)
                    else:
                        motion[:3, :3] = Rotation.from_rotvec(axis * values.get(name, 0)).as_matrix()
                result[child] = result[parent] @ origin(j) @ motion
                del todo[name]
            if len(todo) == previous:
                raise ValueError('URDF 링크 트리가 연결되지 않았습니다.')
        return result

    def sample(self, q_deg, opening_mm, per_link=1800, seed=7):
        fk = self.fk(q_deg, opening_mm)
        points, labels, names = [], [], list(self.meshes)
        for i, name in enumerate(names):
            pts, _ = trimesh.sample.sample_surface(self.meshes[name], per_link, seed=seed + i)
            points.append(pts @ fk[name][:3, :3].T + fk[name][:3, 3])
            labels.append(np.full(per_link, i, dtype=np.int32))
        return np.concatenate(points), np.concatenate(labels), names
