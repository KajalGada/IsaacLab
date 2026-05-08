# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build ur5_with_scoop.usd as a PhysX articulation using only PyUSD (no Isaac Sim extension required).

Run once to generate the asset:
    ./isaaclab.sh -p source/isaaclab_tasks_experimental/isaaclab_tasks_experimental/direct/scoop/assets/create_ur5_scoop_usd.py

The resulting USD uses:
  - primitive collision shapes (capsules/spheres/cylinder) — no mesh dependency
  - mass/inertia values from the original URDF
  - drive gains matching the Newton reference script (ke=2000, kd=100)
  - a fixed joint from world to base_link so the robot is fixed-base
"""

import math
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_DIR = os.path.dirname(os.path.abspath(__file__))
_USD_PATH = os.path.join(_DIR, "ur5_with_scoop.usd")

# UR5 DH parameters (metres)
D1 = 0.089159
A2 = -0.425
A3 = -0.39225
D4 = 0.10915
D5 = 0.09465
D6 = 0.0823

PI = math.pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rpy_to_quatf(r: float, p: float, y: float) -> Gf.Quatf:
    """URDF extrinsic-XYZ RPY → Gf.Quatf (w, x, y, z)."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yq = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Gf.Quatf(w, x, yq, z)


def _add_link(stage: Usd.Stage, path: str, mass: float,
              diag_inertia: tuple, cog: tuple = (0.0, 0.0, 0.0)) -> UsdGeom.Xform:
    xform = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag_inertia))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*cog))
    return xform


def _add_collision(prim: Usd.Prim) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)


def _add_capsule(stage: Usd.Stage, parent_path: str, name: str,
                 radius: float, height: float, axis: str = "X",
                 trans: tuple = (0.0, 0.0, 0.0)) -> UsdGeom.Capsule:
    cap = UsdGeom.Capsule.Define(stage, f"{parent_path}/{name}")
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(height)
    cap.CreateAxisAttr(axis)
    cap.AddTranslateOp().Set(Gf.Vec3d(*trans))
    _add_collision(cap.GetPrim())
    return cap


def _add_sphere(stage: Usd.Stage, parent_path: str, name: str,
                radius: float, trans: tuple = (0.0, 0.0, 0.0)) -> UsdGeom.Sphere:
    sph = UsdGeom.Sphere.Define(stage, f"{parent_path}/{name}")
    sph.CreateRadiusAttr(radius)
    sph.AddTranslateOp().Set(Gf.Vec3d(*trans))
    _add_collision(sph.GetPrim())
    return sph


def _add_cylinder(stage: Usd.Stage, parent_path: str, name: str,
                  radius: float, height: float, axis: str = "Z",
                  trans: tuple = (0.0, 0.0, 0.0)) -> UsdGeom.Cylinder:
    cyl = UsdGeom.Cylinder.Define(stage, f"{parent_path}/{name}")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    cyl.AddTranslateOp().Set(Gf.Vec3d(*trans))
    _add_collision(cyl.GetPrim())
    return cyl


def _add_cube(stage: Usd.Stage, parent_path: str, name: str,
              size: tuple, trans: tuple = (0.0, 0.0, 0.0)) -> UsdGeom.Cube:
    # USD Cube is a unit cube; scale via xformOp:scale
    cube = UsdGeom.Cube.Define(stage, f"{parent_path}/{name}")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.AddTranslateOp().Set(Gf.Vec3d(*trans))
    _add_collision(cube.GetPrim())
    return cube


def _revolute_joint(stage: Usd.Stage, path: str,
                    body0: str, body1: str,
                    local_pos0: tuple, local_rpy0: tuple,
                    lower_deg: float, upper_deg: float,
                    stiffness: float = 2000.0, damping: float = 100.0,
                    effort_limit: float = 150.0) -> UsdPhysics.RevoluteJoint:
    jnt = UsdPhysics.RevoluteJoint.Define(stage, path)
    jnt.CreateAxisAttr("Z")
    jnt.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    jnt.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    jnt.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    jnt.CreateLocalRot0Attr(_rpy_to_quatf(*local_rpy0))
    jnt.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    jnt.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    jnt.CreateLowerLimitAttr(lower_deg)
    jnt.CreateUpperLimitAttr(upper_deg)
    # Angular position drive
    drive = UsdPhysics.DriveAPI.Apply(jnt.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(effort_limit)
    return jnt


def _fixed_joint(stage: Usd.Stage, path: str,
                 body0: str | None, body1: str,
                 local_pos0: tuple = (0.0, 0.0, 0.0),
                 local_rpy0: tuple = (0.0, 0.0, 0.0)) -> UsdPhysics.FixedJoint:
    jnt = UsdPhysics.FixedJoint.Define(stage, path)
    if body0 is not None:
        jnt.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    jnt.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    jnt.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    jnt.CreateLocalRot0Attr(_rpy_to_quatf(*local_rpy0))
    jnt.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    jnt.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return jnt


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(usd_path: str = _USD_PATH) -> None:
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    ROOT = "/Robot"
    root_xform = UsdGeom.Xform.Define(stage, ROOT)
    UsdPhysics.ArticulationRootAPI.Apply(root_xform.GetPrim())

    J = f"{ROOT}/joints"
    UsdGeom.Scope.Define(stage, J)

    # ------------------------------------------------------------------
    # Links — primitive collision shapes, URDF mass/inertia
    # ------------------------------------------------------------------

    # base_link_inertia (merged with base_link for simplicity)
    _add_link(stage, f"{ROOT}/base_link",
              mass=4.0, diag_inertia=(0.00443, 0.00443, 0.0072))
    _add_cylinder(stage, f"{ROOT}/base_link", "col", radius=0.06, height=0.05)

    # shoulder_link
    _add_link(stage, f"{ROOT}/shoulder_link",
              mass=3.7, diag_inertia=(0.01497, 0.01497, 0.01041),
              cog=(0.0, -0.00193, -0.02561))
    _add_sphere(stage, f"{ROOT}/shoulder_link", "col", radius=0.055)

    # upper_arm_link — capsule along X (arm extends in -X from shoulder)
    _add_link(stage, f"{ROOT}/upper_arm_link",
              mass=8.393, diag_inertia=(0.01511, 0.13389, 0.13389),
              cog=(-0.2125, 0.0, 0.11336))
    _add_capsule(stage, f"{ROOT}/upper_arm_link", "col",
                 radius=0.04, height=0.36, axis="X", trans=(-0.2125, 0.0, 0.13585))

    # forearm_link — capsule along X
    _add_link(stage, f"{ROOT}/forearm_link",
              mass=2.33, diag_inertia=(0.004095, 0.03122, 0.03122),
              cog=(-0.24225, 0.0, 0.0265))
    _add_capsule(stage, f"{ROOT}/forearm_link", "col",
                 radius=0.035, height=0.32, axis="X", trans=(-0.196, 0.0, 0.0165))

    # wrist_1_link
    _add_link(stage, f"{ROOT}/wrist_1_link",
              mass=1.219, diag_inertia=(0.002014, 0.002014, 0.002194),
              cog=(0.0, -0.01634, -0.0018))
    _add_sphere(stage, f"{ROOT}/wrist_1_link", "col", radius=0.038, trans=(0.0, 0.0, -0.093))

    # wrist_2_link
    _add_link(stage, f"{ROOT}/wrist_2_link",
              mass=1.219, diag_inertia=(0.001831, 0.001831, 0.002194),
              cog=(0.0, 0.01634, -0.0018))
    _add_sphere(stage, f"{ROOT}/wrist_2_link", "col", radius=0.038, trans=(0.0, 0.0, -0.095))

    # wrist_3_link
    _add_link(stage, f"{ROOT}/wrist_3_link",
              mass=0.1879, diag_inertia=(8.06e-5, 8.06e-5, 1.32e-4),
              cog=(0.0, 0.0, -0.001159))
    _add_sphere(stage, f"{ROOT}/wrist_3_link", "col", radius=0.032, trans=(0.0, 0.0, -0.082))

    # scoop_link — flat box approximating the scoop paddle (15 cm × 12 cm × 3 cm)
    _add_link(stage, f"{ROOT}/scoop_link",
              mass=0.5, diag_inertia=(0.001, 0.001, 0.0005),
              cog=(0.0, 0.0, 0.05))
    _add_cube(stage, f"{ROOT}/scoop_link", "col",
              size=(0.15, 0.12, 0.03), trans=(0.0, 0.0, 0.05))

    # ------------------------------------------------------------------
    # Joints
    # ------------------------------------------------------------------

    # World → base_link: fix the robot to the world
    _fixed_joint(stage, f"{J}/world_to_base",
                 body0=None, body1=f"{ROOT}/base_link")

    # shoulder_pan (base_link → shoulder_link): z offset = d1
    _revolute_joint(stage, f"{J}/shoulder_pan_joint",
                    f"{ROOT}/base_link", f"{ROOT}/shoulder_link",
                    local_pos0=(0.0, 0.0, D1), local_rpy0=(0.0, 0.0, 0.0),
                    lower_deg=-360.0, upper_deg=360.0, effort_limit=150.0)

    # shoulder_lift (shoulder_link → upper_arm_link): rpy = (pi/2, 0, 0)
    _revolute_joint(stage, f"{J}/shoulder_lift_joint",
                    f"{ROOT}/shoulder_link", f"{ROOT}/upper_arm_link",
                    local_pos0=(0.0, 0.0, 0.0), local_rpy0=(PI / 2, 0.0, 0.0),
                    lower_deg=-360.0, upper_deg=360.0, effort_limit=150.0)

    # elbow (upper_arm_link → forearm_link): x offset = a2
    _revolute_joint(stage, f"{J}/elbow_joint",
                    f"{ROOT}/upper_arm_link", f"{ROOT}/forearm_link",
                    local_pos0=(A2, 0.0, 0.0), local_rpy0=(0.0, 0.0, 0.0),
                    lower_deg=-180.0, upper_deg=180.0, effort_limit=150.0)

    # wrist_1 (forearm_link → wrist_1_link): (a3, 0, d4)
    _revolute_joint(stage, f"{J}/wrist_1_joint",
                    f"{ROOT}/forearm_link", f"{ROOT}/wrist_1_link",
                    local_pos0=(A3, 0.0, D4), local_rpy0=(0.0, 0.0, 0.0),
                    lower_deg=-360.0, upper_deg=360.0, effort_limit=28.0)

    # wrist_2 (wrist_1_link → wrist_2_link): (0, -d5, 0), rpy=(pi/2, 0, 0)
    _revolute_joint(stage, f"{J}/wrist_2_joint",
                    f"{ROOT}/wrist_1_link", f"{ROOT}/wrist_2_link",
                    local_pos0=(0.0, -D5, 0.0), local_rpy0=(PI / 2, 0.0, 0.0),
                    lower_deg=-360.0, upper_deg=360.0, effort_limit=28.0)

    # wrist_3 (wrist_2_link → wrist_3_link): (0, d6, 0), rpy=(pi/2, pi, pi)
    _revolute_joint(stage, f"{J}/wrist_3_joint",
                    f"{ROOT}/wrist_2_link", f"{ROOT}/wrist_3_link",
                    local_pos0=(0.0, D6, 0.0), local_rpy0=(PI / 2, PI, PI),
                    lower_deg=-360.0, upper_deg=360.0, effort_limit=28.0)

    # wrist_3 → scoop_link: combined fixed transform through flange/tool0
    # Net transform from wrist_3_link to scoop_link origin:
    #   wrist_3→ft_frame: rpy=(pi,0,0), xyz=(0,0,0)
    #   wrist_3→flange:   rpy=(0,-pi/2,-pi/2), xyz=(0,0,0)
    #   flange→tool0:     rpy=(pi/2,0,pi/2), xyz=(0,0,0)
    #   tool0→scoop:      rpy=(pi/2,0,pi), xyz=(0,0,0)
    # Simplified: scoop_link origin ≈ at wrist_3_link with combined orientation
    _fixed_joint(stage, f"{J}/wrist3_scoop_joint",
                 body0=f"{ROOT}/wrist_3_link", body1=f"{ROOT}/scoop_link",
                 local_pos0=(0.0, 0.0, 0.0), local_rpy0=(PI / 2, 0.0, PI))

    stage.SetDefaultPrim(root_xform.GetPrim())
    stage.GetRootLayer().Save()
    print(f"[create_ur5_scoop_usd] Saved: {usd_path}")

    # Verify: list prims
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()}  [{prim.GetTypeName()}]")


if __name__ == "__main__":
    build()
