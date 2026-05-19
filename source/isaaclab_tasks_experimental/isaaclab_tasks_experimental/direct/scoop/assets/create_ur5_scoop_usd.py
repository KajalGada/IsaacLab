# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build ur5_with_scoop.usd as a PhysX articulation using only PyUSD (no Isaac Sim extension required).

Run once to generate the asset::

    ./isaaclab.sh -p .../direct/scoop/assets/create_ur5_scoop_usd.py

The resulting USD uses:
  - primitive collision shapes (capsules/spheres/cylinder) — no mesh dependency for physics
  - STL visual meshes with per-link colors embedded from the assets directory
  - mass/inertia values from the original URDF
  - drive gains matching the Newton reference script (ke=2000, kd=100)
  - a fixed joint from world to base_link so the robot is fixed-base
"""

import math
import os
import struct

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

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
# Per-link color palette (RGB, linear)
# ---------------------------------------------------------------------------

_LINK_COLORS: dict[str, tuple[float, float, float]] = {
    "base_link": (0.29, 0.29, 0.29),  # UR dark grey #4A4A4A
    "shoulder_link": (0.0, 0.56, 0.85),  # UR blue #009FE3
    "upper_arm_link": (0.29, 0.29, 0.29),  # UR dark grey
    "forearm_link": (0.29, 0.29, 0.29),  # UR dark grey
    "wrist_1_link": (0.0, 0.56, 0.85),  # UR blue
    "wrist_2_link": (0.0, 0.56, 0.85),  # UR blue
    "wrist_3_link": (0.0, 0.56, 0.85),  # UR blue
    "scoop_link": (0.52, 0.52, 0.52),  # light grey (aluminium)
}

# (stl_filename, trans_xyz_m, rpy_rad, uniform_scale)
# Transforms match the URDF <collision><origin> for each link.
# The scoop STL is in millimetres, so scale=0.001.
_LINK_STL_VISUALS: dict[str, tuple[str, tuple, tuple, float]] = {
    "base_link": ("base.stl", (0.0, 0.0, 0.0), (0.0, 0.0, PI), 1.0),
    "shoulder_link": ("shoulder.stl", (0.0, 0.0, 0.0), (0.0, 0.0, PI), 1.0),
    "upper_arm_link": ("upperarm.stl", (0.0, 0.0, 0.13585), (PI / 2, 0.0, -PI / 2), 1.0),
    "forearm_link": ("forearm.stl", (0.0, 0.0, 0.0165), (PI / 2, 0.0, -PI / 2), 1.0),
    "wrist_1_link": ("wrist1.stl", (0.0, 0.0, -0.093), (PI / 2, 0.0, 0.0), 1.0),
    "wrist_2_link": ("wrist2.stl", (0.0, 0.0, -0.095), (0.0, 0.0, 0.0), 1.0),
    "wrist_3_link": ("wrist3.stl", (0.0, 0.0, -0.0818), (PI / 2, 0.0, 0.0), 1.0),
    "scoop_link": ("ur5_scoop.stl", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.001),
}

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


def _add_link(
    stage: Usd.Stage, path: str, mass: float, diag_inertia: tuple, cog: tuple = (0.0, 0.0, 0.0)
) -> UsdGeom.Xform:
    xform = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag_inertia))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*cog))
    return xform


def _add_collision(prim: Usd.Prim) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)


def _add_capsule(
    stage: Usd.Stage,
    parent_path: str,
    name: str,
    radius: float,
    height: float,
    axis: str = "X",
    trans: tuple = (0.0, 0.0, 0.0),
) -> UsdGeom.Capsule:
    cap = UsdGeom.Capsule.Define(stage, f"{parent_path}/{name}")
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(height)
    cap.CreateAxisAttr(axis)
    cap.AddTranslateOp().Set(Gf.Vec3d(*trans))
    _add_collision(cap.GetPrim())
    return cap


def _add_sphere(
    stage: Usd.Stage, parent_path: str, name: str, radius: float, trans: tuple = (0.0, 0.0, 0.0)
) -> UsdGeom.Sphere:
    sph = UsdGeom.Sphere.Define(stage, f"{parent_path}/{name}")
    sph.CreateRadiusAttr(radius)
    sph.AddTranslateOp().Set(Gf.Vec3d(*trans))
    _add_collision(sph.GetPrim())
    return sph


def _add_cylinder(
    stage: Usd.Stage,
    parent_path: str,
    name: str,
    radius: float,
    height: float,
    axis: str = "Z",
    trans: tuple = (0.0, 0.0, 0.0),
) -> UsdGeom.Cylinder:
    cyl = UsdGeom.Cylinder.Define(stage, f"{parent_path}/{name}")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    cyl.AddTranslateOp().Set(Gf.Vec3d(*trans))
    _add_collision(cyl.GetPrim())
    return cyl


def _add_cube(
    stage: Usd.Stage, parent_path: str, name: str, size: tuple, trans: tuple = (0.0, 0.0, 0.0)
) -> UsdGeom.Cube:
    # USD Cube is a unit cube; scale via xformOp:scale
    cube = UsdGeom.Cube.Define(stage, f"{parent_path}/{name}")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.AddTranslateOp().Set(Gf.Vec3d(*trans))
    _add_collision(cube.GetPrim())
    return cube


def _revolute_joint(
    stage: Usd.Stage,
    path: str,
    body0: str,
    body1: str,
    local_pos0: tuple,
    local_rpy0: tuple,
    lower_deg: float,
    upper_deg: float,
    stiffness: float = 2000.0,
    damping: float = 100.0,
    effort_limit: float = 150.0,
) -> UsdPhysics.RevoluteJoint:
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


def _fixed_joint(
    stage: Usd.Stage,
    path: str,
    body0: str | None,
    body1: str,
    local_pos0: tuple = (0.0, 0.0, 0.0),
    local_rpy0: tuple = (0.0, 0.0, 0.0),
) -> UsdPhysics.FixedJoint:
    jnt = UsdPhysics.FixedJoint.Define(stage, path)
    if body0 is not None:
        jnt.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    jnt.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    jnt.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    jnt.CreateLocalRot0Attr(_rpy_to_quatf(*local_rpy0))
    jnt.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    jnt.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return jnt


def _read_binary_stl(path: str) -> tuple[list[Gf.Vec3f], list[int]]:
    """Parse a binary STL file and return (points, face_vertex_indices).

    Returns unshared vertices — 3 points per triangle in order — and sequential
    face_vertex_indices [0,1,2, 3,4,5, ...].  Scale is applied by the caller.
    """
    with open(path, "rb") as f:
        header = f.read(80)  # noqa: F841 — skip header
        (n_tri,) = struct.unpack_from("<I", f.read(4))
        raw = f.read(n_tri * 50)

    points: list[Gf.Vec3f] = []
    face_indices: list[int] = []
    offset = 0
    for i in range(n_tri):
        # 12 bytes normal (skip) + 36 bytes vertices + 2 bytes attribute (skip)
        offset += 12  # skip normal
        vx0, vy0, vz0 = struct.unpack_from("<fff", raw, offset)
        offset += 12
        vx1, vy1, vz1 = struct.unpack_from("<fff", raw, offset)
        offset += 12
        vx2, vy2, vz2 = struct.unpack_from("<fff", raw, offset)
        offset += 12
        offset += 2  # skip attribute byte count
        base = i * 3
        points.append(Gf.Vec3f(vx0, vy0, vz0))
        points.append(Gf.Vec3f(vx1, vy1, vz1))
        points.append(Gf.Vec3f(vx2, vy2, vz2))
        face_indices.extend([base, base + 1, base + 2])

    return points, face_indices


def _create_material(stage: Usd.Stage, mat_path: str, rgb: tuple) -> UsdShade.Material:
    """Create a UsdPreviewSurface material with the given diffuse color."""
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _add_visual_mesh(
    stage: Usd.Stage,
    parent_path: str,
    name: str,
    points: list[Gf.Vec3f],
    face_indices: list[int],
    mat: UsdShade.Material,
    trans: tuple = (0.0, 0.0, 0.0),
    rpy: tuple = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> UsdGeom.Mesh:
    """Create a visual-only mesh prim with a transform and bound material."""
    mesh = UsdGeom.Mesh.Define(stage, f"{parent_path}/{name}")
    n_tri = len(face_indices) // 3
    mesh.CreateFaceVertexCountsAttr([3] * n_tri)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    scaled = [Gf.Vec3f(p[0] * scale, p[1] * scale, p[2] * scale) for p in points]
    mesh.CreatePointsAttr(scaled)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    # Transform
    if trans != (0.0, 0.0, 0.0):
        mesh.AddTranslateOp().Set(Gf.Vec3d(*trans))
    if rpy != (0.0, 0.0, 0.0):
        mesh.AddOrientOp().Set(_rpy_to_quatf(*rpy))
    # Bind material
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh


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

    # Materials scope
    LOOKS = f"{ROOT}/Looks"
    UsdGeom.Scope.Define(stage, LOOKS)
    materials: dict[str, UsdShade.Material] = {}
    for link_name, rgb in _LINK_COLORS.items():
        mat_name = link_name.replace("_link", "") + "_mat"
        materials[link_name] = _create_material(stage, f"{LOOKS}/{mat_name}", rgb)

    # ------------------------------------------------------------------
    # Helper: add visual mesh from STL for a given link
    # ------------------------------------------------------------------

    def _attach_visual(link_name: str) -> None:
        stl_file, trans, rpy, scale = _LINK_STL_VISUALS[link_name]
        stl_path = os.path.join(_DIR, stl_file)
        if not os.path.isfile(stl_path):
            print(f"  [WARN] STL not found, skipping visual for {link_name}: {stl_path}")
            return
        pts, fidx = _read_binary_stl(stl_path)
        _add_visual_mesh(
            stage,
            f"{ROOT}/{link_name}",
            "visual",
            pts,
            fidx,
            materials[link_name],
            trans=trans,
            rpy=rpy,
            scale=scale,
        )

    # ------------------------------------------------------------------
    # Links — primitive collision shapes (invisible), URDF mass/inertia
    # ------------------------------------------------------------------

    # base_link_inertia (merged with base_link for simplicity)
    _add_link(stage, f"{ROOT}/base_link", mass=4.0, diag_inertia=(0.00443, 0.00443, 0.0072))
    col = _add_cylinder(stage, f"{ROOT}/base_link", "col", radius=0.06, height=0.05)
    UsdGeom.Imageable(col.GetPrim()).MakeInvisible()
    _attach_visual("base_link")

    # shoulder_link
    _add_link(
        stage,
        f"{ROOT}/shoulder_link",
        mass=3.7,
        diag_inertia=(0.01497, 0.01497, 0.01041),
        cog=(0.0, -0.00193, -0.02561),
    )
    col = _add_sphere(stage, f"{ROOT}/shoulder_link", "col", radius=0.055)
    UsdGeom.Imageable(col.GetPrim()).MakeInvisible()
    _attach_visual("shoulder_link")

    # upper_arm_link — capsule along X (arm extends in -X from shoulder)
    _add_link(
        stage,
        f"{ROOT}/upper_arm_link",
        mass=8.393,
        diag_inertia=(0.01511, 0.13389, 0.13389),
        cog=(-0.2125, 0.0, 0.11336),
    )
    col = _add_capsule(
        stage, f"{ROOT}/upper_arm_link", "col", radius=0.04, height=0.36, axis="X", trans=(-0.2125, 0.0, 0.13585)
    )
    UsdGeom.Imageable(col.GetPrim()).MakeInvisible()
    _attach_visual("upper_arm_link")

    # forearm_link — capsule along X
    _add_link(
        stage, f"{ROOT}/forearm_link", mass=2.33, diag_inertia=(0.004095, 0.03122, 0.03122), cog=(-0.24225, 0.0, 0.0265)
    )
    col = _add_capsule(
        stage, f"{ROOT}/forearm_link", "col", radius=0.035, height=0.32, axis="X", trans=(-0.196, 0.0, 0.0165)
    )
    UsdGeom.Imageable(col.GetPrim()).MakeInvisible()
    _attach_visual("forearm_link")

    # wrist_1_link
    _add_link(
        stage,
        f"{ROOT}/wrist_1_link",
        mass=1.219,
        diag_inertia=(0.002014, 0.002014, 0.002194),
        cog=(0.0, -0.01634, -0.0018),
    )
    col = _add_sphere(stage, f"{ROOT}/wrist_1_link", "col", radius=0.038, trans=(0.0, 0.0, -0.093))
    UsdGeom.Imageable(col.GetPrim()).MakeInvisible()
    _attach_visual("wrist_1_link")

    # wrist_2_link
    _add_link(
        stage,
        f"{ROOT}/wrist_2_link",
        mass=1.219,
        diag_inertia=(0.001831, 0.001831, 0.002194),
        cog=(0.0, 0.01634, -0.0018),
    )
    col = _add_sphere(stage, f"{ROOT}/wrist_2_link", "col", radius=0.038, trans=(0.0, 0.0, -0.095))
    UsdGeom.Imageable(col.GetPrim()).MakeInvisible()
    _attach_visual("wrist_2_link")

    # wrist_3_link
    _add_link(
        stage, f"{ROOT}/wrist_3_link", mass=0.1879, diag_inertia=(8.06e-5, 8.06e-5, 1.32e-4), cog=(0.0, 0.0, -0.001159)
    )
    col = _add_sphere(stage, f"{ROOT}/wrist_3_link", "col", radius=0.032, trans=(0.0, 0.0, -0.082))
    UsdGeom.Imageable(col.GetPrim()).MakeInvisible()
    _attach_visual("wrist_3_link")

    # scoop_link — box covering the full scoop geometry measured from ur5_scoop.stl (mm→m, scale=0.001):
    #   X: ±3.75 cm (symmetric), Y: 0 → 24.9 cm (blade extends along scoop_link +Y),
    #   Z: ±3.75 cm (symmetric).  Centre: (0, 0.1245, 0) in scoop_link frame.
    # The fixed joint local_rpy0=(π/2, 0, π) maps scoop_link Y → wrist_3 Z, so the
    # 24.9 cm blade length aligns with the tool approach axis.
    _SCOOP_CX, _SCOOP_CY, _SCOOP_CZ = 0.0, 0.1245, 0.0
    _SCOOP_SX, _SCOOP_SY, _SCOOP_SZ = 0.075, 0.249, 0.075
    _add_link(
        stage,
        f"{ROOT}/scoop_link",
        mass=0.5,
        diag_inertia=(0.003, 0.001, 0.003),
        cog=(_SCOOP_CX, _SCOOP_CY, _SCOOP_CZ),
    )
    col = _add_cube(
        stage,
        f"{ROOT}/scoop_link",
        "col",
        size=(_SCOOP_SX, _SCOOP_SY, _SCOOP_SZ),
        trans=(_SCOOP_CX, _SCOOP_CY, _SCOOP_CZ),
    )
    UsdGeom.Imageable(col.GetPrim()).MakeInvisible()
    _attach_visual("scoop_link")

    # ------------------------------------------------------------------
    # Joints
    # ------------------------------------------------------------------

    # World → base_link: fix the robot to the world
    _fixed_joint(stage, f"{J}/world_to_base", body0=None, body1=f"{ROOT}/base_link")

    # shoulder_pan (base_link → shoulder_link): z offset = d1
    _revolute_joint(
        stage,
        f"{J}/shoulder_pan_joint",
        f"{ROOT}/base_link",
        f"{ROOT}/shoulder_link",
        local_pos0=(0.0, 0.0, D1),
        local_rpy0=(0.0, 0.0, 0.0),
        lower_deg=-360.0,
        upper_deg=360.0,
        effort_limit=150.0,
    )

    # shoulder_lift (shoulder_link → upper_arm_link): rpy = (pi/2, 0, 0)
    _revolute_joint(
        stage,
        f"{J}/shoulder_lift_joint",
        f"{ROOT}/shoulder_link",
        f"{ROOT}/upper_arm_link",
        local_pos0=(0.0, 0.0, 0.0),
        local_rpy0=(PI / 2, 0.0, 0.0),
        lower_deg=-360.0,
        upper_deg=360.0,
        effort_limit=150.0,
    )

    # elbow (upper_arm_link → forearm_link): x offset = a2
    _revolute_joint(
        stage,
        f"{J}/elbow_joint",
        f"{ROOT}/upper_arm_link",
        f"{ROOT}/forearm_link",
        local_pos0=(A2, 0.0, 0.0),
        local_rpy0=(0.0, 0.0, 0.0),
        lower_deg=-180.0,
        upper_deg=180.0,
        effort_limit=150.0,
    )

    # wrist_1 (forearm_link → wrist_1_link): (a3, 0, d4)
    _revolute_joint(
        stage,
        f"{J}/wrist_1_joint",
        f"{ROOT}/forearm_link",
        f"{ROOT}/wrist_1_link",
        local_pos0=(A3, 0.0, D4),
        local_rpy0=(0.0, 0.0, 0.0),
        lower_deg=-360.0,
        upper_deg=360.0,
        effort_limit=28.0,
    )

    # wrist_2 (wrist_1_link → wrist_2_link): (0, -d5, 0), rpy=(pi/2, 0, 0)
    _revolute_joint(
        stage,
        f"{J}/wrist_2_joint",
        f"{ROOT}/wrist_1_link",
        f"{ROOT}/wrist_2_link",
        local_pos0=(0.0, -D5, 0.0),
        local_rpy0=(PI / 2, 0.0, 0.0),
        lower_deg=-360.0,
        upper_deg=360.0,
        effort_limit=28.0,
    )

    # wrist_3 (wrist_2_link → wrist_3_link): (0, d6, 0), rpy=(pi/2, pi, pi)
    _revolute_joint(
        stage,
        f"{J}/wrist_3_joint",
        f"{ROOT}/wrist_2_link",
        f"{ROOT}/wrist_3_link",
        local_pos0=(0.0, D6, 0.0),
        local_rpy0=(PI / 2, PI, PI),
        lower_deg=-360.0,
        upper_deg=360.0,
        effort_limit=28.0,
    )

    # wrist_3 → scoop_link: combined fixed transform through flange/tool0
    # Net transform from wrist_3_link to scoop_link origin:
    #   wrist_3→ft_frame: rpy=(pi,0,0), xyz=(0,0,0)
    #   wrist_3→flange:   rpy=(0,-pi/2,-pi/2), xyz=(0,0,0)
    #   flange→tool0:     rpy=(pi/2,0,pi/2), xyz=(0,0,0)
    #   tool0→scoop:      rpy=(pi/2,0,pi), xyz=(0,0,0)
    # Simplified: scoop_link origin ≈ at wrist_3_link with combined orientation
    _fixed_joint(
        stage,
        f"{J}/wrist3_scoop_joint",
        body0=f"{ROOT}/wrist_3_link",
        body1=f"{ROOT}/scoop_link",
        local_pos0=(0.0, 0.0, 0.0),
        local_rpy0=(PI / 2, 0.0, PI),
    )

    stage.SetDefaultPrim(root_xform.GetPrim())
    stage.GetRootLayer().Save()
    print(f"[create_ur5_scoop_usd] Saved: {usd_path}")

    # Verify: list prims
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()}  [{prim.GetTypeName()}]")


if __name__ == "__main__":
    build()
