"""Build the AstroNet-v2 3D visualization in Blender.

Usage (Blender background)::

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python Visuals/blender_build.py

The script reads ``Visuals/network_data.json`` (created by
``Visuals/build_network_data.py``), rebuilds the scene from scratch and saves
``Visuals/satai_network_3d.blend``.  It intentionally uses only bpy/bmesh/
mathutils and Python's standard library so it can also run inside Blender
without the project's virtualenv.

Scene vocabulary
----------------
* Y is the depth axis: input at Y=0, output at Y=100.
* X/Z are the image plane; a filter is a translucent volume with a 3x3
  micro-grid of its real averaged kernel weights.
* Fibers are routed as ordered rectangular buses; their thickness and colour
  encode the signed strength of the strongest real weight links.
* Activations are tiny octahedra, one for every post-ReLU/post-pool node.
  They are not random particles: their position and brightness come from the
  captured tensor.
"""

from __future__ import annotations

import base64
import json
import math
import random
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Visuals" / "network_data.json"
OUT_BLEND = ROOT / "Visuals" / "satai_network_3d.blend"
OUT_FRAMES = ROOT / "Visuals" / "renders"
OUT_HEAD = ROOT / "Visuals" / "satai_network_3d_head.png"
OUT_POSTER = ROOT / "Visuals" / "satai_network_3d_wide.png"

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
Y_INPUT = 0.0
BLOCKS = {
    "conv1": dict(y=14.5, w=10.6, h=10.6, d=2.8, color="#2f7bff", layout=(6, 6)),
    "conv2": dict(y=31.5, w=9.2, h=9.2, d=3.0, color="#38ff9a", layout=(8, 8)),
    "conv3": dict(y=49.5, w=7.8, h=7.8, d=3.2, color="#ffe14a", layout=(12, 11)),
    "conv4": dict(y=68.0, w=6.7, h=6.7, d=3.6, color="#ff7a1a", layout=(16, 16)),
}

#: Interior node/filter colours, deliberately different from the saturated
#: neon frame of their block. The frame says which layer you are in; the
#: interior says what the filter tensor is doing.
INNER = {
    "conv1": "#bdf4ff",
    "conv2": "#b9ffd8",
    "conv3": "#ffe6a0",
    "conv4": "#ffb6d8",
}

#: The filter bodies are deliberately dark complementary jewels, so the
#: saturated frame tells the layer while the module itself reads as a solid
#: physical object. Only the real 3x3 kernel dots keep the layer pastel.
MODULE = {
    "conv1": "#6d1c3f",
    "conv2": "#521a6e",
    "conv3": "#262a7d",
    "conv4": "#0c5a64",
}
GAP_Y = 79.4
GAP_W = 14.6
DENSE1_Y = 88.8
DENSE1_W, DENSE1_H = 10.4, 4.8
OUT_Y = 96.6
OUT_X = 8.0
N_FRAMES = 216
FPS = 24

CAM_START = Vector((112.0, -62.0, 55.0))
CAM_MID = Vector((110.0, -60.0, 53.0))
CAM_END = Vector((108.0, -58.0, 52.0))
# Aim slightly below the network centre: the old wide shot left a large empty
# band at the top and almost no room under the input for its captions.
TARGET = Vector((0.0, 49.0, -4.8))


# ---------------------------------------------------------------------------
# Colour / material helpers
# ---------------------------------------------------------------------------
def srgb(hex_colour: str) -> tuple[float, float, float]:
    """Convert #rrggbb to linear RGB."""
    h = hex_colour.lstrip("#")
    vals = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(v ** 2.2 for v in vals)


def mix(a: tuple[float, float, float], b: tuple[float, float, float],
        t: float) -> tuple[float, float, float]:
    return tuple(a[i] * (1.0 - t) + b[i] * t for i in range(3))


def emission_mat(name: str, colour: str | tuple[float, float, float],
                 strength: float = 2.0) -> tuple[bpy.types.Material, bpy.types.ShaderNode]:
    """Create an emissive material and return it with its Emission node."""
    c = srgb(colour) if isinstance(colour, str) else colour
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.name = "Emission"
    em.inputs["Color"].default_value = (*c, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    mat.diffuse_color = (*c, 1.0)
    return mat, em


def vertex_emission_mat(name: str, strength: float = 2.0
                        ) -> tuple[bpy.types.Material, bpy.types.ShaderNode]:
    """Emission material whose colour is read from the mesh 'Col' attribute."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Col"
    em = nt.nodes.new("ShaderNodeEmission")
    em.name = "Emission"
    em.inputs["Strength"].default_value = strength
    nt.links.new(vc.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    mat.diffuse_color = (0.2, 0.6, 1.0, 1.0)
    return mat, em


def module_mat(name: str, colour: str, strength: float = 0.35
               ) -> tuple[bpy.types.Material, bpy.types.ShaderNode]:
    """A dark, slightly metallic filter body with a controllable glow."""
    c = srgb(colour)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*c, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.45
    bsdf.inputs["Roughness"].default_value = 0.26
    bsdf.inputs["Alpha"].default_value = 0.88
    em = bsdf.inputs["Emission Color"]
    em.default_value = (*c, 1.0)
    em_str = bsdf.inputs["Emission Strength"]
    em_str.default_value = strength
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat.use_backface_culling = False
    for prop, value in (("surface_render_method", "DITHERED"),
                        ("blend_method", "BLEND"),
                        ("use_transparent_shadow", False)):
        try:
            setattr(mat, prop, value)
        except Exception:
            pass
    mat.diffuse_color = (*c, 0.9)
    return mat, em_str


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def mesh_obj(name: str, verts: list, faces: list, colour: tuple | None = None,
             material: bpy.types.Material | None = None,
             collection: bpy.types.Collection | None = None) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    if colour is not None:
        attr = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
        for i in range(len(verts)):
            attr.data[i].color = (*colour, 1.0)
    if material is not None:
        me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def octahedra_mesh(name: str, points: list[tuple[Vector, float, tuple[float, float, float]]],
                   material: bpy.types.Material, collection: bpy.types.Collection) -> bpy.types.Object:
    """One mesh containing one low-poly octahedron per activation/node."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for pos, size, _colour in points:
        b = len(verts)
        x, y, z = pos
        s = size
        verts += [
            (x, y, z + s), (x - s, y, z), (x, y + s, z),
            (x + s, y, z), (x, y - s, z), (x, y, z - s),
        ]
        t, l, bk, r, fr, bo = b, b + 1, b + 2, b + 3, b + 4, b + 5
        faces += [
            (t, l, bk), (t, bk, r), (t, r, fr), (t, fr, l),
            (bo, bk, l), (bo, r, bk), (bo, fr, r), (bo, l, fr),
        ]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    attr = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    for i, (_pos, _size, colour) in enumerate(points):
        for j in range(6):
            attr.data[i * 6 + j].color = (*colour, 1.0)
    me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


def plate_mesh(name: str, center: tuple[float, float, float],
               sx: float, sy: float, sz: float,
               colour: tuple[float, float, float],
               material: bpy.types.Material,
               collection: bpy.types.Collection) -> bpy.types.Object:
    """One thin rectangular plate with independent X/Y/Z dimensions."""
    x, y, z = center
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [
        (x - hx, y - hy, z - hz), (x + hx, y - hy, z - hz),
        (x + hx, y - hy, z + hz), (x - hx, y - hy, z + hz),
        (x - hx, y + hy, z - hz), (x + hx, y + hy, z - hz),
        (x + hx, y + hy, z + hz), (x - hx, y + hy, z + hz),
    ]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_obj(name, verts, faces, colour, material, collection)


def rod_mesh(name: str, length: float, thickness_y: float, thickness_z: float,
             colour: tuple[float, float, float], material: bpy.types.Material,
             collection: bpy.types.Collection) -> bpy.types.Object:
    """A glowing horizontal rod from x=0 to x=length (origin at its left end)."""
    hy, hz = thickness_y / 2, thickness_z / 2
    verts = [(0, -hy, -hz), (length, -hy, -hz), (length, -hy, hz),
             (0, -hy, hz), (0, hy, -hz), (length, hy, -hz),
             (length, hy, hz), (0, hy, hz)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_obj(name, verts, faces, colour, material, collection)


def cube_mesh(name: str, boxes: list[tuple[Vector, float, float, tuple[float, float, float]]],
              material: bpy.types.Material, collection: bpy.types.Collection) -> bpy.types.Object:
    """One mesh containing many axis-aligned cubes.

    Each box is ``(center, size_xz, height_y, colour)``; the box extends from
    center.y - height/2 to center.y + height/2.
    """
    verts = []
    faces = []
    for center, sxz, sy, _colour in boxes:
        x, y, z = center
        h = sxz / 2.0
        b = len(verts)
        verts += [
            (x - h, y - sy / 2, z - h), (x + h, y - sy / 2, z - h),
            (x + h, y - sy / 2, z + h), (x - h, y - sy / 2, z + h),
            (x - h, y + sy / 2, z - h), (x + h, y + sy / 2, z - h),
            (x + h, y + sy / 2, z + h), (x - h, y + sy / 2, z + h),
        ]
        faces += [
            (b, b + 3, b + 2, b + 1), (b + 4, b + 5, b + 6, b + 7),
            (b, b + 1, b + 5, b + 4), (b + 1, b + 2, b + 6, b + 5),
            (b + 2, b + 3, b + 7, b + 6), (b + 3, b, b + 4, b + 7),
        ]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    attr = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    for i, (_center, _sxz, _sy, colour) in enumerate(boxes):
        for j in range(8):
            attr.data[i * 8 + j].color = (*colour, 1.0)
    me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


def cylinder_mesh(name: str, bars: list[tuple[float, float, float, float, float,
                                              tuple[float, float, float]]],
                  material: bpy.types.Material, collection: bpy.types.Collection,
                  segments: int = 12) -> bpy.types.Object:
    """Bars are ``(x, y, z_base, height, radius, colour)``."""
    verts = []
    faces = []
    for x, y, z0, height, radius, _colour in bars:
        b = len(verts)
        for j in range(segments):
            a = 2.0 * math.pi * j / segments
            verts.append((x + radius * math.cos(a), y + radius * math.sin(a), z0))
        for j in range(segments):
            a = 2.0 * math.pi * j / segments
            verts.append((x + radius * math.cos(a), y + radius * math.sin(a), z0 + height))
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((b + j, b + k, b + segments + k, b + segments + j))
        faces.append(tuple(b + j for j in range(segments - 1, -1, -1)))
        faces.append(tuple(b + segments + j for j in range(segments)))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    attr = me.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    for i, (_x, _y, _z, _h, _r, colour) in enumerate(bars):
        for j in range(segments * 2):
            attr.data[i * segments * 2 + j].color = (*colour, 1.0)
    me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


def curve_object(name: str, splines: list[list[tuple[float, float, float]]],
                 bevel: float, material: bpy.types.Material,
                 collection: bpy.types.Collection, cyclic: bool = False,
                 radii: list[float] | None = None) -> bpy.types.Object:
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.resolution_u = 1
    cu.bevel_depth = bevel
    cu.bevel_resolution = 1
    cu.use_fill_caps = True
    for si, coords in enumerate(splines):
        sp = cu.splines.new("POLY")
        sp.points.add(len(coords) - 1)
        for i, co in enumerate(coords):
            sp.points[i].co = (*co, 1.0)
            if radii:
                sp.points[i].radius = radii[si]
        sp.use_cyclic_u = cyclic
    cu.materials.append(material)
    ob = bpy.data.objects.new(name, cu)
    collection.objects.link(ob)
    return ob


def fiber_curve_object(name: str, splines: list[tuple[list[tuple[float, float, float]],
                                                      int, float]],
                       bevel: float, pos_mat: bpy.types.Material,
                       neg_mat: bpy.types.Material,
                       collection: bpy.types.Collection) -> bpy.types.Object:
    """Create bezier fibers; each spline has (points, material_index, radius)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.resolution_u = 1
    cu.bevel_depth = bevel
    cu.bevel_resolution = 1
    cu.use_fill_caps = True
    # Materials must exist before the per-spline material_index is set:
    # appending slots afterwards silently resets every spline to index 0
    # (which is why the first render had no visible negative weights).
    cu.materials.append(pos_mat)
    cu.materials.append(neg_mat)
    for coords, mat_idx, radius in splines:
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(coords) - 1)
        n = max(1, len(coords) - 1)
        for i, co in enumerate(coords):
            bp = sp.bezier_points[i]
            bp.co = co
            bp.handle_left_type = "AUTO"
            bp.handle_right_type = "AUTO"
            # Optical-fibre taper: a thin core at both ends, a slightly
            # thicker luminous waist in the middle.
            taper = 0.55 + 0.62 * math.sin(math.pi * i / n)
            bp.radius = radius * taper
        sp.material_index = mat_idx
        sp.use_cyclic_u = False
    ob = bpy.data.objects.new(name, cu)
    collection.objects.link(ob)
    return ob


def add_frame(name: str, y: float, w: float, h: float, d: float, colour: str,
              collection: bpy.types.Collection, bevel: float = 0.055,
              strength: float = 2.4) -> tuple[bpy.types.Object, bpy.types.ShaderNode]:
    mat, em = emission_mat(f"{name}_frame", colour, strength)
    lines = []
    for yy in (y - d / 2, y + d / 2):
        pts = [(-w / 2, yy, -h / 2), (w / 2, yy, -h / 2),
               (w / 2, yy, h / 2), (-w / 2, yy, h / 2)]
        for i in range(4):
            lines.append([pts[i], pts[(i + 1) % 4]])
    for sx, sz in ((-w / 2, -h / 2), (w / 2, -h / 2),
                   (w / 2, h / 2), (-w / 2, h / 2)):
        lines.append([(sx, y - d / 2, sz), (sx, y + d / 2, sz)])
    ob = curve_object(f"{name}_wire", lines, bevel, mat, collection)
    return ob, em


# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------
def key_socket(socket, frames: list[tuple[int, float]]) -> None:
    for frame, value in frames:
        socket.default_value = value
        socket.keyframe_insert(data_path="default_value", frame=frame)


def key_growth(ob: bpy.types.Object, start: int, end: int) -> None:
    cu = ob.data
    try:
        cu.bevel_factor_mapping_start = "SPLINE"
        cu.bevel_factor_mapping_end = "SPLINE"
    except Exception:
        pass
    cu.bevel_factor_end = 0.0
    cu.keyframe_insert(data_path="bevel_factor_end", frame=max(1, start - 1))
    cu.bevel_factor_end = 1.0
    cu.keyframe_insert(data_path="bevel_factor_end", frame=end)


def key_scale(ob: bpy.types.Object, start: int, end: int,
              start_value: float = 0.001) -> None:
    ob.scale = (start_value, start_value, start_value)
    ob.keyframe_insert(data_path="scale", frame=start)
    ob.scale = (1.0, 1.0, 1.0)
    ob.keyframe_insert(data_path="scale", frame=end)


def key_grow_x(ob: bpy.types.Object, start: int, end: int) -> None:
    """Grow a horizontal rod from its left origin without moving the labels."""
    ob.scale = (0.001, 1.0, 1.0)
    ob.keyframe_insert(data_path="scale", frame=start)
    ob.scale = (1.0, 1.0, 1.0)
    ob.keyframe_insert(data_path="scale", frame=end)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------
def build() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = N_FRAMES
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(OUT_FRAMES / "frame_")
    scene.render.film_transparent = False
    try:
        scene.eevee.taa_render_samples = 32
    except Exception:
        pass
    try:
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        pass
    scene.view_settings.exposure = 0.45

    # World: procedural deep-space nebula. No geometry edge, so the camera
    # can push in without ever revealing a horizon or a backdrop border.
    world = bpy.data.worlds.new("SatAi Space")
    world.use_nodes = True
    wnt = world.node_tree
    wnt.nodes.clear()
    wout = wnt.nodes.new("ShaderNodeOutputWorld")
    wbg = wnt.nodes.new("ShaderNodeBackground")
    wtex = wnt.nodes.new("ShaderNodeTexCoord")
    wnoise = wnt.nodes.new("ShaderNodeTexNoise")
    wnoise.inputs["Scale"].default_value = 2.6
    wnoise.inputs["Detail"].default_value = 7.0
    wnoise.inputs["Roughness"].default_value = 0.62
    wnoise.inputs["Distortion"].default_value = 0.55
    wramp = wnt.nodes.new("ShaderNodeValToRGB")
    stops = ((0.0, (0.0008, 0.0018, 0.0050, 1.0)),
             (0.38, (0.0025, 0.0080, 0.0180, 1.0)),
             (0.58, (0.0080, 0.0300, 0.0520, 1.0)),
             (0.72, (0.0150, 0.0520, 0.0780, 1.0)),
             (1.0, (0.0010, 0.0030, 0.0080, 1.0)))
    wramp.color_ramp.elements[0].position = stops[0][0]
    wramp.color_ramp.elements[0].color = stops[0][1]
    wramp.color_ramp.elements[1].position = stops[1][0]
    wramp.color_ramp.elements[1].color = stops[1][1]
    for pos, col in stops[2:]:
        e = wramp.color_ramp.elements.new(pos)
        e.color = col
    wbg.inputs["Strength"].default_value = 0.62
    wnt.links.new(wtex.outputs["Window"], wnoise.inputs["Vector"])
    wnt.links.new(wnoise.outputs["Fac"], wramp.inputs["Fac"])
    wnt.links.new(wramp.outputs["Color"], wbg.inputs["Color"])
    wnt.links.new(wbg.outputs[0], wout.inputs["Surface"])
    scene.world = world

    root = bpy.data.collections.new("AstroNet_v2")
    scene.collection.children.link(root)

    def sub(name: str) -> bpy.types.Collection:
        c = bpy.data.collections.new(name)
        root.children.link(c)
        return c

    c_input = sub("01_Input")
    c_filters = sub("02_Filters")
    c_activations = sub("03_Activations")
    c_fibers = sub("04_Fibers")
    c_head = sub("05_Head")
    c_annotations = sub("06_Annotations")
    c_environment = sub("07_Environment")

    # Camera and tracking target.
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = CAM_START
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 92.0
    target = bpy.data.objects.new("Camera_Target", None)
    scene.collection.objects.link(target)
    target.location = TARGET
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    scene.camera = cam

    # The animation starts wide, then closes on the head so the nine class
    # labels are actually legible at the end. The orthographic lens avoids
    # the output head shrinking into the distance.
    cam.location = CAM_START
    cam.data.ortho_scale = 92.0
    cam.keyframe_insert(data_path="location", frame=1)
    cam_data.keyframe_insert(data_path="ortho_scale", frame=1)
    cam.location = CAM_MID
    cam.keyframe_insert(data_path="location", frame=150)
    cam.data.ortho_scale = 88.0
    cam_data.keyframe_insert(data_path="ortho_scale", frame=150)
    cam.location = Vector((98.0, -54.0, 50.0))
    cam.keyframe_insert(data_path="location", frame=178)
    cam.data.ortho_scale = 56.0
    cam_data.keyframe_insert(data_path="ortho_scale", frame=178)
    cam.location = Vector((95.0, -48.0, 45.0))
    cam.keyframe_insert(data_path="location", frame=N_FRAMES)
    cam.data.ortho_scale = 50.0
    cam_data.keyframe_insert(data_path="ortho_scale", frame=N_FRAMES)
    target.location = TARGET
    target.keyframe_insert(data_path="location", frame=1)
    target.location = Vector((0.0, 53.0, -4.8))
    target.keyframe_insert(data_path="location", frame=150)
    target.location = Vector((4.5, 86.0, -1.6))
    target.keyframe_insert(data_path="location", frame=178)
    target.location = Vector((5.0, 88.0, -1.6))
    target.keyframe_insert(data_path="location", frame=N_FRAMES)

    # A separate still camera keeps the whole network in frame for the poster.
    hero_data = bpy.data.cameras.new("Camera_Hero")
    hero = bpy.data.objects.new("Camera_Hero", hero_data)
    scene.collection.objects.link(hero)
    hero.location = CAM_START
    hero.data.type = "ORTHO"
    hero.data.ortho_scale = 92.0
    hero_target = bpy.data.objects.new("Camera_Hero_Target", None)
    scene.collection.objects.link(hero_target)
    hero_target.location = TARGET
    hero_track = hero.constraints.new("TRACK_TO")
    hero_track.target = hero_target
    hero_track.track_axis = "TRACK_NEGATIVE_Z"
    hero_track.up_axis = "UP_Y"

    # Area lights reveal the dark filter bodies without flattening the glow.
    for name, loc, colour, energy, size in (
        ("Key_Soft", (-14.0, -58.0, 46.0), (1.0, 0.93, 0.82), 1350.0, 30.0),
        ("Rim_Cyan", (34.0, -24.0, 26.0), (0.16, 0.52, 1.0), 700.0, 22.0),
        ("Rim_Amber", (-30.0, 86.0, 36.0), (1.0, 0.36, 0.10), 620.0, 20.0),
        ("Fill_Blue", (36.0, 34.0, -1.0), (0.10, 0.30, 0.85), 380.0, 26.0),
    ):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy
        ld.color = colour
        ld.shape = "DISK"
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(lo)
        lo.location = loc
        # Point the area light at the center of the network.
        d = Vector((0.0, 48.0, 0.0)) - Vector(loc)
        lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

    # ---------------------------------------------------------------
    # Environment: gradient backdrop + dark studio floor with a thin grid.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Input: 64x64 actual pixels, extruded toward the camera.
    # ---------------------------------------------------------------
    pixels = base64.b64decode(data["input"]["pixels"])
    if len(pixels) != 64 * 64:
        raise ValueError("input pixel count mismatch")
    voxels = []
    pitch = 0.19
    # Almost touching cubes: the 64x64 block must read as one populated
    # volume, not as a grid with dark holes between the bright pixels.
    cube = 0.184
    for row in range(64):
        for col in range(64):
            v = pixels[row * 64 + col] / 255.0
            t = v ** 0.85
            if t < 0.025:
                colour = (0.018, 0.045, 0.095)
            elif t < 0.35:
                colour = mix((0.025, 0.09, 0.18), (0.04, 0.48, 0.72), t / 0.35)
            elif t < 0.82:
                colour = mix((0.04, 0.48, 0.72), (1.0, 0.54, 0.16), (t - 0.35) / 0.47)
            else:
                colour = mix((1.0, 0.54, 0.16), (1.0, 0.95, 0.78), (t - 0.82) / 0.18)
            x = (col - 31.5) * pitch
            z = (31.5 - row) * pitch
            # Every pixel is a real extruded voxel. Even a black pixel gets
            # a small dark-blue body, so the input silhouette never breaks.
            height = 0.10 + 0.69 * (t ** 1.35)
            voxels.append((Vector((x, -height / 2.0, z)), cube, height, colour))
    input_mat, input_em = vertex_emission_mat("Input_Emission", 2.5)
    cube_mesh("Input_64x64", voxels, input_mat, c_input)

    # Dark base plate + neon input frame.
    base_mat, base_em = vertex_emission_mat("Input_Base_Mat", 0.45)
    plate_mesh("Input_Base", (0, 0.045, 0), 12.9, 0.05, 12.9,
               (0.004, 0.012, 0.03), base_mat, c_input)
    input_frame, input_frame_em = add_frame("INPUT", 0.0, 12.9, 12.9, 0.16,
                                            "#8fe9ff", c_input, bevel=0.035,
                                            strength=1.7)

    # ---------------------------------------------------------------
    # Filters and their real 3x3 kernel micro-grids.
    # ---------------------------------------------------------------
    centres: dict[str, list[Vector]] = {}
    block_objects: dict[str, dict] = {}
    for block in data["blocks"]:
        name = block["name"]
        spec = BLOCKS[name]
        y, w, h, d = spec["y"], spec["w"], spec["h"], spec["d"]
        cols, rows = spec["layout"]
        channels = block["shape"][0]
        plate = min(w / cols, h / rows) * 0.74
        # Deeper volumes make the three real input-channel slices visible;
        # the 3x3 dot card inside each slice is the actual kernel, not a glyph.
        depth = min(d * 0.55, plate * 1.15)
        inner_hex = INNER[name]
        module_hex = MODULE[name]
        inner_col = srgb(inner_hex)
        card_mat, card_em = module_mat(f"{name}_Module", module_hex, 0.70)
        outline_mat, outline_em = emission_mat(f"{name}_Outline", inner_hex, 1.35)
        dot_mat, dot_em = vertex_emission_mat(f"{name}_Dots", 2.1)

        card_boxes = []
        post_boxes = []
        outline_lines = []
        centres[name] = []
        for i in range(channels):
            row = i // cols
            col = i % cols
            x = -w / 2 + (col + 0.5) * w / cols
            z = h / 2 - (row + 0.5) * h / rows
            centres[name].append(Vector((x, y, z)))
            # Four thin posts bind the three channel-slices into one filter
            # block, so it reads as a stack instead of loose floating tiles.
            hp = plate / 2
            post = max(0.007, plate * 0.020)
            for px, pz in ((-hp, -hp), (hp, -hp), (hp, hp), (-hp, hp)):
                post_boxes.append((Vector((x + px, y, z + pz)), post, depth,
                                   (0.2, 0.4, 0.8)))
        cube_mesh(f"{name}_FilterPosts", post_boxes, card_mat, c_filters)

        # Three thin cards through the filter volume: low/mid/high input
        # channel groups. Each carries its own real 3x3 kernel slice.
        dot_points = []
        kernel_ceiling = max(
            max(abs(k) for sl in f.get("slices", [f["kernel"]]) for k in sl)
            for f in block["filters"]
        )
        kernel_ceiling = max(kernel_ceiling, 1e-6)
        dot_radius = max(0.008, plate * 0.034)
        for i, filt in enumerate(block["filters"]):
            cx, cy, cz = centres[name][i]
            slices = filt.get("slices") or [filt["kernel"]]
            n_slices = len(slices)
            for si, slice_kernel in enumerate(slices):
                offset = (si - (n_slices - 1) / 2.0) * depth * 0.40
                card_depth = max(0.032, depth * 0.115)
                card_boxes.append((Vector((cx, cy + offset, cz)),
                                   plate * 0.94, card_depth, (0.2, 0.4, 0.8)))
                yy = cy + offset - card_depth / 2 - 0.014
                hp = plate * 0.45
                outline_lines.append([(cx - hp, yy, cz - hp),
                                      (cx + hp, yy, cz - hp),
                                      (cx + hp, yy, cz + hp),
                                      (cx - hp, yy, cz + hp),
                                      (cx - hp, yy, cz - hp)])
                spacing = plate * 0.255
                for ki, kval in enumerate(slice_kernel):
                    kr, kc = divmod(ki, 3)
                    dx = (kc - 1) * spacing
                    dz = (1 - kr) * spacing
                    mag = min(1.0, abs(kval) / kernel_ceiling)
                    if kval >= 0:
                        dot_col = mix(inner_col, (1.0, 0.98, 0.90),
                                      0.30 + 0.55 * mag)
                    else:
                        dot_col = mix((0.55, 0.02, 0.22), (1.0, 0.20, 0.55), mag)
                    dot_points.append((Vector((cx + dx, yy, cz + dz)),
                                       dot_radius * (0.62 + 0.9 * mag), dot_col))
        cube_mesh(f"{name}_KernelSlices", card_boxes, card_mat, c_filters)
        curve_object(f"{name}_FilterOutlines", outline_lines, 0.0055,
                     outline_mat, c_filters)
        octahedra_mesh(f"{name}_KernelDots", dot_points, dot_mat, c_filters)

        frame_obj, frame_em = add_frame(name.upper(), y, w, h, d, spec["color"],
                                        c_filters, bevel=0.055 if name != "conv4" else 0.075,
                                        strength=2.2 if name != "conv4" else 3.2)
        block_objects[name] = {
            "card_em": card_em, "outline_em": outline_em, "dot_em": dot_em,
            "frame_em": frame_em, "frame": frame_obj,
        }

    # ---------------------------------------------------------------
    # Activation octahedra: every post-pool node, actual values.
    # ---------------------------------------------------------------
    point_objects = {}
    for block in data["blocks"]:
        name = block["name"]
        spec = BLOCKS[name]
        y, w, h, d = spec["y"], spec["w"], spec["h"], spec["d"]
        c, hh, ww = block["shape"]
        values = base64.b64decode(block["activation"])
        if len(values) != c * hh * ww:
            raise ValueError(f"{name}: activation length mismatch")
        layer_col = srgb(INNER[name])
        pts = []
        for ci in range(c):
            yc = y + (ci / max(1, c - 1) - 0.5) * d * 0.64 + d * 0.10
            for zi in range(hh):
                for xi in range(ww):
                    a = values[ci * hh * ww + zi * ww + xi] / 255.0
                    if a < 0.070:
                        continue
                    x = -w / 2 + (xi + 0.5) / ww * w * 0.90
                    z = h / 2 - (zi + 0.5) / hh * h * 0.90
                    col = mix(layer_col, (1.0, 0.97, 0.86), 0.05 + 0.30 * (a ** 1.5))
                    size = 0.007 + 0.018 * (a ** 0.75)
                    pts.append((Vector((x, yc, z)), size, col))
        mat, em = vertex_emission_mat(f"{name}_Activation", 1.05 if name != "conv4" else 1.40)
        ob = octahedra_mesh(f"{name}_ActivationField", pts, mat, c_activations)
        point_objects[name] = (ob, em)

    # ---------------------------------------------------------------
    # GAP: 256 channels collapsed into one luminous comb.
    # ---------------------------------------------------------------
    gap_values = base64.b64decode(data["gap"]["activation"])
    gap_bars = []
    gap_col = srgb("#67f4ff")
    for i, v in enumerate(gap_values):
        a = v / 255.0
        x = -GAP_W / 2 + (i + 0.5) * GAP_W / 256
        gap_bars.append((x, GAP_Y, 0.0, 0.22 + 1.35 * a, 0.035 + 0.014 * a,
                         mix(gap_col, (1.0, 1.0, 0.9), 0.25 + 0.65 * a)))
    gap_mat, gap_em = vertex_emission_mat("GAP_Bars", 2.4)
    cylinder_mesh("GAP_256_Channels", gap_bars, gap_mat, c_head, segments=8)
    gap_frame, gap_frame_em = add_frame("GAP", GAP_Y, GAP_W + 0.7, 2.6, 0.55,
                                        "#67f4ff", c_head, bevel=0.035, strength=2.2)
    # No opaque plate behind the head: the bars float on the dark space.

    # ---------------------------------------------------------------
    # Dense 1: 128 glass octahedra in a tight, structured lattice.
    # ---------------------------------------------------------------
    d1_values = base64.b64decode(data["dense1"]["activation"])
    d1_col = srgb("#b46bff")
    d1_pts = []
    cols1, rows1 = 16, 8
    for i, v in enumerate(d1_values):
        a = v / 255.0
        row, col = divmod(i, cols1)
        x = -DENSE1_W / 2 + (col + 0.5) * DENSE1_W / cols1
        z = DENSE1_H / 2 - (row + 0.5) * DENSE1_H / rows1
        colr = mix(d1_col, (1.0, 0.94, 0.83), 0.18 + 0.75 * a)
        d1_pts.append((Vector((x, DENSE1_Y, z)), 0.070 + 0.110 * a, colr))
    d1_mat, d1_em = vertex_emission_mat("Dense1_Nodes", 2.7)
    octahedra_mesh("Dense1_128_Nodes", d1_pts, d1_mat, c_head)
    d1_frame, d1_frame_em = add_frame("DENSE_1", DENSE1_Y, DENSE1_W + 1.0,
                                      DENSE1_H + 1.0, 1.1, "#b46bff", c_head,
                                      bevel=0.045, strength=2.1)

    # Text labels are billboards facing the initial camera. They are 3D text,
    # not a screen-space overlay, so they survive in the .blend and in glTF.
    # No backplates: the labels sit below the modules, directly on the dark
    # space. The lift runs along the camera axis, not towards the camera
    # position: in an orthographic projection this keeps the projected text
    # exactly below its module instead of drifting up over the frame.
    label_view_dir = (CAM_START - TARGET).normalized()

    def add_text(name: str, body: str, loc: tuple[float, float, float],
                 size: float, mat: bpy.types.Material, collection: bpy.types.Collection,
                 align: str = "CENTER", lift: float = 0.0) -> bpy.types.Object:
        cu = bpy.data.curves.new(name, "FONT")
        cu.body = body
        cu.size = size
        cu.align_x = align
        cu.align_y = "CENTER"
        cu.extrude = 0.008
        cu.bevel_depth = 0.002
        cu.space_line = 0.82
        ob = bpy.data.objects.new(name, cu)
        collection.objects.link(ob)
        loc_v = Vector(loc)
        ob.location = loc_v + label_view_dir * lift
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = label_view_dir.to_track_quat("Z", "Y")
        cu.materials.append(mat)
        return ob

    # ---------------------------------------------------------------
    # Output: a vertical 9-row luminescent bar chart. This is much more
    # legible than nine crowded vertical bars, while still being nine
    # separate extrusions connected to nine floating text meshes.
    # ---------------------------------------------------------------
    classes = data["classes"]
    probs = classes["probabilities"]
    supports = classes["support"]
    real = classes["real"]
    bar_objs = []
    label_objs = []
    label_mats: dict[str, tuple[bpy.types.Material, bpy.types.ShaderNode]] = {}
    max_support = max(supports) if supports else 1
    # The output chart sits to the right of the Dense-1 cage, not behind it:
    # from this three-quarter camera the head must be readable.
    OUT_START = 2.2
    for i, (name, p, support, is_real) in enumerate(zip(classes["names"], probs,
                                                         supports, real)):
        z = 3.35 - i * 0.94
        if is_real:
            length = 0.35 + 1.55 * math.sqrt(support / max_support) + 4.4 * p
            col = mix((0.88, 0.44, 0.13), (1.0, 0.96, 0.78), 0.25 + 0.62 * p)
            strength = 2.2 + 4.2 * p
            key = "real"
        else:
            length = 0.18 + 0.45 * p
            col = (0.13, 0.21, 0.36)
            strength = 0.38
            key = "future"
        bar_mat, bar_em = emission_mat(f"OutputBar_{name}", col, strength)
        bar = rod_mesh(f"Output_{name}", length, 0.22, 0.14, col, bar_mat, c_head)
        bar.location = (OUT_START, OUT_Y, z)
        bar_objs.append((bar, bar_em, length, key, name))

        if key not in label_mats:
            label_mats[key] = emission_mat(f"Output_{key}", "#ffd27a" if key == "real"
                                           else "#5e7899", 2.8 if key == "real" else 0.6)
        # Same Y as the bars, so the orthographic camera keeps every name
        # exactly on its row; the labels sit to the right of the bar ends.
        label = add_text(f"Class_{name}", name, (10.8, OUT_Y, z),
                         0.58, label_mats[key][0], c_annotations, align="LEFT")
        label_objs.append((label, is_real, name))

    # A single gold rail replaces the old cage; it keeps the rows aligned.
    rail_mat, _ = emission_mat("Output_Rail", "#ffd27a", 1.8)
    curve_object("Output_Rail", [[(OUT_START - 0.22, OUT_Y - 0.2, -4.45),
                                  (OUT_START - 0.22, OUT_Y - 0.2, 4.50)]],
                 0.025, rail_mat, c_head)
    head_mat, header_em = emission_mat("Output_Header", "#ffd27a", 2.0)
    # The output header is the last caption: it sits below the bar chart,
    # not above it, so every caption in the scene is under its module.
    add_text("OutputHeader", "OUTPUT  ·  9 classi", (5.2, OUT_Y - 1.2, -5.35),
             0.52, head_mat, c_annotations)

    # The title is part of the caption system too: it lives below the input,
    # not in the empty top band, so every piece of text sits under its module.
    title_mat, title_em = emission_mat("Title_Mat", "#dff7ff", 1.4)
    title = add_text("Title", "AstroNet-v2", (0.0, -0.6, -11.35), 1.00,
                     title_mat, c_annotations)
    subtitle = add_text(
        "Subtitle", "422.857 parametri  ·  61.833 neuroni",
        (0.0, -0.5, -12.80), 0.48, title_mat, c_annotations)

    # All module captions live BELOW their frame, directly on the background.
    # The camera-side lift keeps the fibre buses from drawing over the glyphs
    # while leaving them visually underneath their module.
    label_life: dict[str, tuple[bpy.types.ShaderNode, bpy.types.Object]] = {}

    def module_label(key: str, body: str, loc: tuple[float, float, float],
                     size: float, colour: str) -> None:
        mat, em = emission_mat(f"Label_{key}", colour, 1.0)
        fixed_names = {"input": "InputLabel", "gap": "GAPLabel",
                       "dense1": "D1Label"}
        ob = add_text(fixed_names.get(key, f"LayerLabel_{key}"),
                      body, loc, size, mat, c_annotations, lift=25.0)
        label_life[key] = (em, ob)

    module_label("input", "INPUT\n64 × 64", (0.0, Y_INPUT, -9.70),
                 0.72, "#8fe9ff")
    layer_labels = {
        "conv1": ("CONV 1\n32 filtri  ·  32.768", -9.00),
        "conv2": ("CONV 2\n64 filtri  ·  16.384", -8.00),
        "conv3": ("CONV 3\n128 filtri  ·  8.192", -7.00),
        "conv4": ("CONV 4\n256 filtri  ·  4.096", -6.45),
    }
    for name, (body, z_label) in layer_labels.items():
        spec = BLOCKS[name]
        module_label(name, body,
                     (0.0, spec["y"], z_label),
                     0.86, spec["color"])
    module_label("gap", "GAP\n256 canali", (0.0, GAP_Y, -5.00),
                 0.78, "#67f4ff")
    module_label("dense1", "DENSE 1\n128 nodi",
                 (0.0, DENSE1_Y, -8.20),
                 0.76, "#d8b8ff")

    # Legend lives in the accompanying README/HTML rather than in the 3D
    # close-up, where it would cover the output rail.

    # ---------------------------------------------------------------
    # Fibers: real strongest links, routed as ordered buses.
    # ---------------------------------------------------------------
    # Saturated but not blown out: a 2.0-strength pink turned white in AgX,
    # so the negative links were effectively invisible. 1.15 keeps the hue.
    neg_col = "#ff2d7f"
    stage_specs = [
        ("input_conv1", "#68d8ff", 0.0060, 14, 44),
        ("conv1_conv2", "#2f7bff", 0.0080, 36, 72),
        ("conv2_conv3", "#38ff9a", 0.0070, 68, 105),
        ("conv3_conv4", "#ffe14a", 0.0060, 100, 140),
        ("conv4_gap", "#ff7a1a", 0.0060, 134, 158),
        ("gap_dense1", "#b46bff", 0.0050, 152, 180),
        ("dense1_dense2", "#ffd27a", 0.0050, 174, 202),
    ]
    stage_objs = {}
    label_windows = {
        "input": (1, 4, 30, 44),
        "conv1": (12, 22, 52, 64),
        "conv2": (38, 50, 80, 92),
        "conv3": (66, 78, 108, 120),
        "conv4": (96, 108, 142, 158),
        "gap": (136, 152, 190, 198),
        "dense1": (166, 182, N_FRAMES, None),
    }

    def label_lifecycle() -> None:
        """Fade each caption with its layer (no backplate to animate)."""
        for key, (show_start, full_start, full_end, hide_end) in label_windows.items():
            em, ob = label_life[key]
            keys = [(1, 0.0), (show_start, 0.0), (full_start, 1.0),
                    (full_end, 1.0)]
            if hide_end is not None:
                keys.append((hide_end, 0.0))
            key_socket(em.inputs["Strength"], keys)
            ob.scale = (0.001, 0.001, 0.001)
            ob.keyframe_insert(data_path="scale", frame=1)
            ob.keyframe_insert(data_path="scale", frame=max(1, show_start))
            ob.scale = (1.0, 1.0, 1.0)
            ob.keyframe_insert(data_path="scale", frame=full_start)
            if hide_end is not None:
                ob.keyframe_insert(data_path="scale",
                                   frame=max(full_start, hide_end - 6))
                ob.scale = (0.001, 0.001, 0.001)
                ob.keyframe_insert(data_path="scale", frame=hide_end)

    for stage, stage_colour, bevel, start, end in stage_specs:
        conns = data["connections"][stage]
        splines = []
        if stage == "input_conv1":
            # Mass matters here: every one of the 4,096 input pixels emits a
            # filament into Conv 1. The 64 spatial blocks are distributed
            # across the 32 filters, so each filter receives 128 pixels.
            conv1_filters = data["blocks"][0]["filters"]
            for row in range(64):
                for col in range(64):
                    dst = ((row // 8) * 8 + (col // 8)) % 32
                    filt = conv1_filters[dst]
                    sx = (col - 31.5) * pitch
                    sz = (31.5 - row) * pitch
                    sy = -0.035
                    tx = centres["conv1"][dst].x
                    tz = centres["conv1"][dst].z
                    ty = BLOCKS["conv1"]["y"] - BLOCKS["conv1"]["d"] * 0.48
                    intensity = pixels[row * 64 + col] / 255.0
                    w = filt["norm"] * (0.45 + 0.55 * intensity)
                    sign = 1 if sum(filt.get("kernel", [0])) >= 0 else -1
                    dx, dz, dy = tx - sx, tz - sz, ty - sy
                    wave = (dst % 5 - 2) * 0.12
                    coords = [
                        (sx, sy, sz),
                        (sx + dx * 0.18, sy + dy * 0.28 + wave, sz + dz * 0.18),
                        (sx + dx * 0.50, sy + dy * 0.50, sz + dz * 0.50),
                        (sx + dx * 0.82, sy + dy * 0.72 + wave, sz + dz * 0.82),
                        (tx, ty, tz),
                    ]
                    radius = 0.42 + 0.85 * w
                    if sign < 0:
                        radius *= 1.30  # keep the negative minority visible
                    splines.append((coords, 0 if sign > 0 else 1, radius))
            pos_mat, pos_em = emission_mat(f"{stage}_positive", stage_colour, 2.2)
            neg_mat, neg_em = emission_mat(f"{stage}_negative", neg_col, 1.15)
            ob = fiber_curve_object(f"Fibers_{stage}", splines, bevel,
                                    pos_mat, neg_mat, c_fibers)
            key_growth(ob, start, end)
            key_socket(pos_em.inputs["Strength"], [(start - 1, 0.0), (start + 1, 1.0),
                                                   (end, 2.0), (N_FRAMES, 1.9)])
            key_socket(neg_em.inputs["Strength"], [(start - 1, 0.0), (start + 1, 0.75),
                                                   (end, 1.25), (N_FRAMES, 1.2)])
            stage_objs[stage] = ob
            continue

        for src, dst, w, sign in conns:
            # Source / target coordinates for each stage.
            if stage == "input_conv1":
                sx, sy, sz = centres["conv1"][dst].x * 0.82, -0.05, centres["conv1"][dst].z * 0.82
                tx, ty, tz = centres["conv1"][dst].x, BLOCKS["conv1"]["y"] - BLOCKS["conv1"]["d"] * 0.5, centres["conv1"][dst].z
                ry = 5.4
            elif stage == "conv1_conv2":
                sx, sy, sz = centres["conv1"][src].x, BLOCKS["conv1"]["y"] + BLOCKS["conv1"]["d"] * 0.45, centres["conv1"][src].z
                tx, ty, tz = centres["conv2"][dst].x, BLOCKS["conv2"]["y"] - BLOCKS["conv2"]["d"] * 0.45, centres["conv2"][dst].z
                ry = 23.0
            elif stage == "conv2_conv3":
                sx, sy, sz = centres["conv2"][src].x, BLOCKS["conv2"]["y"] + BLOCKS["conv2"]["d"] * 0.45, centres["conv2"][src].z
                tx, ty, tz = centres["conv3"][dst].x, BLOCKS["conv3"]["y"] - BLOCKS["conv3"]["d"] * 0.45, centres["conv3"][dst].z
                ry = 40.5
            elif stage == "conv3_conv4":
                sx, sy, sz = centres["conv3"][src].x, BLOCKS["conv3"]["y"] + BLOCKS["conv3"]["d"] * 0.45, centres["conv3"][src].z
                tx, ty, tz = centres["conv4"][dst].x, BLOCKS["conv4"]["y"] - BLOCKS["conv4"]["d"] * 0.45, centres["conv4"][dst].z
                ry = 58.5
            elif stage == "conv4_gap":
                sx, sy, sz = centres["conv4"][src].x, BLOCKS["conv4"]["y"] + BLOCKS["conv4"]["d"] * 0.45, centres["conv4"][src].z
                tx, ty, tz = -GAP_W / 2 + (dst + 0.5) * GAP_W / 256, GAP_Y, 0.0
                ry = 74.2
            elif stage == "gap_dense1":
                sx, sy, sz = -GAP_W / 2 + (src + 0.5) * GAP_W / 256, GAP_Y + 0.9, 0.0
                row, col = divmod(dst, 16)
                tx = -DENSE1_W / 2 + (col + 0.5) * DENSE1_W / 16
                tz = DENSE1_H / 2 - (row + 0.5) * DENSE1_H / 8
                ty, ry = DENSE1_Y - 0.65, 84.2
            else:  # dense1_dense2
                row, col = divmod(src, 16)
                sx = -DENSE1_W / 2 + (col + 0.5) * DENSE1_W / 16
                sz = DENSE1_H / 2 - (row + 0.5) * DENSE1_H / 8
                sy = DENSE1_Y + 0.65
                tx = OUT_START
                tz, ty, ry = 3.35 - dst * 0.94, OUT_Y - 0.2, 92.4
            # Four fine strands per Conv 4 -> GAP channel create the dense
            # optical-fibre curtain; all other stages already use many links.
            strands = 4 if stage == "conv4_gap" else 1
            for strand in range(strands):
                off = (strand - (strands - 1) / 2.0) * 0.035
                dx = tx - sx
                dz = tz - sz
                dy = ty - sy
                wave = (dst % 7 - 3) * 0.11
                coords = [
                    (sx + off, sy, sz + off),
                    (sx + dx * 0.18 + off, sy + dy * 0.28 + wave, sz + dz * 0.18 + off),
                    (sx + dx * 0.50 + off, sy + dy * 0.50, sz + dz * 0.50 + off),
                    (sx + dx * 0.82 + off, sy + dy * 0.72 + wave, sz + dz * 0.82 + off),
                    (tx + off, ty, tz + off),
                ]
                thick = (0.50 + 1.20 * (w ** 0.7)) * (0.76 if strands > 1 else 1.0)
                if sign < 0:
                    thick *= 1.30  # keep the negative minority visible
                splines.append((coords, 0 if sign > 0 else 1, thick))
        pos_mat, pos_em = emission_mat(f"{stage}_positive", stage_colour, 2.2)
        neg_mat, neg_em = emission_mat(f"{stage}_negative", neg_col, 1.15)
        ob = fiber_curve_object(f"Fibers_{stage}", splines, bevel,
                                pos_mat, neg_mat, c_fibers)
        key_growth(ob, start, end)
        key_socket(pos_em.inputs["Strength"], [(start - 1, 0.0), (start + 1, 1.0),
                                               (end, 2.0), (N_FRAMES, 1.9)])
        key_socket(neg_em.inputs["Strength"], [(start - 1, 0.0), (start + 1, 0.75),
                                               (end, 1.25), (N_FRAMES, 1.2)])
        stage_objs[stage] = ob

    # Captions belong to the layer they describe: they fade in as that
    # layer lights up and disappear before the camera closes on the head, so
    # no half-clipped label is left at the left edge of the final shot.
    label_lifecycle()

    # ---------------------------------------------------------------
    # Animation of the layer sequence.
    # ---------------------------------------------------------------
    key_socket(input_em.inputs["Strength"], [(1, 0.0), (5, 0.4), (20, 3.2),
                                              (N_FRAMES, 2.6)])
    key_socket(input_frame_em.inputs["Strength"], [(1, 0.0), (5, 0.8), (20, 2.0)])
    key_socket(base_em.inputs["Strength"], [(1, 0.0), (10, 0.5), (N_FRAMES, 0.35)])

    for name, spec in BLOCKS.items():
        start = {"conv1": 18, "conv2": 44, "conv3": 72, "conv4": 102}[name]
        end = {"conv1": 50, "conv2": 78, "conv3": 108, "conv4": 142}[name]
        b = block_objects[name]
        key_socket(b["frame_em"].inputs["Strength"], [(1, 0.0), (start, 0.2),
                                                       (end, 3.8), (N_FRAMES, 3.2)])
        key_socket(b["card_em"], [(1, 0.0), (start, 0.05),
                                  (end, 0.65), (N_FRAMES, 0.5)])
        key_socket(b["outline_em"].inputs["Strength"], [(1, 0.0), (start, 0.2),
                                                         (end, 2.4), (N_FRAMES, 1.8)])
        key_socket(b["dot_em"].inputs["Strength"], [(1, 0.0), (start + 2, 0.0),
                                                     (end, 2.4), (N_FRAMES, 2.0)])
        key_socket(point_objects[name][1].inputs["Strength"],
                   [(1, 0.0), (start, 0.0), (end, 3.2), (N_FRAMES, 2.8)])

    key_socket(gap_em.inputs["Strength"], [(1, 0.0), (136, 0.0), (158, 3.2),
                                            (N_FRAMES, 2.8)])
    key_socket(gap_frame_em.inputs["Strength"], [(1, 0.0), (136, 0.2), (158, 2.8)])
    key_socket(d1_em.inputs["Strength"], [(1, 0.0), (152, 0.0), (180, 3.0),
                                           (N_FRAMES, 2.6)])
    key_socket(d1_frame_em.inputs["Strength"], [(1, 0.0), (152, 0.2), (180, 2.8)])

    for bar, bar_em, length, key, name in bar_objs:
        start = 176 if key == "real" else 184
        end = 202 if key == "real" else 208
        key_grow_x(bar, start, end)
        key_socket(bar_em.inputs["Strength"],
                   [(1, 0.0), (start, 0.0), (end, 2.8 if key == "real" else 0.5),
                    (N_FRAMES, 2.8 if key == "real" else 0.45)])
    for label, is_real, name in label_objs:
        key_scale(label, 176 if is_real else 184, 201 if is_real else 210)
        # Label material is shared by real/future, so animate it once below.
    key_socket(label_mats["real"][1].inputs["Strength"],
               [(1, 0.0), (176, 0.0), (201, 2.8), (N_FRAMES, 2.8)])
    key_socket(label_mats["future"][1].inputs["Strength"],
               [(1, 0.0), (184, 0.0), (210, 0.55), (N_FRAMES, 0.45)])
    key_socket(title_em.inputs["Strength"], [(1, 0.0), (8, 1.4), (N_FRAMES, 1.4)])
    key_socket(header_em.inputs["Strength"], [(1, 0.0), (174, 0.0), (202, 2.6)])

    # ---------------------------------------------------------------
    # Star field: very low intensity, only for depth cues.
    # ---------------------------------------------------------------
    random.seed(17)
    star_pts = []
    for _ in range(220):
        x = random.uniform(-28, 28)
        y = random.uniform(-10, 110)
        z = random.uniform(-18, 24)
        if abs(x) < 9 and abs(z) < 9:
            continue
        b = random.uniform(0.25, 0.8)
        star_pts.append((Vector((x, y, z)), random.uniform(0.015, 0.04),
                         (0.35 * b, 0.55 * b, 1.0 * b)))
    star_mat, star_em = vertex_emission_mat("Starfield", 0.8)
    octahedra_mesh("Starfield", star_pts, star_mat, c_environment)

    # A very subtle Y-axis ruler at the left edge.
    axis_mat, axis_em = emission_mat("Y_Axis", "#274466", 0.55)
    axis_splines = [ [(-8.2, -1.0, -8.2), (-8.2, 102.0, -8.2)] ]
    for yy in (0, 14.5, 31.5, 49.5, 68.0, 79.4, 88.8, 96.6, 100):
        axis_splines.append([(-8.2, yy, -8.2), (-7.8, yy, -8.2)])
    curve_object("Y_Axis_Ruler", axis_splines, 0.008, axis_mat, c_environment)

    # Compositor glow. Blender 5.2 moved compositing to a node group on the
    # scene; the Glare node exposes its options as sockets (not properties).
    try:
        scene.use_nodes = True
        tree = bpy.data.node_groups.new("SatAi_Compositing", "CompositorNodeTree")
        scene.compositing_node_group = tree
        try:
            tree.interface.new_socket(name="Image", in_out="OUTPUT",
                                      socket_type="NodeSocketColor")
        except Exception:
            pass
        rl = tree.nodes.new("CompositorNodeRLayers")
        glare = tree.nodes.new("CompositorNodeGlare")
        glare.inputs["Type"].default_value = "Fog Glow"
        glare.inputs["Quality"].default_value = "High"
        glare.inputs["Threshold"].default_value = 0.55
        glare.inputs["Smoothness"].default_value = 0.32
        glare.inputs["Strength"].default_value = 0.90
        glare.inputs["Size"].default_value = 0.85
        comp = tree.nodes.new("NodeGroupOutput")
        tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
        tree.links.new(glare.outputs["Image"], comp.inputs[0])
        print("compositor: Fog Glow enabled")
    except Exception as exc:
        print("compositor skipped:", exc)

    # Scene metadata for anyone opening the .blend.
    scene["satai_model"] = "AstroNet-v2"
    scene["satai_parameters"] = data["meta"]["parameters"]
    scene["satai_crop"] = data["meta"]["crop"]
    scene["satai_prediction"] = data["meta"]["predicted_class"]
    scene["satai_confidence"] = data["meta"]["confidence"]
    scene["satai_note"] = (
        "Every filter kernel, fiber and activation comes from the trained "
        "checkpoint. Fibers are a dense real-link sample: every input pixel "
        "has a filament and every downstream target receives its strongest "
        "connections; thickness=|weight|."
    )

    OUT_FRAMES.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
    print(f"saved {OUT_BLEND}")
    print(f"objects: {len(bpy.data.objects)}  materials: {len(bpy.data.materials)}")
    print(f"prediction: {data['meta']['predicted_class']} "
          f"{data['meta']['confidence']:.4f}")

    # Deliverable stills. The head still uses the animation as authored.
    scene.frame_set(N_FRAMES)
    scene.camera = cam
    scene.render.filepath = str(OUT_HEAD)
    bpy.ops.render.render(write_still=True)
    print(f"wrote {OUT_HEAD}")

    # The wide poster is a documentation frame, not a frame of the clip:
    # every caption is forced visible even at the end of the timeline. The
    # overrides live only in this process; the saved .blend keeps the fades.
    for mat in bpy.data.materials:
        if not (mat.name.startswith("Label_") and mat.use_nodes):
            continue
        mat.node_tree.animation_data_clear()
        for node in mat.node_tree.nodes:
            if node.type == "EMISSION":
                node.inputs["Strength"].default_value = 1.0
    for ob in bpy.data.objects:
        if ob.type == "FONT" and (ob.name.endswith("Label") or
                                   "LayerLabel_" in ob.name):
            ob.animation_data_clear()
            ob.scale = (1.0, 1.0, 1.0)
    hero.data.ortho_scale = 88.0
    scene.camera = hero
    scene.frame_set(N_FRAMES)
    scene.render.filepath = str(OUT_POSTER)
    bpy.ops.render.render(write_still=True)
    print(f"wrote {OUT_POSTER}")


if __name__ == "__main__":
    build()
