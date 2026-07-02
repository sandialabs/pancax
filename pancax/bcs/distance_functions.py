from __future__ import annotations
from dataclasses import dataclass
from jax import numpy as jnp
from jax import vmap
from typing import Dict, Iterable, List, Optional, Tuple, Union
import jax
import numpy as np


SIDE_NODES_2D = {
    "TRI3": {
        1: [0, 1],
        2: [1, 2],
        3: [2, 0],
    },
    "QUAD4": {
        1: [0, 1],
        2: [1, 2],
        3: [2, 3],
        4: [3, 0],
    },
}
SIDE_NODES_3D = {
    "TET4": {
        1: [0, 1, 3],
        2: [1, 2, 3],
        3: [0, 3, 2],
        4: [0, 2, 1],
    },
    "HEX8": {
        # Bottom, top, front, right, back, left for standard HEX8.
        1: [0, 3, 2, 1],
        2: [4, 5, 6, 7],
        3: [0, 1, 5, 4],
        4: [1, 2, 6, 5],
        5: [2, 3, 7, 6],
        6: [3, 0, 4, 7],
    },
}


# some basic math stuff
def safe_norm(x, eps: float = 0.0):
    """
    Euclidean norm. Set eps > 0 only if you need numerical regularization.
    For exact distance-function behavior, keep eps = 0.
    """
    return jnp.sqrt(jnp.sum(x * x) + eps * eps)


def r_conj(a, b):
    """
    R-conjunction a ∧ b.

    Positive when both a and b are positive.
    Zero when either a or b is zero.
    Negative when either is negative.

    This is the alpha=0 R-conjunction:
        a ∧ b = a + b - sqrt(a^2 + b^2)
    """
    return a + b - jnp.sqrt(a * a + b * b)


def r_equiv_combine(phi_values, m: float = 1.0):
    """
    R-equivalence combination:

        phi = (sum_i 1 / phi_i^m)^(-1/m)

    phi_values: shape (..., n_entities)

    returns: shape (...)
    """
    inv_sum = jnp.sum(jnp.power(phi_values, -m), axis=-1)
    return jnp.power(inv_sum, -1.0 / m)


def line_segment_adf(x, segment):
    """
    Approximate distance function to a finite line segment in 2D.

    Parameters
    ----------
    x:
        Shape (2,)

    segment:
        Shape (2, 2)
        segment[0, :] = first endpoint
        segment[1, :] = second endpoint

    Returns
    -------
    phi:
        Scalar ADF to the segment.
    """
    x1 = segment[0, :]
    x2 = segment[1, :]

    dx = x2 - x1
    L = safe_norm(dx)

    x_c = 0.5 * (x1 + x2)

    f = (
        (x[0] - x1[0]) * (x2[1] - x1[1])
        - (x[1] - x1[1]) * (x2[0] - x1[0])
    ) / L

    t = ((0.5 * L) ** 2 - safe_norm(x - x_c) ** 2) / L

    varphi = jnp.sqrt(t * t + f ** 4)

    phi = jnp.sqrt(f * f + 0.25 * (varphi - t) ** 2)

    return phi


def polygon_face_adf(x, face):
    """
    Approximate distance function to a finite planar polygonal face in 3D.

    The face is assumed to be a triangle or quadrilateral with outward-oriented
    vertex ordering.

    Parameters
    ----------
    x:
        Shape (3,)

    face:
        Shape (n_vertices, 3), where n_vertices is 3 or 4.

    Returns
    -------
    phi:
        Scalar ADF to the finite face.
    """
    n_vertices = face.shape[0]

    x0 = face[0, :]
    x1 = face[1, :]
    x2 = face[2, :]

    # Outward normal from vertex ordering.
    n_out_unnorm = jnp.cross(x1 - x0, x2 - x0)
    n_out = n_out_unnorm / safe_norm(n_out_unnorm)

    # Inward normal.
    nu = -n_out

    # Signed distance to the infinite plane, positive inside.
    f = jnp.dot(x - x0, nu)

    # Projection of x onto the face plane.
    p = x - f * nu

    centroid = jnp.mean(face, axis=0)

    # Build edge half-space functions.
    def edge_s(i):
        a = face[i, :]
        b = face[(i + 1) % n_vertices, :]

        e = b - a
        ehat = e / safe_norm(e)

        # Candidate inward in-plane normal.
        #
        # For outward-oriented polygon, n_out x ehat usually points inward.
        m = jnp.cross(n_out, ehat)

        # Robustly flip if needed so that it points toward centroid.
        sign = jnp.dot(centroid - a, m)
        m = jnp.where(sign < 0.0, -m, m)

        return jnp.dot(p - a, m)

    s0 = edge_s(0)
    s1 = edge_s(1)
    t = r_conj(s0, s1)

    if n_vertices == 3:
        s2 = edge_s(2)
        t = r_conj(t, s2)
    elif n_vertices == 4:
        s2 = edge_s(2)
        s3 = edge_s(3)
        t = r_conj(t, s2)
        t = r_conj(t, s3)
    else:
        raise ValueError("polygon_face_adf only supports triangles and quads.")

    varphi = jnp.sqrt(t * t + f ** 4)

    phi = jnp.sqrt(f * f + 0.25 * (varphi - t) ** 2)

    return phi


def triangle_face_adf(x, tri_face):
    """
    tri_face shape: (3, 3)
    """
    return polygon_face_adf(x, tri_face)


def quad_face_adf(x, quad_face):
    """
    quad_face shape: (4, 3)
    """
    return polygon_face_adf(x, quad_face)


def normalize_elem_type(elem_type: str) -> str:
    """
    Normalize common ExodusII element-type names.
    """
    et = elem_type.upper()

    # Strip common suffixes.
    et = et.replace("SHELL", "QUAD")
    et = et.replace("TRIANGLE", "TRI")
    et = et.replace("TETRA", "TET")

    if et in ("TRI", "TRI3"):
        return "TRI3"

    if et in ("QUAD", "QUAD4"):
        return "QUAD4"

    if et in ("TET", "TET4"):
        return "TET4"

    if et in ("HEX", "HEX8", "HEXAHEDRON"):
        return "HEX8"

    raise ValueError(f"Unsupported element type: {elem_type}")


@dataclass
class ElementBlock:
    elem_type: str
    connectivity: np.ndarray
    elem_ids: Optional[np.ndarray] = None


def iter_element_blocks(mesh) -> Iterable[ElementBlock]:
    """
    Adapter for common ExodusII-like mesh objects.

    You may need to modify this function to match your mesh class.

    Expected output:
        ElementBlock(elem_type, connectivity, elem_ids)

    connectivity:
        Shape (n_elem_in_block, n_nodes_per_elem)
        Should be zero-based if possible. If one-based, we convert later.

    elem_ids:
        Optional global element IDs. If absent, we assume elements are globally
        numbered consecutively in block order.
    """

    # Case 1: mesh.elementBlocks is a dict or list.
    if hasattr(mesh, "elementBlocks"):
        blocks = mesh.elementBlocks

        if isinstance(blocks, dict):
            iterable = blocks.values()
        else:
            iterable = blocks

        for block in iterable:
            elem_type = (
                getattr(block, "elem_type", None)
                or getattr(block, "elemType", None)
                or getattr(block, "type", None)
                or getattr(block, "name", None)
            )

            conn = (
                getattr(block, "connectivity", None)
                or getattr(block, "conn", None)
            )

            elem_ids = (
                getattr(block, "elem_ids", None)
                or getattr(block, "elemIds", None)
                or None
            )

            if elem_type is None or conn is None:
                raise AttributeError(
                    "Could not infer elem_type/connectivity from mesh.elementBlocks."
                )

            yield ElementBlock(
                elem_type=normalize_elem_type(elem_type),
                connectivity=np.asarray(conn),
                elem_ids=None if elem_ids is None else np.asarray(elem_ids),
            )

        return

    # Case 2: mesh.blocks is a dict or list.
    if hasattr(mesh, "blocks"):
        blocks = mesh.blocks

        if isinstance(blocks, dict):
            iterable = blocks.values()
        else:
            iterable = blocks

        for block in iterable:
            elem_type = (
                getattr(block, "elem_type", None)
                or getattr(block, "elemType", None)
                or getattr(block, "type", None)
                or getattr(block, "name", None)
            )

            conn = (
                getattr(block, "connectivity", None)
                or getattr(block, "conn", None)
            )

            elem_ids = (
                getattr(block, "elem_ids", None)
                or getattr(block, "elemIds", None)
                or None
            )

            if elem_type is None or conn is None:
                raise AttributeError(
                    "Could not infer elem_type/connectivity from mesh.blocks."
                )

            yield ElementBlock(
                elem_type=normalize_elem_type(elem_type),
                connectivity=np.asarray(conn),
                elem_ids=None if elem_ids is None else np.asarray(elem_ids),
            )

        return

    # Case 3: single-block mesh.
    if hasattr(mesh, "connectivity") and (
        hasattr(mesh, "elem_type") or hasattr(mesh, "elemType")
    ):
        elem_type = getattr(mesh, "elem_type", None) or getattr(mesh, "elemType", None)

        yield ElementBlock(
            elem_type=normalize_elem_type(elem_type),
            connectivity=np.asarray(mesh.connectivity),
            elem_ids=getattr(mesh, "elem_ids", None),
        )

        return

    raise AttributeError(
        "Could not find element blocks. Modify iter_element_blocks(mesh) "
        "for your mesh object."
    )


def to_zero_based_connectivity(conn: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Convert connectivity to zero-based if it appears to be one-based.
    """
    conn = np.asarray(conn, dtype=np.int64)

    if conn.size == 0:
        return conn

    cmin = conn.min()
    cmax = conn.max()

    # Common Exodus case: node IDs from 1 to n_nodes.
    if cmin >= 1 and cmax <= n_nodes:
        return conn - 1

    return conn


@dataclass
class ElementRecord:
    elem_type: str
    conn: np.ndarray


def build_element_lookup(mesh) -> Dict[int, ElementRecord]:
    """
    Build mapping:
        global_element_id -> ElementRecord(elem_type, conn)

    Element IDs in the lookup are whatever IDs are found in the mesh.
    If no element IDs are stored, we assume 1-based Exodus-style global IDs.
    """
    coords = np.asarray(mesh.coords)
    n_nodes = coords.shape[0]

    lookup: Dict[int, ElementRecord] = {}

    next_elem_id = 1

    for block in iter_element_blocks(mesh):
        conn = to_zero_based_connectivity(block.connectivity, n_nodes)
        elem_type = normalize_elem_type(block.elem_type)

        n_elem = conn.shape[0]

        if block.elem_ids is None:
            elem_ids = np.arange(next_elem_id, next_elem_id + n_elem, dtype=np.int64)
        else:
            elem_ids = np.asarray(block.elem_ids, dtype=np.int64)

        for eid, elem_conn in zip(elem_ids, conn):
            lookup[int(eid)] = ElementRecord(
                elem_type=elem_type,
                conn=np.asarray(elem_conn, dtype=np.int64),
            )

        next_elem_id += n_elem

    return lookup


def orient_face_outward(face_nodes: np.ndarray, elem_conn: np.ndarray, coords: np.ndarray):
    """
    Orient a 3D face outward relative to its parent element.

    face_nodes:
        Shape (3,) or (4,)

    elem_conn:
        Parent element node IDs.

    coords:
        Mesh coordinates, shape (n_nodes, 3)

    Returns
    -------
    face_nodes_out:
        Reordered face nodes whose cross-product normal points outward.
    """
    face_nodes = np.asarray(face_nodes, dtype=np.int64)

    face_pts = coords[face_nodes]
    elem_pts = coords[elem_conn]

    face_centroid = np.mean(face_pts, axis=0)
    elem_centroid = np.mean(elem_pts, axis=0)

    v1 = face_pts[1] - face_pts[0]
    v2 = face_pts[2] - face_pts[0]

    n = np.cross(v1, v2)

    # Vector from face to element interior.
    to_elem = elem_centroid - face_centroid

    # If normal points toward the element interior, flip it.
    if np.dot(n, to_elem) > 0.0:
        face_nodes = face_nodes[::-1]

    return face_nodes


@dataclass
class BoundaryEntities:
    edges_2d: Optional[np.ndarray] = None       # shape (n_edges, 2)
    tri_faces_3d: Optional[np.ndarray] = None   # shape (n_tri, 3)
    quad_faces_3d: Optional[np.ndarray] = None  # shape (n_quad, 4)


def get_boundary_entities(
    domain,
    sset_names: List[str],
    side_set_format: str = "exodus",
    deduplicate: bool = True,
) -> BoundaryEntities:
    """
    Extract 2D edges or 3D faces from side sets.

    Parameters
    ----------
    domain:
        Object with domain.fspace.mesh.

    sset_names:
        List of side-set names to combine.

    side_set_format:
        "exodus":
            Side sets contain (element_id, side_id) pairs.

        "nodes":
            Side sets already contain node IDs.

    deduplicate:
        Remove duplicate boundary entities.

    Returns
    -------
    BoundaryEntities
    """
    mesh = domain.fspace.mesh
    coords = np.asarray(mesh.coords)
    spatial_dim = coords.shape[1]

    raw_ssets = []
    for name in sset_names:
        raw_ssets.append(np.asarray(mesh.sideSets[name], dtype=np.int64))

    raw = np.vstack(raw_ssets)

    if side_set_format not in ("exodus", "nodes"):
        raise ValueError("side_set_format must be 'exodus' or 'nodes'.")

    # -------------------------------------------------------------------------
    # Case A: side sets already contain node IDs.
    # -------------------------------------------------------------------------
    if side_set_format == "nodes":
        entities = raw.copy()

        # Convert to zero-based if necessary.
        n_nodes = coords.shape[0]
        entities = to_zero_based_connectivity(entities, n_nodes)

        if spatial_dim == 2:
            if entities.shape[1] != 2:
                raise ValueError(
                    f"2D node-based side sets should have shape (n, 2). "
                    f"Got {entities.shape}."
                )

            # For an unsigned line-segment ADF, orientation is not important.
            # Sorting helps deduplicate.
            if deduplicate:
                sorted_edges = np.sort(entities, axis=1)
                edges = np.unique(sorted_edges, axis=0)
            else:
                edges = entities

            return BoundaryEntities(edges_2d=edges)

        elif spatial_dim == 3:
            if entities.shape[1] == 3:
                tri_faces = entities
                quad_faces = None
            elif entities.shape[1] == 4:
                tri_faces = None
                quad_faces = entities
            else:
                raise ValueError(
                    f"3D node-based side sets should have 3 or 4 nodes per face. "
                    f"Got {entities.shape}."
                )

            # WARNING:
            # For 3D node-based side sets, orientation may be unknown.
            # The ADF for Neumann BCs requires inward-positive phi.
            # Prefer side_set_format='exodus' when possible so faces can be
            # oriented using the parent element centroid.
            if deduplicate:
                if tri_faces is not None:
                    key = np.sort(tri_faces, axis=1)
                    _, idx = np.unique(key, axis=0, return_index=True)
                    tri_faces = tri_faces[np.sort(idx)]

                if quad_faces is not None:
                    key = np.sort(quad_faces, axis=1)
                    _, idx = np.unique(key, axis=0, return_index=True)
                    quad_faces = quad_faces[np.sort(idx)]

            return BoundaryEntities(
                tri_faces_3d=tri_faces,
                quad_faces_3d=quad_faces,
            )

        else:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")

    # -------------------------------------------------------------------------
    # Case B: Exodus-style side set: (element_id, side_id).
    # -------------------------------------------------------------------------
    if side_set_format == "exodus":
        if raw.shape[1] != 2:
            raise ValueError(
                f"Exodus side sets should have shape (n, 2): "
                f"(element_id, side_id). Got {raw.shape}."
            )

        elem_lookup = build_element_lookup(mesh)

        if spatial_dim == 2:
            edges = []

            for elem_id, side_id in raw:
                elem_id = int(elem_id)
                side_id = int(side_id)

                rec = elem_lookup[elem_id]
                elem_type = rec.elem_type
                conn = rec.conn

                if elem_type not in SIDE_NODES_2D:
                    raise ValueError(
                        f"Element type {elem_type} is not a supported 2D element."
                    )

                local_nodes = SIDE_NODES_2D[elem_type][side_id]
                edge_nodes = conn[local_nodes]
                edges.append(edge_nodes)

            edges = np.asarray(edges, dtype=np.int64)

            if deduplicate:
                sorted_edges = np.sort(edges, axis=1)
                edges = np.unique(sorted_edges, axis=0)

            return BoundaryEntities(edges_2d=edges)

        elif spatial_dim == 3:
            tri_faces = []
            quad_faces = []

            for elem_id, side_id in raw:
                elem_id = int(elem_id)
                side_id = int(side_id)

                rec = elem_lookup[elem_id]
                elem_type = rec.elem_type
                conn = rec.conn

                if elem_type not in SIDE_NODES_3D:
                    raise ValueError(
                        f"Element type {elem_type} is not a supported 3D element."
                    )

                local_nodes = SIDE_NODES_3D[elem_type][side_id]
                face_nodes = conn[local_nodes]

                # Orient face outward relative to parent element.
                face_nodes = orient_face_outward(face_nodes, conn, coords)

                if len(face_nodes) == 3:
                    tri_faces.append(face_nodes)
                elif len(face_nodes) == 4:
                    quad_faces.append(face_nodes)
                else:
                    raise ValueError("Only triangular and quadrilateral faces supported.")

            tri_faces = (
                np.asarray(tri_faces, dtype=np.int64)
                if len(tri_faces) > 0
                else None
            )

            quad_faces = (
                np.asarray(quad_faces, dtype=np.int64)
                if len(quad_faces) > 0
                else None
            )

            if deduplicate:
                if tri_faces is not None:
                    key = np.sort(tri_faces, axis=1)
                    _, idx = np.unique(key, axis=0, return_index=True)
                    tri_faces = tri_faces[np.sort(idx)]

                if quad_faces is not None:
                    key = np.sort(quad_faces, axis=1)
                    _, idx = np.unique(key, axis=0, return_index=True)
                    quad_faces = quad_faces[np.sort(idx)]

            return BoundaryEntities(
                tri_faces_3d=tri_faces,
                quad_faces_3d=quad_faces,
            )

        else:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")


def distance_function(
    domain,
    ssets: List[str],
    m: float = 1.0,
    side_set_format: str = "exodus",
    deduplicate: bool = True,
):
    """
    Build a batched approximate distance function phi(X) for boundary side sets.

    Supports:
        2D:
            TRI3, QUAD4 element side sets -> edges
            or node-based edge side sets

        3D:
            TET4 side sets -> triangular faces
            HEX8 side sets -> quadrilateral faces
            or node-based triangular/quadrilateral face side sets

    Parameters
    ----------
    domain:
        Object with domain.fspace.mesh.

    ssets:
        List of side-set names.

    m:
        R-equivalence exponent.
        Usually start with m = 1.0.

    side_set_format:
        "exodus":
            side sets contain (element_id, side_id) pairs.

        "nodes":
            side sets already contain boundary node IDs.

    deduplicate:
        Whether to remove repeated edges/faces.

    Returns
    -------
    phi_batched:
        Function mapping X with shape (n_points, dim) to phi values
        with shape (n_points,).
    """
    mesh = domain.fspace.mesh
    coords_np = np.asarray(mesh.coords)
    spatial_dim = coords_np.shape[1]

    entities = get_boundary_entities(
        domain,
        sset_names=ssets,
        side_set_format=side_set_format,
        deduplicate=deduplicate,
    )

    coords = jnp.asarray(coords_np)

    if spatial_dim == 2:
        if entities.edges_2d is None or len(entities.edges_2d) == 0:
            raise ValueError("No 2D boundary edges were found.")

        edges = jnp.asarray(entities.edges_2d)
        segments = coords[edges, :]  # shape (n_edges, 2, 2)

        def phi_one(x):
            phi_edges = vmap(line_segment_adf, in_axes=(None, 0))(x, segments)
            return r_equiv_combine(phi_edges, m=m)

        return vmap(phi_one)

    elif spatial_dim == 3:
        tri_faces = None
        quad_faces = None

        if entities.tri_faces_3d is not None and len(entities.tri_faces_3d) > 0:
            tri_faces = coords[jnp.asarray(entities.tri_faces_3d), :]
            # shape (n_tri, 3, 3)

        if entities.quad_faces_3d is not None and len(entities.quad_faces_3d) > 0:
            quad_faces = coords[jnp.asarray(entities.quad_faces_3d), :]
            # shape (n_quad, 4, 3)

        if tri_faces is None and quad_faces is None:
            raise ValueError("No 3D boundary faces were found.")

        def phi_one(x):
            pieces = []

            if tri_faces is not None:
                phi_tri = vmap(triangle_face_adf, in_axes=(None, 0))(x, tri_faces)
                pieces.append(phi_tri)

            if quad_faces is not None:
                phi_quad = vmap(quad_face_adf, in_axes=(None, 0))(x, quad_faces)
                pieces.append(phi_quad)

            phi_all = jnp.concatenate(pieces, axis=0)

            return r_equiv_combine(phi_all, m=m)

        return vmap(phi_one)

    else:
        raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")
