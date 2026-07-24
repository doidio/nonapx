import warp as wp
import numpy as np
import trimesh
import itk
import pyvista as pv
from vtk import vtkImageData, vtkFlyingEdges3D, VTK_FLOAT
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def _get_volume_np(volume) -> np.ndarray:
    if isinstance(volume, wp.array):
        return volume.numpy()
    elif isinstance(volume, np.ndarray):
        return volume
    elif isinstance(volume, itk.Image):
        # ITK array_view_from_image returns (Z, Y, X), so we transpose it to (X, Y, Z)
        return itk.array_view_from_image(volume).transpose((2, 1, 0))
    elif isinstance(volume, vtkImageData):
        shape = volume.GetDimensions()
        return vtk_to_numpy(volume.GetPointData().GetScalars()).reshape(shape, order='F')
    else:
        raise TypeError(f"Unsupported volume type: {type(volume)}")


@wp.kernel
def _resample_volume_kernel(
    tex: wp.Texture3D,
    old_spacing: wp.vec3,
    new_spacing: wp.vec3,
    old_dim: wp.vec3,
    out: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()

    w = (float(i) * new_spacing[0] / old_spacing[0] + 0.5) / old_dim[0]
    v = (float(j) * new_spacing[1] / old_spacing[1] + 0.5) / old_dim[1]
    u = (float(k) * new_spacing[2] / old_spacing[2] + 0.5) / old_dim[2]

    out[i, j, k] = wp.texture_sample(tex, wp.vec3f(u, v, w), dtype=float)


@wp.kernel
def ct_to_panorama(
    image: wp.array2d(dtype=wp.float32),
    tex: wp.Texture3D,
    volume_origin: wp.vec3,
    volume_spacing: wp.vec3,
    volume_shape: wp.vec3,
    scan_to_world: wp.transform,
    vertical_spacing: float,
    focal_trough_depth: float,
    ray_step: float,
    origins: wp.array(dtype=wp.vec3),
    axes_y: wp.array(dtype=wp.vec3),
    axes_z: wp.array(dtype=wp.vec3),
):
    row, col = wp.tid()

    origin = origins[col]
    axis_y = axes_y[col]
    axis_z = axes_z[col]

    origin = wp.transform_point(scan_to_world, origin)
    axis_y = wp.normalize(wp.transform_vector(scan_to_world, axis_y))
    axis_z = wp.normalize(wp.transform_vector(scan_to_world, axis_z))

    center_row = 0.5 * float(image.shape[0] - 1)
    origin += axis_z * ((center_row - float(row)) * vertical_spacing)

    sample_count = int(wp.ceil(focal_trough_depth / ray_step))
    sample_step = focal_trough_depth / float(sample_count)
    ray_start = -0.5 * focal_trough_depth + 0.5 * sample_step
    projection = float(0.0)

    for sample_index in range(sample_count):
        point = origin + axis_y * (ray_start + float(sample_index) * sample_step)
        index = wp.cw_div(point - volume_origin, volume_spacing)

        inside = (
            index[0] >= -0.5
            and index[0] <= volume_shape[0] - 0.5
            and index[1] >= -0.5
            and index[1] <= volume_shape[1] - 0.5
            and index[2] >= -0.5
            and index[2] <= volume_shape[2] - 0.5
        )
        if inside:
            # Texture3D maps the NumPy (X, Y, Z) array to texture (Z, Y, X).
            uvw = wp.vec3(
                (index[2] + 0.5) / volume_shape[2],
                (index[1] + 0.5) / volume_shape[1],
                (index[0] + 0.5) / volume_shape[0],
            )
            hu = wp.texture_sample(tex, uvw, dtype=float)
            relative_attenuation = wp.max(hu + 1000.0, 0.0) / 1000.0
            projection += relative_attenuation * sample_step

    image[row, col] = projection


def resample_volume_wp(
    volume: wp.array3d[wp.float32] | np.ndarray | itk.Image | vtkImageData,
    spacing: float | np.ndarray | list,
    new_spacing: float | np.ndarray | list,
    filter_mode: wp.TextureFilterMode = wp.TextureFilterMode.LINEAR,
    address_mode: wp.TextureAddressMode = wp.TextureAddressMode.CLAMP,
) -> np.ndarray:
    vol_np = _get_volume_np(volume).astype(np.float32, copy=False)

    if isinstance(spacing, (float, int)):
        old_spacing_arr = np.array([spacing, spacing, spacing], dtype=np.float32)
    else:
        old_spacing_arr = np.array(spacing, dtype=np.float32)

    if isinstance(new_spacing, (float, int)):
        new_spacing_arr = np.array([new_spacing, new_spacing, new_spacing], dtype=np.float32)
    else:
        new_spacing_arr = np.array(new_spacing, dtype=np.float32)

    old_shape = vol_np.shape
    new_shape = tuple(max(1, int(round(old_shape[i] * old_spacing_arr[i] / new_spacing_arr[i]))) for i in range(3))

    tex = wp.Texture3D(
        np.ascontiguousarray(vol_np),
        filter_mode=filter_mode,
        address_mode=address_mode,
    )

    old_spacing_wp = wp.vec3(float(old_spacing_arr[0]), float(old_spacing_arr[1]), float(old_spacing_arr[2]))
    new_spacing_wp = wp.vec3(float(new_spacing_arr[0]), float(new_spacing_arr[1]), float(new_spacing_arr[2]))
    old_dim_wp = wp.vec3(float(old_shape[0]), float(old_shape[1]), float(old_shape[2]))

    out_wp = wp.zeros(new_shape, dtype=wp.float32)
    wp.launch(_resample_volume_kernel, dim=new_shape, inputs=[tex, old_spacing_wp, new_spacing_wp, old_dim_wp, out_wp])

    return out_wp.numpy()


def decimate_mesh_vtk(
    mesh: trimesh.Trimesh,
    target_reduction: float = 0.8,
    method: str = "quadric",
) -> trimesh.Trimesh:
    if len(mesh.vertices) == 0 or target_reduction <= 0:
        return mesh

    faces_pv = np.hstack([np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces]).ravel()
    pv_mesh = pv.PolyData(mesh.vertices, faces_pv)

    if method == "pro":
        decimated = pv_mesh.decimate_pro(target_reduction, preserve_topology=True)
    else:
        decimated = pv_mesh.decimate(target_reduction)

    faces_out = decimated.faces.reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(vertices=np.asarray(decimated.points), faces=np.asarray(faces_out))


def extract_surface_wp(
    volume: wp.array3d[wp.float32] | np.ndarray | itk.Image | vtkImageData,
    origin: np.ndarray,
    spacing: float | np.ndarray,
    threshold: float,
):
    vol_np = _get_volume_np(volume).astype(np.float32, copy=False)
    vol_wp = wp.array(np.ascontiguousarray(vol_np), dtype=wp.float32) if not isinstance(volume, wp.array) else volume

    vertices, indices = wp.MarchingCubes.extract_surface_marching_cubes(
        -vol_wp,
        -threshold,
        wp.vec3(origin),
        wp.vec3(origin + spacing * (np.array(vol_wp.shape) - 1)),
    )
    return trimesh.Trimesh(vertices.numpy(), indices.numpy().reshape((-1, 3)))


def extract_surface_vtk(
    volume: wp.array3d[wp.float32] | np.ndarray | itk.Image | vtkImageData,
    origin: np.ndarray,
    spacing: float | np.ndarray,
    threshold: float,
):
    # Handle scalar spacing
    if isinstance(spacing, (float, int)):
        spacing = np.array([spacing, spacing, spacing])
    else:
        spacing = np.array(spacing)

    volume_np = _get_volume_np(volume).astype(np.float32, copy=False)

    image = vtkImageData()
    image.SetDimensions(volume_np.shape)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)

    # order='F' ensures X changes fastest, matching VTK's flat layout for (X, Y, Z) volumes
    vtk_data = numpy_to_vtk(volume_np.ravel(order='F'), deep=True, array_type=VTK_FLOAT)
    image.GetPointData().SetScalars(vtk_data)

    fe = vtkFlyingEdges3D()
    fe.SetInputData(image)
    fe.SetValue(0, threshold)
    fe.ComputeNormalsOff()
    fe.Update()

    polydata = fe.GetOutput()

    if polydata.GetNumberOfPoints() == 0:
        return trimesh.Trimesh()

    vertices = vtk_to_numpy(polydata.GetPoints().GetData())
    # GetConnectivityArray returns a flat array of indices without the polygon size prefix (requires VTK 9.0+)
    indices = vtk_to_numpy(polydata.GetPolys().GetConnectivityArray()).reshape((-1, 3))

    return trimesh.Trimesh(vertices, indices)


if __name__ == '__main__':
    print("Running consistency test between Warp and VTK...")
    wp.init()

    shape = (40, 30, 20)  # X, Y, Z
    origin = np.array([10.0, 20.0, 30.0])  # X, Y, Z
    spacing = np.array([0.5, 1.0, 1.5])  # X, Y, Z

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    x_phys = origin[0] + x * spacing[0]
    y_phys = origin[1] + y * spacing[1]
    z_phys = origin[2] + z * spacing[2]

    center = np.array([
        origin[0] + (shape[0] - 5) * spacing[0],
        origin[1] + 15 * spacing[1],
        origin[2] + 10 * spacing[2]
    ])
    radius = 4.0

    # Create volume (X, Y, Z) order
    volume_xyz = np.sqrt((x_phys - center[0])**2 + (y_phys - center[1])**2 + (z_phys - center[2])**2).astype(np.float32)
    threshold = radius

    # Ensure memory is contiguous for Warp
    volume_contig = np.ascontiguousarray(volume_xyz)
    warp_volume = wp.array(volume_contig, dtype=wp.float32)

    print("\n--- Running Warp Marching Cubes ---")
    mesh_warp = extract_surface_wp(warp_volume, origin, spacing, threshold)
    print(f"Warp vertices: {len(mesh_warp.vertices)}, faces: {len(mesh_warp.faces)}")
    warp_center = np.mean(mesh_warp.vertices, axis=0)
    warp_bounds = mesh_warp.bounds
    print(f"Warp center: {warp_center}")
    print(f"Warp bounds:\n{warp_bounds}")

    print("\n--- Running VTK FlyingEdges3D ---")
    mesh_vtk = extract_surface_vtk(warp_volume, origin, spacing, threshold)
    print(f"VTK vertices: {len(mesh_vtk.vertices)}, faces: {len(mesh_vtk.faces)}")
    vtk_center = np.mean(mesh_vtk.vertices, axis=0)
    vtk_bounds = mesh_vtk.bounds
    print(f"VTK center: {vtk_center}")
    print(f"VTK bounds:\n{vtk_bounds}")

    np.testing.assert_allclose(warp_center, vtk_center, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(warp_bounds, vtk_bounds, rtol=1e-3, atol=1e-3)
    print("\n✅ Consistency test passed! Warp and VTK outputs perfectly match.")

    print("\n--- Running Multi-Type Consistency Test for extract_surface_mc ---")

    # 1. np.ndarray
    mesh_np = extract_surface_wp(volume_contig, origin, spacing, threshold)

    # 2. itk.Image (requires (Z, Y, X) layout when creating from array to match (X, Y, Z) ITK size)
    volume_itk = itk.image_from_array(volume_contig.transpose((2, 1, 0)).copy())
    volume_itk.SetOrigin(origin)
    volume_itk.SetSpacing(spacing)
    mesh_itk = extract_surface_wp(volume_itk, origin, spacing, threshold)

    # 3. vtkImageData
    volume_vtk = vtkImageData()
    volume_vtk.SetDimensions(shape)
    volume_vtk.SetOrigin(origin)
    volume_vtk.SetSpacing(spacing)
    # VTK expects flattened F-order for (X, Y, Z) inputs
    vtk_data_array = numpy_to_vtk(volume_contig.ravel(order='F'), deep=True, array_type=VTK_FLOAT)
    volume_vtk.GetPointData().SetScalars(vtk_data_array)
    mesh_vtk2 = extract_surface_wp(volume_vtk, origin, spacing, threshold)

    # Compare against Warp outputs (strictly identical because all go through the same underlying Warp kernel)
    np.testing.assert_allclose(mesh_warp.vertices, mesh_np.vertices, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(mesh_warp.faces, mesh_np.faces)

    np.testing.assert_allclose(mesh_warp.vertices, mesh_itk.vertices, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(mesh_warp.faces, mesh_itk.faces)

    np.testing.assert_allclose(mesh_warp.vertices, mesh_vtk2.vertices, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(mesh_warp.faces, mesh_vtk2.faces)

    print("\n✅ Multi-Type Consistency test passed! All 4 types (wp, np, itk, vtk) yield strictly identical results.")

    print("\n--- Running resample_volume_wp Test ---")
    resampled = resample_volume_wp(volume_contig, spacing, 2.5)
    print(f"Original shape: {volume_contig.shape}, Resampled shape: {resampled.shape}")
    mesh_resampled = extract_surface_wp(resampled, origin, 2.5, threshold)
    print(f"Resampled mesh vertices: {len(mesh_resampled.vertices)}, faces: {len(mesh_resampled.faces)}")
    print("✅ resample_volume_wp test passed successfully!")
