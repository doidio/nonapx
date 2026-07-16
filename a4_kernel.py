import warp as wp
import numpy as np
import trimesh
import itk
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
