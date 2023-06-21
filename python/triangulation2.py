import numpy as np

def triangulate_point(camera_poses, pixel_coords):
    if len(camera_poses) != len(pixel_coords):
        raise ValueError("Number of camera poses and pixel coordinates must match.")

    num_cameras = len(camera_poses)

    # Projection matrices per camera pose
    projection_matrices = []
    for i in range(num_cameras):
        rotation_matrix, translation_vector = camera_poses[i]
        projection_matrix = np.concatenate((rotation_matrix, translation_vector), axis=1)
        projection_matrices.append(projection_matrix)

    # Set up a coefficient matrix for the linear system
    A = np.zeros((2 * num_cameras, 4))
    for i in range(num_cameras):
        P = projection_matrices[i]
        u, v = pixel_coords[i]

        A[2 * i] = u * P[2] - P[0]
        A[2 * i + 1] = v * P[2] - P[1]

    # Solve the linear system using SVD
    _, _, V = np.linalg.svd(A)

	# Normalize homogeneous coordinates
    point_homogeneous = V[-1]
    point_homogeneous /= point_homogeneous[-1]  

    # Extract a 3D point in world space
    point_3d = point_homogeneous[:3]

    return point_3d

camera_poses = [
    (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0], [0], [0]])),
    (np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]]), np.array([[2], [0], [0]])),
    (np.array([[0.5, -0.866, 0], [0.866, 0.5, 0], [0, 0, 1]]), np.array([[0], [2], [0]])),
    (np.array([[-0.866, -0.5, 0], [0.5, -0.866, 0], [0, 0, 1]]), np.array([[2], [2], [0]]))
]

pixel_coords = [
    (100, 100),
    (200, 150),
    (150, 200),
    (250, 250)
]

point_3d = triangulate_point(camera_poses, pixel_coords)
print("Triangulated 3D point:", point_3d)
