import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def load_image(image_path):
    """Load and display the image."""
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.show()
    return image


def compute_homography(world_points, image_points):
    """Compute the homography matrix H using DLT."""
    num_points = len(world_points)
    A = np.zeros((2 * num_points, 9))
    for i in range(num_points):
        X, Y = world_points[i]
        x, y = image_points[i]
        A[2 * i] = [-X, -Y, -1, 0, 0, 0, x * X, x * Y, x]
        A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, y * X, y * Y, y]

    _, _, V = svd(A)
    H = V[-1].reshape((3, 3))
    return H


def compute_projection_matrix(K, H):
    """Compute the projection matrix P."""
    RT = np.linalg.inv(K) @ H
    R1, R2, T = RT[:, 0], RT[:, 1], RT[:, 2]
    R3 = np.cross(R1, R2)
    R = np.column_stack((R1, R2, R3))
    D = np.linalg.det(R)
    alpha = np.power(D, 1 / 4)  # Same formula as MATLAB
    Rup = np.column_stack((R1 / alpha, R2 / alpha, R3 / (alpha ** 2)))
    Tup = T / alpha
    RTup = np.column_stack((Rup, Tup))
    P = K @ RTup
    return P


def plot_3d_cube(image, P):
    """Project and plot a 3D cube on the image."""
    world_cube = np.array([
        [0, 0, 0, 1], [0, 2, 0, 1], [2, 0, 0, 1], [2, 2, 0, 1],
        [0, 0, 2, 1], [0, 2, 2, 1], [2, 0, 2, 1], [2, 2, 2, 1]
    ])

    cube_2d_homogeneous = P @ world_cube.T
    cube_2d = cube_2d_homogeneous / cube_2d_homogeneous[2, :]
    x_cube = cube_2d[0, :]
    y_cube = cube_2d[1, :]

    plt.imshow(image)
    plt.plot(x_cube, y_cube, 'yx', markersize=10, linewidth=2)

    # Edges of the cube
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    for p1, p2 in edges:
        plt.plot([x_cube[p1], x_cube[p2]], [y_cube[p1], y_cube[p2]], 'm-', linewidth=2)

    # Highlight cube points
    plt.plot(x_cube, y_cube, 'bo', markerfacecolor='g', markersize=8)
    plt.title("3D Cube Projection onto the Image")
    plt.show()


