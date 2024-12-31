import numpy as np
import cv2
from scipy.linalg import svd
import matplotlib.pyplot as plt


def detect_qr_code(frame):
    """Detect and return the bounding box of the QR code in the frame."""
    detector = cv2.QRCodeDetector()
    _, corners, _ = detector.detectAndDecode(frame)
    if corners is not None:
        return np.array(corners[0], dtype=np.float32)  # Extract the corner points
    return None


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
    alpha = np.power(D, 1 / 4)
    alpha=-alpha# Same formula as MATLAB
    Rup = np.column_stack((R1 / alpha, R2 / alpha, R3 / (alpha ** 2)))
    Tup = T / alpha
    RTup = np.column_stack((Rup, Tup))
    P = K @ RTup
    return P


def plot_3d_cube(frame, P):
    """Project and draw a 3D cube on the image."""
    world_cube = np.array([
        [0, 0, 0, 1], [0, 2, 0, 1], [2, 0, 0, 1], [2, 2, 0, 1],
        [0, 0, 2, 1], [0, 2, 2, 1], [2, 0, 2, 1], [2, 2, 2, 1]
    ])

    # Project the cube into 2D
    cube_2d_homogeneous = P @ world_cube.T
    cube_2d = cube_2d_homogeneous / cube_2d_homogeneous[2, :]  # Normalize
    x_cube = cube_2d[0, :]
    y_cube = cube_2d[1, :]

    # Draw the cube on the frame using OpenCV
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    for p1, p2 in edges:
        cv2.line(frame, (int(x_cube[p1]), int(y_cube[p1])), (int(x_cube[p2]), int(y_cube[p2])), (0, 255, 0), 2)

    return frame


def main():
    video_path = "IMG_4620.MOV"  # Replace with your video file
    cap = cv2.VideoCapture(video_path)

    # Intrinsic matrix (example)
    K = np.array([
        [1682.793625028055, 0, 780.7846303429646],
        [0, 1682.351387650792, 1027.805128038535],
        [0, 0, 1]
    ])

    # Define world points for the QR code
    world_points = np.array([
        [0, 0], [2, 0], [2, 2], [0, 2]  # World coordinates in 3D (flat plane)
    ], dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect QR code in the current frame
        qr_corners = detect_qr_code(frame)

        if qr_corners is not None:
            # Compute the homography and projection matrix
            H = compute_homography(world_points, qr_corners)
            P = compute_projection_matrix(K, H)

            # Draw the cube on the frame
            frame = plot_3d_cube(frame, P)

        # Display the frame
        cv2.imshow("3D Cube Projection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
