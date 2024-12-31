import numpy as np
import cv2
from scipy.linalg import svd


def detect_qr_code(frame):
    """Detect and return the bounding box of the QR code in the frame."""
    detector = cv2.QRCodeDetector()
    _, corners, _ = detector.detectAndDecode(frame)
    if corners is not None:
        return np.array(corners[0], dtype=np.float32)  # Extract the corner points
    return None


def select_region(frame, qr_corners):
    """Allow the user to select a region within the QR code."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [qr_corners.astype(int)], 255)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    _, thresh = cv2.threshold(masked_gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        cv2.polylines(frame, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(frame, f"{idx}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Select Region", frame)
    cv2.waitKey(0)

    region_index = int(input(f"Enter the index of the region to select: "))
    selected_region = contours[region_index]
    return selected_region


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
    Rup = np.column_stack((R1 / alpha, R2 / alpha, R3 / (alpha ** 2)))
    Tup = T / alpha
    RTup = np.column_stack((Rup, Tup))
    P = K @ RTup
    return P


def plot_3d_chair(frame, P):
    """Project and draw a 3D chair on the selected region."""
    # Define world points for the chair
    world_chair = np.array([
        [0, 0, 0, 1], [1.5, 0, 0, 1], [1.5, 1.5, 0, 1], [0, 1.5, 0, 1],  # Seat corners
        [0.2, 0.2, -1.5, 1], [0.2, 1.3, -1.5, 1],  # Front-left leg
        [1.3, 0.2, -1.5, 1], [1.3, 1.3, -1.5, 1],  # Front-right leg
        [0, 0, 1, 1], [1.5, 0, 1, 1], [1.5, 1.5, 1, 1], [0, 1.5, 1, 1]  # Backrest
    ])

    # Project the chair into 2D
    chair_2d_homogeneous = P @ world_chair.T
    chair_2d = chair_2d_homogeneous / chair_2d_homogeneous[2, :]  # Normalize
    x_chair = chair_2d[0, :]
    y_chair = chair_2d[1, :]

    # Draw the seat
    seat_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for p1, p2 in seat_edges:
        cv2.line(frame, (int(x_chair[p1]), int(y_chair[p1])), (int(x_chair[p2]), int(y_chair[p2])), (255, 0, 0), 2)

    # Draw the legs
    leg_edges = [(0, 4), (1, 6), (2, 7), (3, 5)]
    for p1, p2 in leg_edges:
        cv2.line(frame, (int(x_chair[p1]), int(y_chair[p1])), (int(x_chair[p2]), int(y_chair[p2])), (0, 255, 0), 2)

    # Draw the backrest
    backrest_edges = [(8, 9), (9, 10), (10, 11), (11, 8)]
    for p1, p2 in backrest_edges:
        cv2.line(frame, (int(x_chair[p1]), int(y_chair[p1])), (int(x_chair[p2]), int(y_chair[p2])), (0, 0, 255), 2)

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

    world_points = np.array([[0, 0], [1.5, 0], [1.5, 1.5], [0, 1.5]], dtype=np.float32)
    selected_region = None
    initial_qr_corners = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        qr_corners = detect_qr_code(frame)

        if qr_corners is not None:
            if selected_region is None:
                initial_qr_corners = qr_corners
                selected_region = select_region(frame.copy(), qr_corners)
            else:
                H = compute_homography(world_points, qr_corners)
                P = compute_projection_matrix(K, H)
                frame = plot_3d_chair(frame, P)

        cv2.imshow("3D Chair Projection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
