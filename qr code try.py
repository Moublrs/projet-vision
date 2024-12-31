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
    """Allow the user to select a square region within the QR code."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [qr_corners.astype(int)], 255)

    # Restrict the view to the QR code area
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    _, thresh = cv2.threshold(masked_gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours for selection
    for idx, contour in enumerate(contours):
        cv2.polylines(frame, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(frame, f"{idx}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Select Region", frame)
    cv2.waitKey(0)

    region_index = int(input(f"Enter the index of the region to select: "))
    selected_region = contours[region_index]

    # Get the corner points of the selected region
    epsilon = 0.02 * cv2.arcLength(selected_region, True)
    approx = cv2.approxPolyDP(selected_region, epsilon, True)
    if len(approx) == 4:  # Ensure it's a quadrilateral
        return np.array([point[0] for point in approx], dtype=np.float32)
    else:
        print("Selected region is not a square. Please try again.")
        exit()


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
    #alpha=-alpha
    Rup = np.column_stack((R1 / alpha, R2 / alpha, R3 / (alpha ** 2)))
    Tup = T / alpha
    RTup = np.column_stack((Rup, Tup))
    P = K @ RTup
    return P


def transform_region(initial_region, homography):
    """Transform the region dynamically using the computed homography."""
    transformed_region = cv2.perspectiveTransform(initial_region.reshape(-1, 1, 2), homography)
    return transformed_region.reshape(-1, 2)


def plot_3d_cube(frame, P):
    """
    Project and draw a more elaborate 3D twisted tower with 4 levels,
    each having 20 corners (for a total of 80 base points),
    plus a single apex point at the top => 81 points total.

    Key features:
      - 4 levels (not too tall).
      - 20 corners per ring => more detail in each level.
      - Slight "twist" between consecutive levels.
      - Multiple diagonals both within each ring and between levels.
      - Color-coded edges for variety.

    Indices / Layout (for reference):
      Level 0 => [  0.. 19 ]  (z= 0)
      Level 1 => [ 20.. 39 ]  (z=-1)
      Level 2 => [ 40.. 59 ]  (z=-2)
      Level 3 => [ 60.. 79 ]  (z=-3)
      Apex    =>   80         (z=-4)
    """
    import numpy as np
    import cv2

    # ----------------------------------------------------------------
    # 1) DEFINE 3D POINTS (in homogeneous coordinates [x, y, z, 1])
    # ----------------------------------------------------------------
    # We'll define a function to create a ring with `n` corners, rotated by
    # some small angle for each "level", and gradually shrinking in radius.
    # Then we build 4 levels + 1 apex.

    def make_ring(level_idx, n=20):
        """
        Create a ring of `n` corners (default = 20) at depth z = -level_idx.
        Each ring is centered at (0.5, 0.5) in XY, with a radius that shrinks
        a bit per level, and a slight twist in rotation.
        """
        # Base radius at level 0
        base_rx = base_ry = 0.6

        # Shrink factor: how much the radius reduces each level
        shrink = 0.1 * level_idx
        rx = base_rx - shrink
        ry = base_ry - shrink
        rx = max(rx, 0.05)
        ry = max(ry, 0.05)

        # Twist angle in degrees
        twist_deg = 8.0 * level_idx
        twist_rad = np.radians(twist_deg)

        # z coordinate (one unit deeper each level)
        z = -float(level_idx)

        ring_points = []
        for k in range(n):
            # Angle around the ring
            theta = 2.0 * np.pi * (k / n)

            # Basic ellipse coords
            x_ell = rx * np.cos(theta)
            y_ell = ry * np.sin(theta)

            # Apply rotation for the twist
            x_rot = x_ell * np.cos(twist_rad) - y_ell * np.sin(twist_rad)
            y_rot = x_ell * np.sin(twist_rad) + y_ell * np.cos(twist_rad)

            # Shift center => (0.5, 0.5)
            x = 0.5 + x_rot
            y = 0.5 + y_rot

            ring_points.append([x, y, z, 1.0])
        return ring_points

    # Build the 4 levels
    level0 = make_ring(0, 20)  # indices 0..19
    level1 = make_ring(1, 20)  # indices 20..39
    level2 = make_ring(2, 20)  # indices 40..59
    level3 = make_ring(3, 20)  # indices 60..79

    # Apex (index 80), placed slightly above level3
    apex = [[0.5, 0.5, -4.0, 1.0]]

    # Combine everything
    tower_points_3d = np.array(level0 + level1 + level2 + level3 + apex, dtype=np.float32)

    # ----------------------------------------------------------------
    # 2) DEFINE EDGES
    # ----------------------------------------------------------------
    # We'll define:
    #   - ring edges (with multiple diagonals) for each level,
    #   - vertical edges (with diagonals) between consecutive levels,
    #   - apex edges from the top ring to apex (also with diagonals).
    #
    # We'll color-code each ring differently, and use white/gray for verticals.

    def ring_edges(base_idx, n, color_main, color_diag1, color_diag2,
                   thick_main=2, thick_diag=1):
        """
        Return edges for ring perimeter + multiple diagonals:
          - perimeter edges (p->p+1)
          - short diagonal (p->p+2)
          - longer diagonal (p->p+3)
        This adds more lines for a busier look.
        """
        edges_list = []
        for k in range(n):
            p1 = base_idx + k
            p_next = base_idx + ((k + 1) % n)
            p_skip2 = base_idx + ((k + 2) % n)
            p_skip3 = base_idx + ((k + 3) % n)

            # perimeter
            edges_list.append({
                'p1': p1, 'p2': p_next,
                'color': color_main, 'thickness': thick_main
            })
            # diag skip 2
            edges_list.append({
                'p1': p1, 'p2': p_skip2,
                'color': color_diag1, 'thickness': thick_diag
            })
            # diag skip 3
            edges_list.append({
                'p1': p1, 'p2': p_skip3,
                'color': color_diag2, 'thickness': thick_diag
            })
        return edges_list

    def vertical_edges(base_idx1, base_idx2, n, color_main, color_diag, thick_main=2, thick_diag=1):
        """
        Connect ring1..ring2 with vertical lines + a diagonal from corner k
        to corner (k+1) in the upper ring => more bracing.
        """
        edges_list = []
        for k in range(n):
            p1 = base_idx1 + k
            p2 = base_idx2 + k
            # vertical
            edges_list.append({
                'p1': p1, 'p2': p2,
                'color': color_main, 'thickness': thick_main
            })

            # diagonal to next corner in upper ring
            p3 = base_idx2 + ((k + 1) % n)
            edges_list.append({
                'p1': p1, 'p2': p3,
                'color': color_diag, 'thickness': thick_diag
            })
        return edges_list

    def apex_edges(apex_idx, base_idx, n, color_main, color_diag, thick_main=2, thick_diag=1):
        """
        Connect the apex to each corner of the top ring, plus diagonals
        from apex to the next corner on the ring for extra complexity.
        """
        edges_list = []
        for k in range(n):
            corner = base_idx + k
            corner_next = base_idx + ((k + 1) % n)

            # main line
            edges_list.append({
                'p1': apex_idx, 'p2': corner,
                'color': color_main, 'thickness': thick_main
            })
            # diagonal line apex -> next corner
            edges_list.append({
                'p1': apex_idx, 'p2': corner_next,
                'color': color_diag, 'thickness': thick_diag
            })
        return edges_list

    # Ring edges: different colors for each level
    # (B, G, R) format. Feel free to adjust or randomize.
    ring0 = ring_edges(
        base_idx=0, n=20,
        color_main=(0, 255, 0),      # green
        color_diag1=(0, 180, 0),
        color_diag2=(0, 130, 0),
        thick_main=2, thick_diag=1
    )
    ring1 = ring_edges(
        base_idx=20, n=20,
        color_main=(255, 255, 0),    # cyan
        color_diag1=(180, 180, 0),
        color_diag2=(130, 130, 0),
        thick_main=2, thick_diag=1
    )
    ring2 = ring_edges(
        base_idx=40, n=20,
        color_main=(255, 0, 255),    # magenta
        color_diag1=(180, 0, 180),
        color_diag2=(130, 0, 130),
        thick_main=2, thick_diag=1
    )
    ring3 = ring_edges(
        base_idx=60, n=20,
        color_main=(0, 255, 255),    # yellow-ish
        color_diag1=(0, 180, 180),
        color_diag2=(0, 130, 130),
        thick_main=2, thick_diag=1
    )

    # Vertical edges between rings (white + gray diagonals)
    v01 = vertical_edges(0, 20, 20, (255, 255, 255), (180, 180, 180))
    v12 = vertical_edges(20, 40, 20, (255, 255, 255), (180, 180, 180))
    v23 = vertical_edges(40, 60, 20, (255, 255, 255), (180, 180, 180))

    # Apex edges (red + darker red diagonals)
    # top ring => indices [60..79], apex => index=80
    apex_ed = apex_edges(
        apex_idx=80, base_idx=60, n=20,
        color_main=(0, 0, 255),     # red
        color_diag=(0, 0, 180),
        thick_main=2, thick_diag=1
    )

    # Combine all edges
    edges = (ring0 + ring1 + ring2 + ring3 +
             v01 + v12 + v23 +
             apex_ed)

    # ----------------------------------------------------------------
    # 3) PROJECT 3D POINTS INTO 2D
    # ----------------------------------------------------------------
    projected_hom = P @ tower_points_3d.T  # shape => (3, 81)
    z_vals = projected_hom[2, :] + 1e-9
    projected_2d = projected_hom / z_vals
    x_coords = projected_2d[0, :]
    y_coords = projected_2d[1, :]

    # ----------------------------------------------------------------
    # 4) DRAW ALL EDGES
    # ----------------------------------------------------------------
    for e in edges:
        p1, p2 = e['p1'], e['p2']
        color = e['color']
        thickness = e['thickness']

        x1, y1 = int(x_coords[p1]), int(y_coords[p1])
        x2, y2 = int(x_coords[p2]), int(y_coords[p2])
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

    return frame


def main():
    video_path = "IMG_4636.MOV"  # Replace with your video file
    cap = cv2.VideoCapture(video_path)

    # Intrinsic matrix (example)
    K = np.array([
        [1682.793625028055, 0, 780.7846303429646],
        [0, 1682.351387650792, 1027.805128038535],
        [0, 0, 1]
    ])

    # Define world points for a flat square region
    world_points = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1]  # Coordinates for a flat plane
    ], dtype=np.float32)

    initial_qr_corners = None
    selected_region = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect QR code in the current frame
        qr_corners = detect_qr_code(frame)

        if qr_corners is not None:
            if initial_qr_corners is None:
                # First frame: Allow the user to select a square region
                initial_qr_corners = qr_corners
                selected_region = select_region(frame.copy(), qr_corners)
            else:
                # Compute the homography between the initial QR corners and current QR corners
                H = compute_homography(initial_qr_corners, qr_corners)

                # Transform the selected region dynamically
                dynamic_region = transform_region(selected_region, H)

                # Compute projection matrix for the transformed region
                H_region = compute_homography(world_points, dynamic_region)
                P = compute_projection_matrix(K, H_region)

                # Draw the cube on the dynamically updated region
                frame = plot_3d_cube(frame, P)

                # Draw the transformed region for visualization
                #cv2.polylines(frame, [dynamic_region.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)

        # Display the frame
        cv2.imshow("3D Cube Projection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
