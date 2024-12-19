from functions import *

def main():
    image_path = "APC_0016.jpg"
    image = load_image(image_path)

    world_points = np.array([
        [2, 1], [8, 1], [4, 2], [6, 3], [2, 4], [4, 4], [8, 4], [7, 5]
    ])
    image_points = np.array([
        [1971, 1159],
        [1981, 2140],
        [1802, 1539],
        [1641, 1890],
        [1415, 1202],
        [1452, 1566],
        [1506, 2193],
        [1330, 2059]
    ])

    H = compute_homography(world_points, image_points)

    K = np.array([
        [1682.793625028055, 0, 780.7846303429646],
        [0, 1682.351387650792, 1027.805128038535],
        [0, 0, 1]
    ])
    P = compute_projection_matrix(K, H)

    plot_3d_cube(image, P)


if __name__ == "__main__":
    main()
