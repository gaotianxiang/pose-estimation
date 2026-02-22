import numpy as np
import matplotlib.pyplot as plt

R_ANKLE = 0
R_KNEE = 1
R_HIP = 2
L_HIP = 3
L_KNEE = 4
L_ANKLE = 5
PELVIS = 6
THORAX = 7
UPPER_NECK = 8
HEAD_TOP = 9
R_WRIST = 10
R_ELBOW = 11
R_SHOULDER = 12
L_SHOULDER = 13
L_ELBOW = 14
L_WRIST = 15

MPII_BONES = [
    [R_ANKLE, R_KNEE],
    [R_KNEE, R_HIP],
    [R_HIP, PELVIS],
    [L_HIP, PELVIS],
    [L_HIP, L_KNEE],
    [L_KNEE, L_ANKLE],
    [PELVIS, THORAX],
    [THORAX, UPPER_NECK],
    [UPPER_NECK, HEAD_TOP],
    [R_WRIST, R_ELBOW],
    [R_ELBOW, R_SHOULDER],
    [THORAX, R_SHOULDER],
    [THORAX, L_SHOULDER],
    [L_SHOULDER, L_ELBOW],
    [L_ELBOW, L_WRIST],
]


def find_max_coordinates(heatmaps: np.ndarray) -> np.ndarray:
    """Return (x, y) peak coordinates for each of 16 keypoints.

    Args:
        heatmaps: shape (64, 64, 16)

    Returns:
        shape (16, 2) with columns [x, y]
    """
    flat = heatmaps.reshape(4096, 16)
    indices = np.argmax(flat, axis=0)          # (16,)
    y = indices // 64
    x = indices - 64 * y
    return np.stack([x, y], axis=1)            # (16, 2)


def extract_keypoints_from_heatmap(heatmaps: np.ndarray) -> np.ndarray:
    """Extract sub-pixel keypoint locations from heatmaps.

    Args:
        heatmaps: shape (64, 64, 16), float32

    Returns:
        normalised keypoints, shape (16, 2), values in [0, 1]
    """
    max_keypoints = find_max_coordinates(heatmaps)
    # Pad so border maxima can be handled uniformly
    padded = np.pad(heatmaps, [[1, 1], [1, 1], [0, 0]])

    adjusted = []
    for i, kp in enumerate(max_keypoints):
        # Shift by 1 due to padding
        max_y = int(kp[1]) + 1
        max_x = int(kp[0]) + 1

        patch = padded[max_y - 1:max_y + 2, max_x - 1:max_x + 2, i].copy()
        patch[1, 1] = 0  # zero out the peak to find the next-best neighbour

        idx = np.argmax(patch)
        ny = idx // 3
        nx = idx - ny * 3
        delta_y = (ny - 1) / 4
        delta_x = (nx - 1) / 4

        adjusted.append((kp[0] + delta_x, kp[1] + delta_y))

    adjusted = np.clip(np.array(adjusted), 0, 64)
    return adjusted / 64  # normalise to [0, 1]


def draw_keypoints_on_image(image: np.ndarray, keypoints: np.ndarray,
                             index=None, save_path: str = None):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i, joint in enumerate(keypoints):
        if index is not None and index != i:
            continue
        jx = joint[0] * image.shape[1]
        jy = joint[1] * image.shape[0]
        plt.scatter(jx, jy, s=10, c='red', marker='o')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def draw_skeleton_on_image(image: np.ndarray, keypoints: np.ndarray, save_path: str):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    joints = [(joint[0] * image.shape[1], joint[1] * image.shape[0])
               for joint in keypoints]
    for bone in MPII_BONES:
        j1, j2 = joints[bone[0]], joints[bone[1]]
        plt.plot([j1[0], j2[0]], [j1[1], j2[1]], linewidth=5, alpha=0.7)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close(fig)
