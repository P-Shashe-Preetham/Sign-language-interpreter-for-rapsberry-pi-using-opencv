"""
Anatomically Authentic Sign Language Landmark Generator & Importer
Generates realistic 21 3D MediaPipe landmark configurations based on true human hand kinematics.
"""
import numpy as np
import csv
from pathlib import Path
try:
    from . import config
except ImportError:
    import config


def get_base_skeleton():
    """Returns the resting 21-joint 3D coordinate template of an upright right hand."""
    # (x: right+, y: down+, z: away+)
    # MediaPipe coordinate convention: y=0 is top, y=1 is bottom, so upward fingers have negative relative y.
    return {
        # 0: Wrist
        0: np.array([0.0, 0.0, 0.0]),
        
        # 1-4: Thumb
        1: np.array([-0.12, -0.15, 0.02]),
        2: np.array([-0.22, -0.28, 0.04]),
        3: np.array([-0.30, -0.40, 0.06]),
        4: np.array([-0.38, -0.52, 0.08]), # Tip extended
        
        # 5-8: Index
        5: np.array([-0.14, -0.45, 0.0]),
        6: np.array([-0.15, -0.63, 0.01]),
        7: np.array([-0.15, -0.77, 0.01]),
        8: np.array([-0.15, -0.92, 0.01]), # Tip extended
        
        # 9-12: Middle
        9:  np.array([0.00, -0.48, 0.0]),
        10: np.array([0.00, -0.68, 0.01]),
        11: np.array([0.00, -0.83, 0.01]),
        12: np.array([0.00, -0.98, 0.01]), # Tip extended
        
        # 13-16: Ring
        13: np.array([0.13, -0.44, 0.0]),
        14: np.array([0.14, -0.61, 0.01]),
        15: np.array([0.14, -0.74, 0.01]),
        16: np.array([0.14, -0.87, 0.01]), # Tip extended
        
        # 17-20: Pinky
        17: np.array([0.23, -0.38, 0.0]),
        18: np.array([0.25, -0.52, 0.01]),
        19: np.array([0.26, -0.63, 0.01]),
        20: np.array([0.27, -0.75, 0.01]), # Tip extended
    }


def build_pose(
    thumb="extended",
    index="extended",
    middle="extended",
    ring="extended",
    pinky="extended",
    special=None
):
    """Constructs 21 3D landmarks for a specific finger articulation configuration."""
    skel = get_base_skeleton()

    # Index configuration
    if index == "curled":
        mcp = skel[5]
        skel[6] = mcp + np.array([-0.01, 0.10, -0.10])
        skel[7] = mcp + np.array([-0.01, 0.16, -0.06])
        skel[8] = mcp + np.array([-0.01, 0.14, -0.01])

    # Middle configuration
    if middle == "curled":
        mcp = skel[9]
        skel[10] = mcp + np.array([0.0, 0.11, -0.10])
        skel[11] = mcp + np.array([0.0, 0.18, -0.06])
        skel[12] = mcp + np.array([0.0, 0.16, -0.01])

    # Ring configuration
    if ring == "curled":
        mcp = skel[13]
        skel[14] = mcp + np.array([0.01, 0.10, -0.10])
        skel[15] = mcp + np.array([0.01, 0.17, -0.06])
        skel[16] = mcp + np.array([0.01, 0.15, -0.01])

    # Pinky configuration
    if pinky == "curled":
        mcp = skel[17]
        skel[18] = mcp + np.array([0.01, 0.09, -0.09])
        skel[19] = mcp + np.array([0.01, 0.15, -0.05])
        skel[20] = mcp + np.array([0.01, 0.13, -0.01])

    # Thumb configuration
    if thumb == "folded": # folded across palm / fist
        skel[2] = np.array([-0.10, -0.28, -0.08])
        skel[3] = np.array([-0.02, -0.32, -0.12])
        skel[4] = np.array([0.06, -0.34, -0.12])
    elif thumb == "up": # thumbs up
        skel[1] = np.array([-0.08, -0.20, 0.0])
        skel[2] = np.array([-0.09, -0.38, 0.02])
        skel[3] = np.array([-0.10, -0.55, 0.03])
        skel[4] = np.array([-0.10, -0.72, 0.04])
    elif thumb == "down": # thumbs down
        skel[1] = np.array([-0.08, 0.15, 0.0])
        skel[2] = np.array([-0.09, 0.32, 0.02])
        skel[3] = np.array([-0.10, 0.48, 0.03])
        skel[4] = np.array([-0.10, 0.65, 0.04])
    elif thumb == "opposed_index": # touching index (for OK sign)
        # Thumb tip and index tip meet in a ring
        touch_point = np.array([-0.08, -0.58, -0.06])
        skel[3] = np.array([-0.16, -0.46, -0.04])
        skel[4] = touch_point
        skel[7] = np.array([-0.10, -0.52, -0.04])
        skel[8] = touch_point + np.array([0.01, 0.01, 0.0])
    elif thumb == "snap_no": # touching index & middle tips (for NO sign)
        touch_point = np.array([-0.05, -0.48, -0.08])
        skel[4] = touch_point
        skel[8] = touch_point + np.array([-0.01, 0.0, 0.0])
        skel[12] = touch_point + np.array([0.01, 0.0, 0.0])

    if special == "PEACE":
        # Spread index and middle into a V
        skel[7] += np.array([-0.06, 0.0, 0.0])
        skel[8] += np.array([-0.12, 0.0, 0.0])
        skel[11] += np.array([0.06, 0.0, 0.0])
        skel[12] += np.array([0.12, 0.0, 0.0])

    coords = np.array([skel[i] for i in range(21)], dtype=np.float32)
    return coords


def get_pose_for_sign(sign_name: str) -> np.ndarray:
    """Returns canonical 21-landmark coordinates for a recognized sign."""
    s = sign_name.upper()
    if s == "HELLO":
        return build_pose(thumb="extended", index="extended", middle="extended", ring="extended", pinky="extended")
    elif s == "THANK_YOU":
        # Flat hand with fingers together, tilted forward 35 degrees (moving from chin)
        coords = build_pose(thumb="folded", index="extended", middle="extended", ring="extended", pinky="extended")
        coords = rotate_3d(coords, angle_x=np.radians(35))
        return coords
    elif s == "PEACE":
        return build_pose(thumb="folded", index="extended", middle="extended", ring="curled", pinky="curled", special="PEACE")
    elif s == "OK":
        return build_pose(thumb="opposed_index", index="curled", middle="extended", ring="extended", pinky="extended")
    elif s == "THUMBS_UP":
        return build_pose(thumb="up", index="curled", middle="curled", ring="curled", pinky="curled")
    elif s == "THUMBS_DOWN":
        return build_pose(thumb="down", index="curled", middle="curled", ring="curled", pinky="curled")
    elif s == "FIST":
        return build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
    elif s == "I_LOVE_YOU":
        return build_pose(thumb="extended", index="extended", middle="curled", ring="curled", pinky="extended")
    elif s == "YES":
        # In ASL, YES is a fist nodding downward (wrist flexed forward by 35 deg)
        base = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
        nodded = rotate_3d(base, angle_x=np.radians(35))
        return nodded
    elif s == "NO":
        # Index, middle, and thumb touching, ring & pinky curled
        return build_pose(thumb="snap_no", index="curled", middle="curled", ring="curled", pinky="curled")
    else:
        # Default open hand
        return build_pose()


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Exact normalization matching HandTracker.process_frame."""
    wrist = coords[0]
    rel = coords - wrist
    max_d = np.max(np.linalg.norm(rel, axis=1))
    if max_d > 1e-6:
        norm = rel / max_d
    else:
        norm = rel
    return norm.flatten()


def rotate_3d(coords: np.ndarray, angle_x=0.0, angle_y=0.0, angle_z=0.0) -> np.ndarray:
    """Rotates 3D landmark points by Euler angles (radians)."""
    # Z rotation (roll)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # Y rotation (yaw)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    # X rotation (pitch)
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    
    R = Rz @ Ry @ Rx
    return coords @ R.T


def generate_anatomical_dataset(samples_per_class: int = 150):
    """
    Synthesizes an anatomically authentic dataset for all configured signs,
    incorporating realistic hand variations (rotation, camera distance, joint jitter).
    """
    print(f"[INFO] Generating anatomical ground-truth dataset ({samples_per_class} samples per sign)...")
    np.random.seed(42)
    header = ["label"] + [f"feat_{i}" for i in range(63)]
    rows = []

    for sign in config.SIGNS:
        base_coords = get_pose_for_sign(sign)
        
        for _ in range(samples_per_class):
            # 1. Realistic hand rotation perturbations
            # Yaw (+- 18 deg), Pitch (+- 15 deg), Roll (+- 20 deg)
            ax = np.random.uniform(-np.radians(15), np.radians(15))
            ay = np.random.uniform(-np.radians(18), np.radians(18))
            az = np.random.uniform(-np.radians(20), np.radians(20))
            rotated = rotate_3d(base_coords.copy(), ax, ay, az)

            # 2. Individual joint flexion noise (slight biological variations)
            joint_jitter = np.random.normal(0, 0.015, rotated.shape)
            joint_jitter[0] = 0.0 # Keep wrist anchored
            perturbed = rotated + joint_jitter

            # 3. MediaPipe normalization
            feat_vector = normalize_coords(perturbed)
            rows.append([sign] + list(feat_vector))

    # Save to landmarks.csv
    with open(config.CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[SUCCESS] Generated {len(rows)} anatomical samples for {len(config.SIGNS)} signs at {config.CSV_PATH}")
    return len(rows)


if __name__ == "__main__":
    generate_anatomical_dataset()
