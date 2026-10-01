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
    """Returns canonical 21-landmark coordinates for a recognized sign or ASL letter."""
    s = sign_name.upper()

    # ---------------------------------------------
    # 1. ASL ALPHABET (A - Z)
    # ---------------------------------------------
    if s == "A":
        # Closed fist with thumb straight up resting along side of index finger
        skel = build_pose(thumb="alongside", index="curled", middle="curled", ring="curled", pinky="curled")
        skel[2] = np.array([-0.18, -0.28, 0.02])
        skel[3] = np.array([-0.18, -0.42, 0.03])
        skel[4] = np.array([-0.18, -0.56, 0.03])
        return skel
    elif s == "B":
        # 4 fingers straight up, held flat & tight together. Thumb folded across palm.
        skel = build_pose(thumb="folded", index="extended", middle="extended", ring="extended", pinky="extended")
        skel[5:9, 0] = np.linspace(-0.06, -0.06, 4)
        skel[9:13, 0] = np.linspace(-0.01, -0.01, 4)
        skel[13:17, 0] = np.linspace(0.04, 0.04, 4)
        skel[17:21, 0] = np.linspace(0.09, 0.09, 4)
        return skel
    elif s == "C":
        # Hand curved in a 'C' cup shape
        skel = get_base_skeleton()
        for base_idx in [5, 9, 13, 17]:
            skel[base_idx+1] += np.array([0.0, 0.08, -0.15])
            skel[base_idx+2] += np.array([0.0, 0.22, -0.22])
            skel[base_idx+3] += np.array([0.0, 0.38, -0.18])
        skel[2] = np.array([-0.20, -0.25, -0.05])
        skel[3] = np.array([-0.16, -0.32, -0.15])
        skel[4] = np.array([-0.08, -0.35, -0.18])
        return np.array([skel[i] for i in range(21)], dtype=np.float32)
    elif s == "D":
        # Index pointing straight up, remaining 3 fingers touch thumb tip in circle
        skel = build_pose(thumb="folded", index="extended", middle="curled", ring="curled", pinky="curled")
        d_touch = np.array([-0.06, -0.38, -0.08])
        skel[4] = d_touch
        skel[12] = d_touch + np.array([0.01, 0.01, 0.0])
        skel[16] = d_touch + np.array([0.02, 0.02, 0.0])
        return skel
    elif s == "E":
        # All fingers tightly curled with fingernails resting on thumb folded below
        skel = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
        skel[4] = np.array([-0.05, -0.25, -0.08])
        for tip in [8, 12, 16, 20]:
            skel[tip][1] = -0.28
            skel[tip][2] = -0.07
        return skel
    elif s == "F":
        # Index & thumb touching in O ring, middle, ring, pinky straight up
        return build_pose(thumb="opposed_index", index="curled", middle="extended", ring="extended", pinky="extended")
    elif s == "G":
        # Index pointing horizontally left/forward, thumb parallel. Other 3 curled.
        skel = build_pose(thumb="alongside", index="curled", middle="curled", ring="curled", pinky="curled")
        skel[6] = skel[5] + np.array([-0.15, -0.02, 0.0])
        skel[7] = skel[6] + np.array([-0.15, -0.02, 0.0])
        skel[8] = skel[7] + np.array([-0.15, -0.02, 0.0])
        skel[3] = skel[2] + np.array([-0.12, -0.02, 0.02])
        skel[4] = skel[3] + np.array([-0.14, -0.02, 0.02])
        return skel
    elif s == "H":
        # Index & middle extended horizontally side-by-side. Thumb curled over ring/pinky.
        skel = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
        for tip_base in [5, 9]:
            skel[tip_base+1] = skel[tip_base] + np.array([-0.16, -0.02, 0.0])
            skel[tip_base+2] = skel[tip_base+1] + np.array([-0.16, -0.02, 0.0])
            skel[tip_base+3] = skel[tip_base+2] + np.array([-0.16, -0.02, 0.0])
        return skel
    elif s == "I":
        # Pinky straight up, other fingers curled with thumb folded over
        return build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="extended")
    elif s == "J":
        # Pinky extended with hook trace rotation
        skel = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="extended")
        return rotate_3d(skel, angle_z=np.radians(-25), angle_y=np.radians(20))
    elif s == "K":
        # Index straight up, middle finger forward 45 deg, thumb resting on middle PIP
        skel = build_pose(thumb="folded", index="extended", middle="curled", ring="curled", pinky="curled")
        skel[10] = skel[9] + np.array([0.0, -0.15, -0.12])
        skel[11] = skel[10] + np.array([0.0, -0.14, -0.14])
        skel[12] = skel[11] + np.array([0.0, -0.14, -0.14])
        skel[4] = skel[10] + np.array([-0.02, 0.02, 0.02])
        return skel
    elif s == "L":
        # L shape: Thumb pointing sideways 90 deg, index pointing straight up
        skel = build_pose(thumb="folded", index="extended", middle="curled", ring="curled", pinky="curled")
        skel[1] = np.array([-0.12, -0.12, 0.0])
        skel[2] = np.array([-0.28, -0.14, 0.0])
        skel[3] = np.array([-0.44, -0.16, 0.0])
        skel[4] = np.array([-0.60, -0.18, 0.0])
        return skel
    elif s == "M":
        # Fist with thumb tucked under 3 fingers (peeking between ring & pinky)
        skel = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
        skel[4] = np.array([0.18, -0.28, -0.06])
        return skel
    elif s == "N":
        # Fist with thumb tucked under 2 fingers (peeking between middle & ring)
        skel = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
        skel[4] = np.array([0.05, -0.28, -0.06])
        return skel
    elif s == "O":
        # All 5 fingers touching tips forming an O
        skel = get_base_skeleton()
        o_center = np.array([-0.05, -0.45, -0.18])
        for tip in [4, 8, 12, 16, 20]:
            skel[tip] = o_center
        for base in [5, 9, 13, 17]:
            skel[base+1] += np.array([0.0, 0.08, -0.12])
            skel[base+2] += np.array([0.0, 0.18, -0.18])
        skel[2] = np.array([-0.16, -0.22, -0.08])
        skel[3] = np.array([-0.12, -0.34, -0.14])
        return np.array([skel[i] for i in range(21)], dtype=np.float32)
    elif s == "P":
        # K shape pitched downward 80 deg
        k_pose = get_pose_for_sign("K")
        return rotate_3d(k_pose, angle_x=np.radians(80))
    elif s == "Q":
        # G shape pitched downward 80 deg
        g_pose = get_pose_for_sign("G")
        return rotate_3d(g_pose, angle_x=np.radians(80))
    elif s == "R":
        # Index & middle crossed
        skel = build_pose(thumb="folded", index="extended", middle="extended", ring="curled", pinky="curled")
        skel[11] += np.array([-0.08, 0.0, -0.03])
        skel[12] += np.array([-0.16, 0.0, -0.05])
        skel[7] += np.array([0.04, 0.0, 0.02])
        skel[8] += np.array([0.08, 0.0, 0.04])
        return skel
    elif s == "S":
        # Closed fist with thumb wrapped across knuckles
        skel = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
        skel[4] = np.array([0.02, -0.32, -0.14])
        return skel
    elif s == "T":
        # Fist with thumb tucked under index (peeking between index & middle)
        skel = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
        skel[4] = np.array([-0.08, -0.32, -0.08])
        return skel
    elif s == "U":
        # Index & middle straight up and touching together
        skel = build_pose(thumb="folded", index="extended", middle="extended", ring="curled", pinky="curled")
        skel[5:9, 0] = np.linspace(-0.04, -0.04, 4)
        skel[9:13, 0] = np.linspace(0.01, 0.01, 4)
        return skel
    elif s == "V":
        # Peace / V sign (index and middle spread apart)
        return build_pose(thumb="folded", index="extended", middle="extended", ring="curled", pinky="curled", special="PEACE")
    elif s == "W":
        # W shape: index, middle, ring straight up and spread
        skel = build_pose(thumb="folded", index="extended", middle="extended", ring="extended", pinky="curled")
        skel[7] += np.array([-0.08, 0.0, 0.0])
        skel[8] += np.array([-0.14, 0.0, 0.0])
        skel[15] += np.array([0.08, 0.0, 0.0])
        skel[16] += np.array([0.14, 0.0, 0.0])
        return skel
    elif s == "X":
        # Index finger bent into hook
        skel = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
        skel[6] = skel[5] + np.array([0.0, -0.16, 0.0])
        skel[7] = skel[6] + np.array([0.0, -0.08, -0.12])
        skel[8] = skel[7] + np.array([0.0, 0.04, -0.10])
        return skel
    elif s == "Y":
        # Shaka / Hang loose: Thumb & pinky out, middle 3 curled
        skel = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="extended")
        skel[1] = np.array([-0.12, -0.12, 0.0])
        skel[2] = np.array([-0.26, -0.18, 0.0])
        skel[3] = np.array([-0.40, -0.24, 0.0])
        skel[4] = np.array([-0.54, -0.30, 0.0])
        skel[18] += np.array([0.06, 0.0, 0.0])
        skel[19] += np.array([0.12, 0.0, 0.0])
        skel[20] += np.array([0.18, 0.0, 0.0])
        return skel
    elif s == "Z":
        # Index pointing forward tracing Z
        skel = build_pose(thumb="folded", index="extended", middle="curled", ring="curled", pinky="curled")
        return rotate_3d(skel, angle_y=np.radians(25), angle_z=np.radians(10))

    # ---------------------------------------------
    # 2. CONVERSATIONAL SIGNS
    # ---------------------------------------------
    elif s == "HELLO":
        return build_pose(thumb="extended", index="extended", middle="extended", ring="extended", pinky="extended")
    elif s == "THANK_YOU":
        coords = build_pose(thumb="folded", index="extended", middle="extended", ring="extended", pinky="extended")
        return rotate_3d(coords, angle_x=np.radians(35))
    elif s == "THUMBS_UP":
        return build_pose(thumb="up", index="curled", middle="curled", ring="curled", pinky="curled")
    elif s == "THUMBS_DOWN":
        return build_pose(thumb="down", index="curled", middle="curled", ring="curled", pinky="curled")
    elif s == "I_LOVE_YOU":
        return build_pose(thumb="extended", index="extended", middle="curled", ring="curled", pinky="extended")
    elif s == "YES":
        base = build_pose(thumb="folded", index="curled", middle="curled", ring="curled", pinky="curled")
        return rotate_3d(base, angle_x=np.radians(35))
    elif s == "NO":
        return build_pose(thumb="snap_no", index="curled", middle="curled", ring="curled", pinky="curled")
    elif s == "PEACE":
        return get_pose_for_sign("V")
    elif s == "OK":
        return get_pose_for_sign("F")
    elif s == "FIST":
        return get_pose_for_sign("S")
    else:
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
        
        for _ in range(samples_per_class // 2):
            # 1. Realistic hand rotation perturbations
            # Yaw (+- 20 deg), Pitch (+- 18 deg), Roll (+- 25 deg)
            ax = np.random.uniform(-np.radians(18), np.radians(18))
            ay = np.random.uniform(-np.radians(20), np.radians(20))
            az = np.random.uniform(-np.radians(25), np.radians(25))
            rotated = rotate_3d(base_coords.copy(), ax, ay, az)

            # 2. Individual joint flexion noise (slight biological variations)
            joint_jitter = np.random.normal(0, 0.02, rotated.shape)
            joint_jitter[0] = 0.0 # Keep wrist anchored
            perturbed = rotated + joint_jitter

            # 3. Right hand sample
            feat_right = normalize_coords(perturbed)
            rows.append([sign] + list(feat_right))

            # 4. Left hand / Mirrored sample (negate X coordinate)
            mirrored = perturbed.copy()
            mirrored[:, 0] = -mirrored[:, 0]
            feat_left = normalize_coords(mirrored)
            rows.append([sign] + list(feat_left))

    # Save to landmarks.csv
    with open(config.CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[SUCCESS] Generated {len(rows)} anatomical samples for {len(config.SIGNS)} signs at {config.CSV_PATH}")
    return len(rows)


if __name__ == "__main__":
    generate_anatomical_dataset()
