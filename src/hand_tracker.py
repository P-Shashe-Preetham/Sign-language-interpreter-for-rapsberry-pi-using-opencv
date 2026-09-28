import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List

try:
    from . import config
except ImportError:
    import config


class HandTracker:
    """
    Wrapper around MediaPipe Hands for high-performance landmark extraction
    with translation and scale normalization invariant to hand position and distance.
    """
    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], any]:
        """
        Process a single BGR camera frame.
        
        Returns:
            - normalized_features: 1D numpy array of 63 normalized coordinates (or None if no hand)
            - bbox: (x_min, y_min, x_max, y_max) in pixel coordinates (or None)
            - raw_landmarks: MediaPipe NormalizedLandmarkList for drawing
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None, None, None

        # Take the primary hand detected
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract 21 landmarks (x, y, z)
        coords = []
        x_pixels, y_pixels = [], []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
            x_pixels.append(int(lm.x * w))
            y_pixels.append(int(lm.y * h))

        coords = np.array(coords, dtype=np.float32)  # Shape (21, 3)

        # 1. Translation Invariance: Shift wrist (landmark 0) to origin (0, 0, 0)
        wrist = coords[0]
        relative_coords = coords - wrist

        # 2. Scale Invariance: Divide by max Euclidean distance from the wrist
        distances = np.linalg.norm(relative_coords, axis=1)
        max_dist = np.max(distances)
        if max_dist > 1e-6:
            normalized_coords = relative_coords / max_dist
        else:
            normalized_coords = relative_coords

        # Flatten into 1D feature vector of length 63
        features = normalized_coords.flatten()

        # Compute bounding box with padding
        padding = 20
        x_min = max(0, min(x_pixels) - padding)
        y_min = max(0, min(y_pixels) - padding)
        x_max = min(w, max(x_pixels) + padding)
        y_max = min(h, max(y_pixels) + padding)
        bbox = (x_min, y_min, x_max, y_max)

        return features, bbox, hand_landmarks

    def draw_landmarks(self, frame_bgr: np.ndarray, hand_landmarks) -> None:
        """Draws skeletal hand connections and landmark joints on frame."""
        if hand_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

    def close(self):
        self.hands.close()
