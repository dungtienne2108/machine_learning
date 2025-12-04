"""
Active Liveness Detection Module - Eye Blink Detection
Sử dụng Eye Aspect Ratio (EAR) để phát hiện nháy mắt người dùng
"""

import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from typing import Tuple, Dict, Optional


class LivenessDetector:
    """
    Module phát hiện độ sống (Liveness) dựa trên nháy mắt

    Công thức Eye Aspect Ratio (EAR):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Các điểm landmark mắt (6 điểm):
    - p1: mắt trái/phải (góc ngoài cùng)
    - p2: mắt trên (phần trên ngoài)
    - p3: mắt trên (phần trên trong)
    - p4: mắt phải/trái (góc trong cùng)
    - p5: mắt dưới (phần dưới trong)
    - p6: mắt dưới (phần dưới ngoài)
    """

    def __init__(self, ear_threshold: float = 0.2, consecutive_frames: int = 3):
        """
        Khởi tạo Liveness Detector

        Args:
            ear_threshold: Ngưỡng Eye Aspect Ratio để xem như mắt đóng (< 0.2 = đóng)
            consecutive_frames: Số frame liên tiếp phải nhắm mắt để tính là một lần nháy
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames

        # Mediapipe Face Mesh để lấy landmark
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Chỉ số landmark cho mắt phải và trái (theo MediaPipe)
        # Mắt phải: 33-133
        # Mắt trái: 159-374
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

        # Theo dõi trạng thái nháy
        self.eye_closed_frames = 0  # Số frame liên tiếp mắt đóng
        self.blink_count = 0  # Tổng số lần nháy
        self.blink_detected = False  # Có phát hiện nháy trong frame này

        # Lịch sử EAR để debug
        self.ear_history = deque(maxlen=30)

        # Liveness passed hay chưa
        self.liveness_passed = False

    def compute_eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """
        Tính Eye Aspect Ratio (EAR) từ 6 điểm landmark của một mắt

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Args:
            eye_points: numpy array hình dạng (6, 2) chứa tọa độ các điểm mắt

        Returns:
            float: Giá trị EAR (càng nhỏ = mắt càng đóng)
        """
        # Khoảng cách dọc (vertical distances)
        vertical_dist_1 = np.linalg.norm(eye_points[1] - eye_points[5])  # p2-p6
        vertical_dist_2 = np.linalg.norm(eye_points[2] - eye_points[4])  # p3-p5

        # Khoảng cách ngang (horizontal distance)
        horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])  # p1-p4

        # Tính EAR
        ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist + 1e-6)

        return ear

    def get_eye_points(self, landmarks, eye_indices: list) -> Optional[np.ndarray]:
        """
        Lấy tọa độ các điểm landmark của một mắt

        Args:
            landmarks: Danh sách tất cả các landmark từ MediaPipe
            eye_indices: Chỉ số của các điểm mắt

        Returns:
            numpy array hình dạng (6, 2) hoặc None nếu không tìm thấy
        """
        if landmarks is None:
            return None

        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                eye_points.append([lm.x, lm.y])

        if len(eye_points) == 6:
            return np.array(eye_points, dtype=np.float32)
        return None

    def detect_blink(self, frame: np.ndarray) -> Dict:
        """
        Phát hiện nháy mắt từ một frame video

        Args:
            frame: Khung hình từ video (BGR format từ OpenCV)

        Returns:
            dict chứa:
                - 'eye_open': bool, trạng thái mắt (mở/đóng)
                - 'right_ear': float, EAR của mắt phải
                - 'left_ear': float, EAR của mắt trái
                - 'blink_detected': bool, có phát hiện nháy
                - 'blink_count': int, tổng số nháy
                - 'liveness_passed': bool, đã xác minh liveness
        """
        self.blink_detected = False

        # Chuyển sang RGB (MediaPipe yêu cầu RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Phát hiện landmarks
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        result_dict = {
            'eye_open': True,
            'right_ear': 0.0,
            'left_ear': 0.0,
            'blink_detected': False,
            'blink_count': self.blink_count,
            'liveness_passed': self.liveness_passed,
            'landmarks': None
        }

        if results.multi_face_landmarks is None or len(results.multi_face_landmarks) == 0:
            return result_dict

        landmarks = results.multi_face_landmarks[0].landmark
        result_dict['landmarks'] = landmarks

        # Lấy điểm mắt phải và trái
        right_eye = self.get_eye_points(landmarks, self.RIGHT_EYE_INDICES)
        left_eye = self.get_eye_points(landmarks, self.LEFT_EYE_INDICES)

        if right_eye is None or left_eye is None:
            return result_dict

        # Tính EAR cho cả hai mắt
        right_ear = self.compute_eye_aspect_ratio(right_eye)
        left_ear = self.compute_eye_aspect_ratio(left_eye)

        # Lấy giá trị trung bình EAR
        avg_ear = (right_ear + left_ear) / 2.0

        result_dict['right_ear'] = right_ear
        result_dict['left_ear'] = left_ear

        # Lưu lịch sử EAR
        self.ear_history.append(avg_ear)

        # Kiểm tra xem mắt có đóng không
        if avg_ear < self.ear_threshold:
            self.eye_closed_frames += 1
        else:
            # Nếu mắt mở lại sau khi đóng, tính là một lần nháy
            if self.eye_closed_frames >= self.consecutive_frames:
                self.blink_count += 1
                self.blink_detected = True

                if not self.liveness_passed:
                    self.liveness_passed = True
                    print("[✓] Liveness Passed – Blink detected")

            self.eye_closed_frames = 0

        result_dict['eye_open'] = avg_ear >= self.ear_threshold
        result_dict['blink_detected'] = self.blink_detected
        result_dict['blink_count'] = self.blink_count
        result_dict['liveness_passed'] = self.liveness_passed

        return result_dict

    def draw_eye_status(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Vẽ thông tin trạng thái mắt lên frame

        Args:
            frame: Khung hình gốc
            result: Kết quả từ detect_blink()

        Returns:
            Frame sau khi vẽ
        """
        frame_h, frame_w = frame.shape[:2]

        # Vẽ trạng thái mắt
        status = "OPEN" if result['eye_open'] else "CLOSED"
        status_color = (0, 255, 0) if result['eye_open'] else (0, 0, 255)

        cv2.putText(
            frame,
            f"Eye Status: {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        # Vẽ EAR values
        cv2.putText(
            frame,
            f"Right EAR: {result['right_ear']:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

        cv2.putText(
            frame,
            f"Left EAR: {result['left_ear']:.2f}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )

        # Vẽ số lần nháy
        cv2.putText(
            frame,
            f"Blinks: {result['blink_count']}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Vẽ ngưỡng EAR
        cv2.putText(
            frame,
            f"Threshold: {self.ear_threshold:.2f}",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

        # Vẽ thông báo Liveness Passed
        if result['liveness_passed']:
            cv2.rectangle(frame, (0, 0), (frame_w, 50), (0, 255, 0), -1)
            cv2.putText(
                frame,
                "✓ LIVENESS PASSED - Blink Detected",
                (frame_w // 2 - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
        elif result['blink_detected']:
            cv2.rectangle(frame, (0, 0), (frame_w, 50), (0, 200, 200), -1)
            cv2.putText(
                frame,
                "Blink Detected!",
                (frame_w // 2 - 100, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Vẽ biểu đồ EAR nhỏ
        self._draw_ear_graph(frame, result)

        return frame

    def _draw_ear_graph(self, frame: np.ndarray, result: Dict) -> None:
        """Vẽ biểu đồ EAR lịch sử"""
        if len(self.ear_history) < 2:
            return

        graph_width = 150
        graph_height = 50
        graph_x = frame.shape[1] - graph_width - 10
        graph_y = frame.shape[0] - graph_height - 10

        # Background
        cv2.rectangle(frame, (graph_x, graph_y),
                     (graph_x + graph_width, graph_y + graph_height),
                     (50, 50, 50), -1)

        # Draw line chart
        history_list = list(self.ear_history)
        for i in range(len(history_list) - 1):
            x1 = graph_x + int(i * graph_width / len(history_list))
            y1 = graph_y + int(graph_height * (1 - history_list[i] / 0.4))

            x2 = graph_x + int((i + 1) * graph_width / len(history_list))
            y2 = graph_y + int(graph_height * (1 - history_list[i + 1] / 0.4))

            color = (0, 255, 0) if history_list[i] >= self.ear_threshold else (0, 0, 255)
            cv2.line(frame, (x1, y1), (x2, y2), color, 1)

        # Draw threshold line
        threshold_y = graph_y + int(graph_height * (1 - self.ear_threshold / 0.4))
        cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_width, threshold_y),
                (200, 200, 0), 1)

    def reset(self):
        """Reset trạng thái detector (không reset blink_count)"""
        self.eye_closed_frames = 0
        self.blink_detected = False
        self.ear_history.clear()

    def reset_liveness(self):
        """Reset toàn bộ trạng thái liveness"""
        self.eye_closed_frames = 0
        self.blink_count = 0
        self.blink_detected = False
        self.liveness_passed = False
        self.ear_history.clear()
