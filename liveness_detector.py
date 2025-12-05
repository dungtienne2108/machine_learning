import cv2
import numpy as np
from collections import deque

class LivenessDetector:
    """
    Phát hiện khuôn mặt thật vs ảnh/video giả mạo
    Sử dụng nhiều kỹ thuật:
    1. Texture Analysis (LBP - Local Binary Pattern)
    2. Motion Detection (phát hiện chuyển động tự nhiên)
    3. Eye Blink Detection (phát hiện chớp mắt)
    4. Frequency Analysis (phân tích tần số - phát hiện màn hình)
    """
    
    def __init__(self):
        # Load Haar Cascade cho mắt
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Lưu trữ lịch sử frames để phân tích
        self.frame_history = deque(maxlen=30)  # 30 frames (1 giây ở 30fps)
        self.motion_history = deque(maxlen=10)
        self.blink_history = deque(maxlen=60)  # 2 giây
        
        # Ngưỡng phát hiện
        self.TEXTURE_THRESHOLD = 0.3
        self.MOTION_THRESHOLD = 5.0
        self.BLINK_THRESHOLD = 2  # Số lần chớp mắt tối thiểu trong 2 giây
        
        # Trạng thái
        self.last_eye_state = None
        self.blink_count = 0
        
    def reset(self):
        """Reset trạng thái detector"""
        self.frame_history.clear()
        self.motion_history.clear()
        self.blink_history.clear()
        self.last_eye_state = None
        self.blink_count = 0
    
    def analyze_texture_lbp(self, face_gray):
        """
        Phân tích texture bằng LBP (Local Binary Pattern)
        Ảnh in/màn hình có texture khác với khuôn mặt thật
        Returns: lbp_score (0-1), higher = more likely real
        """
        try:
            # Resize để tăng tốc
            face_resized = cv2.resize(face_gray, (64, 64))
            
            # Tính LBP
            lbp = np.zeros_like(face_resized)
            for i in range(1, face_resized.shape[0] - 1):
                for j in range(1, face_resized.shape[1] - 1):
                    center = face_resized[i, j]
                    code = 0
                    code |= (face_resized[i-1, j-1] > center) << 7
                    code |= (face_resized[i-1, j] > center) << 6
                    code |= (face_resized[i-1, j+1] > center) << 5
                    code |= (face_resized[i, j+1] > center) << 4
                    code |= (face_resized[i+1, j+1] > center) << 3
                    code |= (face_resized[i+1, j] > center) << 2
                    code |= (face_resized[i+1, j-1] > center) << 1
                    code |= (face_resized[i, j-1] > center) << 0
                    lbp[i, j] = code
            
            # Tính histogram của LBP
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            
            # Tính entropy - khuôn mặt thật có entropy cao hơn
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # Normalize entropy (0-1)
            # Entropy tối đa cho 256 bins là log2(256) = 8
            normalized_entropy = entropy / 8.0
            
            return normalized_entropy
            
        except Exception as e:
            print(f"[LỖI] analyze_texture_lbp: {e}")
            return 0.5
    
    def analyze_motion(self, current_frame_gray, face_box):
        """
        Phân tích chuyển động tự nhiên
        Ảnh in không có chuyển động, video loop có pattern lặp lại
        Returns: motion_score (0-1), higher = more natural motion
        """
        try:
            x, y, w, h = face_box
            
            # Lấy vùng khuôn mặt
            face_region = current_frame_gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_region, (64, 64))
            
            # Thêm vào lịch sử
            self.frame_history.append(face_resized)
            
            if len(self.frame_history) < 2:
                return 0.5
            
            # Tính optical flow giữa frame hiện tại và frame trước
            prev_frame = self.frame_history[-2]
            curr_frame = self.frame_history[-1]
            
            # Tính sự khác biệt
            diff = cv2.absdiff(curr_frame, prev_frame)
            motion_score = np.mean(diff)
            
            # Lưu vào lịch sử motion
            self.motion_history.append(motion_score)
            
            if len(self.motion_history) < 5:
                return 0.5
            
            # Phân tích pattern của motion
            motion_array = np.array(self.motion_history)
            motion_std = np.std(motion_array)  # Độ biến thiên
            motion_mean = np.mean(motion_array)  # Giá trị trung bình
            
            # Khuôn mặt thật: có motion nhẹ và không đều
            # Ảnh in: motion = 0
            # Video: motion cao và đều
            
            if motion_mean < 1.0:  # Quá ít chuyển động
                return 0.2
            elif motion_mean > 15.0:  # Quá nhiều chuyển động
                return 0.3
            else:
                # Có chuyển động vừa phải và biến thiên tự nhiên
                score = min(1.0, motion_std / 5.0)
                return score
                
        except Exception as e:
            print(f"[LỖI] analyze_motion: {e}")
            return 0.5
    
    def detect_blink(self, face_gray, face_box):
        """
        Phát hiện chớp mắt - dấu hiệu quan trọng của khuôn mặt thật
        Returns: blink_score (0-1), higher = detected blinks
        """
        try:
            x, y, w, h = face_box
            
            # Vùng mắt (1/3 trên của khuôn mặt)
            eye_region = face_gray[y:y+h//2, x:x+w]
            
            # Phát hiện mắt
            eyes = self.eye_cascade.detectMultiScale(
                eye_region, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # Đếm số mắt phát hiện được
            current_eye_count = len(eyes)
            
            # Lưu vào lịch sử
            self.blink_history.append(current_eye_count)
            
            # Phát hiện chớp mắt: số mắt giảm từ 2 xuống 0 hoặc 1
            if self.last_eye_state is not None:
                if self.last_eye_state >= 2 and current_eye_count < 2:
                    self.blink_count += 1
                    print(f"[INFO] Phát hiện chớp mắt! Tổng: {self.blink_count}")
            
            self.last_eye_state = current_eye_count
            
            # Tính điểm dựa trên số lần chớp mắt
            if len(self.blink_history) >= 30:  # Đủ dữ liệu (1 giây)
                recent_blinks = self.blink_count
                if recent_blinks >= self.BLINK_THRESHOLD:
                    return 1.0
                else:
                    return min(1.0, recent_blinks / self.BLINK_THRESHOLD)
            
            return 0.5  # Chưa đủ dữ liệu
            
        except Exception as e:
            print(f"[LỖI] detect_blink: {e}")
            return 0.5
    
    def analyze_frequency(self, face_gray):
        """
        Phân tích tần số - phát hiện màn hình
        Màn hình có tần số refresh đặc trưng (50Hz, 60Hz)
        Returns: frequency_score (0-1), higher = less likely screen
        """
        try:
            # Resize để tăng tốc
            face_resized = cv2.resize(face_gray, (64, 64))
            
            # Tính FFT 2D
            f_transform = np.fft.fft2(face_resized)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Tính energy trong các dải tần số
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            # Low frequency (0-10% radius)
            low_freq_radius = int(min(h, w) * 0.1)
            low_mask = np.zeros((h, w))
            cv2.circle(low_mask, (center_w, center_h), low_freq_radius, 1, -1)
            low_energy = np.sum(magnitude * low_mask)
            
            # High frequency (40-50% radius)
            high_freq_radius = int(min(h, w) * 0.5)
            high_mask = np.zeros((h, w))
            cv2.circle(high_mask, (center_w, center_h), high_freq_radius, 1, -1)
            high_mask = high_mask - low_mask
            high_energy = np.sum(magnitude * high_mask)
            
            # Khuôn mặt thật: nhiều low frequency hơn
            # Màn hình: nhiều high frequency (do pixel grid)
            ratio = low_energy / (high_energy + 1e-7)
            
            # Normalize
            score = min(1.0, ratio / 10.0)
            return score
            
        except Exception as e:
            print(f"[LỖI] analyze_frequency: {e}")
            return 0.5
    
    def check_liveness(self, frame, face_box):
        """
        Kiểm tra tổng hợp - khuôn mặt có thật hay không
        Returns: (is_real, confidence, details)
        """
        try:
            x, y, w, h = face_box
            
            # Lấy vùng khuôn mặt
            face_region = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # 1. Phân tích texture
            texture_score = self.analyze_texture_lbp(face_gray)
            
            # 2. Phân tích motion
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = self.analyze_motion(frame_gray, face_box)
            
            # 3. Phát hiện chớp mắt
            blink_score = self.detect_blink(face_gray, face_box)
            
            # 4. Phân tích tần số
            frequency_score = self.analyze_frequency(face_gray)
            
            # Tính điểm tổng hợp (weighted average)
            weights = {
                'texture': 0.25,
                'motion': 0.25,
                'blink': 0.35,    # Quan trọng nhất
                'frequency': 0.15
            }
            
            total_score = (
                texture_score * weights['texture'] +
                motion_score * weights['motion'] +
                blink_score * weights['blink'] +
                frequency_score * weights['frequency']
            )
            
            # Chi tiết các điểm
            details = {
                'texture': texture_score,
                'motion': motion_score,
                'blink': blink_score,
                'frequency': frequency_score,
                'total': total_score,
                'blink_count': self.blink_count
            }
            
            # Ngưỡng quyết định: cần điểm > 0.6 và có chớp mắt
            is_real = total_score > 0.6 and self.blink_count >= 1
            
            return is_real, total_score, details
            
        except Exception as e:
            print(f"[LỖI] check_liveness: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, {}
    
    def draw_liveness_info(self, frame, face_box, is_real, confidence, details):
        """Vẽ thông tin liveness lên frame"""
        try:
            x, y, w, h = face_box
            
            # Màu dựa trên kết quả
            if is_real:
                color = (0, 255, 0)  # Xanh lá - Real
                status = "REAL"
            else:
                color = (0, 0, 255)  # Đỏ - Fake
                status = "FAKE/PHOTO"
            
            # Vẽ box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Vẽ status
            cv2.putText(frame, f"{status} ({confidence*100:.1f}%)",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Vẽ chi tiết (góc trên bên phải)
            info_x = frame.shape[1] - 250
            info_y = 30
            line_height = 25
            
            cv2.rectangle(frame, (info_x-10, info_y-20), 
                         (frame.shape[1]-10, info_y + line_height*6), 
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (info_x-10, info_y-20), 
                         (frame.shape[1]-10, info_y + line_height*6), 
                         (255, 255, 255), 1)
            
            cv2.putText(frame, "Liveness Detection:", (info_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += line_height
            
            # Các metrics
            metrics = [
                f"Texture: {details.get('texture', 0)*100:.0f}%",
                f"Motion: {details.get('motion', 0)*100:.0f}%",
                f"Blinks: {details.get('blink_count', 0)}",
                f"Frequency: {details.get('frequency', 0)*100:.0f}%",
                f"Total: {details.get('total', 0)*100:.0f}%"
            ]
            
            for metric in metrics:
                cv2.putText(frame, metric, (info_x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                info_y += line_height
            
        except Exception as e:
            print(f"[LỖI] draw_liveness_info: {e}")


class SimpleLivenessDetector:
    """
    Phiên bản đơn giản hơn - chỉ dùng motion + blink
    Sử dụng khi cần hiệu năng cao
    """
    
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.prev_frame = None
        self.blink_count = 0
        self.last_eye_state = None
        self.frame_count = 0
        
    def reset(self):
        self.prev_frame = None
        self.blink_count = 0
        self.last_eye_state = None
        self.frame_count = 0
    
    def check_liveness(self, frame, face_box):
        """
        Kiểm tra đơn giản: motion + blink
        Returns: (is_real, confidence, details)
        """
        try:
            x, y, w, h = face_box
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_gray = gray[y:y+h, x:x+w]
            
            self.frame_count += 1
            
            # 1. Kiểm tra motion
            motion_score = 0.5
            if self.prev_frame is not None:
                diff = cv2.absdiff(face_gray, self.prev_frame)
                motion = np.mean(diff)
                
                if motion < 1.0:  # Quá ít motion - có thể là ảnh
                    motion_score = 0.2
                elif motion > 15.0:  # Quá nhiều motion
                    motion_score = 0.3
                else:
                    motion_score = 0.8
            
            self.prev_frame = face_gray.copy()
            
            # 2. Phát hiện chớp mắt
            eye_region = gray[y:y+h//2, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                eye_region, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            
            current_eye_count = len(eyes)
            if self.last_eye_state is not None:
                if self.last_eye_state >= 2 and current_eye_count < 2:
                    self.blink_count += 1
            self.last_eye_state = current_eye_count
            
            blink_score = min(1.0, self.blink_count / 2.0)
            
            # Tổng hợp
            total_score = motion_score * 0.4 + blink_score * 0.6
            
            # Cần ít nhất 1 cú chớp mắt và motion hợp lý
            is_real = self.blink_count >= 1 and motion_score > 0.3
            
            details = {
                'motion': motion_score,
                'blink': blink_score,
                'blink_count': self.blink_count,
                'total': total_score
            }
            
            return is_real, total_score, details
            
        except Exception as e:
            print(f"[LỖI] SimpleLivenessDetector: {e}")
            return False, 0.0, {}