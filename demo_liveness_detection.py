#!/usr/bin/env python3
"""
Demo script: Active Liveness Detection using Eye Blink
Chạy real-time blink detection từ webcam

Chạy script:
    python demo_liveness_detection.py

Điều khiển:
    - Press 'q' or 'ESC' để thoát
    - Press 'r' để reset counter
    - Press 's' để save frame (nếu cần)
"""

import cv2
import sys
import time
from liveness_detection import LivenessDetector


def main():
    """Main demo function"""

    # Khởi tạo Liveness Detector
    print("[INFO] Khởi tạo Liveness Detector...")
    liveness_detector = LivenessDetector(
        ear_threshold=0.2,  # Ngưỡng Eye Aspect Ratio
        consecutive_frames=3  # Số frame mắt phải đóng để tính là nháy
    )

    # Mở webcam
    print("[INFO] Mở webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[LỖI] Không thể mở webcam!")
        return False

    # Lấy thông tin webcam
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Độ phân giải: {frame_width}x{frame_height}")
    print(f"[INFO] FPS: {fps}")
    print("[INFO] Nhấn 'q' hoặc 'ESC' để thoát, 'r' để reset")

    frame_count = 0
    start_time = time.time()
    last_detection_time = start_time

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[LỖI] Không đọc được frame từ webcam!")
                break

            frame_count += 1

            # Resize frame để tối ưu FPS (tuỳ chọn)
            # frame = cv2.resize(frame, (640, 480))

            # Phát hiện nháy mắt
            result = liveness_detector.detect_blink(frame)

            # Vẽ thông tin lên frame
            frame_with_info = liveness_detector.draw_eye_status(frame.copy(), result)

            # Hiển thị frame
            cv2.imshow("Active Liveness Detection - Eye Blink", frame_with_info)

            # Tính FPS
            current_time = time.time()
            elapsed = current_time - start_time
            fps_display = frame_count / elapsed if elapsed > 0 else 0

            # In thông tin mỗi 30 frame
            if frame_count % 30 == 0:
                print(f"[FRAME {frame_count}] "
                      f"Eye: {'OPEN' if result['eye_open'] else 'CLOSED'} | "
                      f"EAR: Right={result['right_ear']:.3f}, Left={result['left_ear']:.3f} | "
                      f"Blinks: {result['blink_count']} | "
                      f"FPS: {fps_display:.1f}")

                if result['blink_detected']:
                    print(f"[BLINK DETECTED] Total blinks: {result['blink_count']}")

                if result['liveness_passed']:
                    print(f"[✓ LIVENESS PASSED] - Blink detected!")
                    print(f"[✓] Người dùng xác minh thành công (Verified)")

            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 27 = ESC
                print("[INFO] Thoát chương trình...")
                break
            elif key == ord('r'):  # Reset
                print("[INFO] Reset counter...")
                liveness_detector.reset_liveness()
            elif key == ord('s'):  # Save frame
                filename = f"liveness_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame_with_info)
                print(f"[INFO] Đã lưu frame: {filename}")

    except KeyboardInterrupt:
        print("[INFO] Bị gián đoạn bởi người dùng")
    except Exception as e:
        print(f"[LỖI] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Đã đóng webcam")

    return True


def demo_with_image(image_path):
    """Demo với ảnh tĩnh (nếu không có webcam)"""
    print(f"[INFO] Load ảnh: {image_path}")

    import os
    if not os.path.exists(image_path):
        print(f"[LỖI] File không tồn tại: {image_path}")
        return False

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[LỖI] Không đọc được ảnh: {image_path}")
        return False

    liveness_detector = LivenessDetector()
    result = liveness_detector.detect_blink(frame)

    frame_with_info = liveness_detector.draw_eye_status(frame.copy(), result)

    print(f"[INFO] Eye Status: {'OPEN' if result['eye_open'] else 'CLOSED'}")
    print(f"[INFO] Right EAR: {result['right_ear']:.3f}")
    print(f"[INFO] Left EAR: {result['left_ear']:.3f}")

    cv2.imshow("Liveness Detection", frame_with_info)
    print("[INFO] Nhấn bất kỳ phím nào để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Active Liveness Detection - Eye Blink Detection")
    print("=" * 60)

    if len(sys.argv) > 1:
        # Nếu có tham số, sử dụng ảnh
        demo_with_image(sys.argv[1])
    else:
        # Nếu không, chạy webcam
        main()
