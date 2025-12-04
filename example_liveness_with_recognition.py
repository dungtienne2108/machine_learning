#!/usr/bin/env python3
"""
Example: Integration Liveness Detection with Face Recognition

Ví dụ tích hợp:
1. Xác minh Liveness (nháy mắt)
2. Nhận diện khuôn mặt (Face Recognition)

Quy trình:
- Người dùng nháy mắt để xác minh là "người thật"
- Sau đó, nhận diện khuôn mặt để xác định danh tính
"""

import cv2
import sys
from face_service import FaceRecognitionService


class LivenessAndRecognitionSystem:
    """Hệ thống kết hợp Liveness Detection + Face Recognition"""

    def __init__(self):
        """Khởi tạo hệ thống"""
        print("[INFO] Khởi tạo Face Recognition Service...")
        self.service = FaceRecognitionService()
        self.embeddings = self.service.load_embeddings()

        print(f"[INFO] Đã load {len(self.embeddings)} embeddings")

    def run_verification_flow(self):
        """Chạy quy trình xác minh hoàn chỉnh"""

        print("\n" + "=" * 60)
        print("VERIFICATION FLOW: Liveness + Face Recognition")
        print("=" * 60)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[LỖI] Không thể mở webcam!")
            return False

        print("[STEP 1] Yêu cầu nháy mắt để xác minh Liveness...")
        print("Nhấn 'SPACE' khi bạn muốn bắt đầu, 'q' để thoát")

        liveness_passed = False
        recognized_person = None
        confidence = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Display instruction
            cv2.putText(frame, "[STEP 1] Liveness Verification - Please blink",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Verify liveness
            liveness_result = self.service.verify_liveness(frame)
            frame = liveness_result['frame']

            # Show frame
            cv2.imshow("Verification System", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Thoát...")
                cap.release()
                cv2.destroyAllWindows()
                return False

            # Check if liveness passed
            if liveness_result['liveness_passed']:
                liveness_passed = True
                print(f"\n[✓ STEP 1 PASSED] Liveness verified!")
                print(f"[✓] Detected {liveness_result['blink_count']} blink(s)")

                # Move to face recognition
                break

        # STEP 2: Face Recognition
        if liveness_passed:
            print("\n[STEP 2] Yêu cầu nhận diện khuôn mặt...")
            print("Nhìn vào camera, nhấn SPACE để chụp, 'q' để thoát")

            capture_count = 0
            captured_faces = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(frame, "[STEP 2] Face Recognition - Look at camera",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Captured: {capture_count}/3",
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow("Verification System", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("[INFO] Thoát...")
                    break

                if key == ord(' '):  # SPACE
                    print(f"[CAPTURE {capture_count + 1}] Processing face...")

                    # Trích xuất embedding từ face
                    try:
                        from deepface import DeepFace

                        embedding_result = DeepFace.represent(
                            img_path=frame,
                            model_name=self.service.model_name,
                            detector_backend=self.service.detector_backend,
                            enforce_detection=True
                        )

                        if embedding_result:
                            face_embedding = embedding_result[0]['embedding']
                            captured_faces.append(face_embedding)
                            capture_count += 1
                            print(f"[✓] Face captured ({capture_count}/3)")

                            if capture_count >= 1:  # Chỉ cần 1 ảnh
                                print(f"[✓ STEP 2] Got {capture_count} face capture(s)")
                                break

                    except Exception as e:
                        print(f"[LỖI] Không trích xuất được face: {e}")

            # STEP 3: Match recognized person
            if captured_faces:
                print("\n[STEP 3] Matching face with database...")

                face_embedding = captured_faces[0]
                recognized_person, confidence = self.service.find_best_match(
                    face_embedding, self.embeddings
                )

                print(f"\n[✓ STEP 3 COMPLETED] Recognition result:")
                print(f"  Person: {recognized_person}")
                print(f"  Confidence: {confidence:.2%}")

        # Show final result
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)

        if liveness_passed:
            print(f"[✓] Liveness: PASSED ✓")
        else:
            print(f"[✗] Liveness: FAILED ✗")

        if recognized_person and recognized_person != "Unknown":
            print(f"[✓] Person: {recognized_person}")
            print(f"[✓] Confidence: {confidence:.2%}")
        else:
            print(f"[✗] Person: UNKNOWN")

        if liveness_passed and recognized_person != "Unknown" and confidence > 0.5:
            print("\n[✓✓✓] VERIFICATION SUCCESSFUL! ✓✓✓")
            print(f"Welcome, {recognized_person}!")
            return True
        else:
            print("\n[✗✗✗] VERIFICATION FAILED! ✗✗✗")
            return False

    def run_registration_flow(self):
        """Chạy quy trình đăng ký: Liveness -> Face Recognition"""

        print("\n" + "=" * 60)
        print("REGISTRATION FLOW: Liveness + Face Capture")
        print("=" * 60)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[LỖI] Không thể mở webcam!")
            return False

        print("[STEP 1] Kiểm tra Liveness - Vui lòng nháy mắt...")
        print("Nhấn 'SPACE' khi bạn muốn bắt đầu, 'q' để thoát")

        liveness_passed = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, "[STEP 1] Liveness Verification",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            liveness_result = self.service.verify_liveness(frame)
            frame = liveness_result['frame']

            cv2.imshow("Registration System", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False

            if liveness_result['liveness_passed']:
                liveness_passed = True
                print(f"[✓ STEP 1 PASSED] Liveness verified!")
                break

        if liveness_passed:
            print("\n[STEP 2] Chụp ảnh khuôn mặt để đăng ký...")
            print("Nhấn SPACE để chụp, 'q' để thoát")

            capture_count = 0
            captured_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(frame, "[STEP 2] Capture face for registration",
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow("Registration System", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                if key == ord(' '):
                    captured_frame = frame.copy()
                    capture_count += 1
                    print(f"[✓] Frame captured!")
                    break

            if captured_frame is not None:
                print(f"\n[✓ STEP 2 COMPLETED] Registration image captured")

                cv2.imshow("Captured Frame", captured_frame)
                print("Nhấn bất kỳ phím nào để hoàn tất...")
                cv2.waitKey(0)

                print("\n[✓✓✓] REGISTRATION SUCCESSFUL! ✓✓✓")
                print("Vui lòng thêm ảnh này vào hệ thống với tên người dùng")

        cap.release()
        cv2.destroyAllWindows()

        return True


def main():
    """Main function"""

    print("=" * 60)
    print("Liveness + Face Recognition Integration")
    print("=" * 60)
    print("\nChọn chế độ:")
    print("1. Verification (Xác minh người dùng)")
    print("2. Registration (Đăng ký người dùng mới)")
    print("q. Thoát")

    choice = input("\nNhập lựa chọn (1/2/q): ").strip().lower()

    system = LivenessAndRecognitionSystem()

    if choice == '1':
        system.run_verification_flow()
    elif choice == '2':
        system.run_registration_flow()
    elif choice == 'q':
        print("[INFO] Thoát...")
    else:
        print("[LỖI] Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    main()
