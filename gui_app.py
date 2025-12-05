import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import threading
from datetime import datetime
from database_helper import DatabaseHelper
from face_service import FaceRecognitionService
from liveness_detector import LivenessDetector  # Import liveness detector
from dialogs.login_dialog import LoginDialog
from dialogs.add_person_dialog import AddPersonDialog
from dialogs.manage_persons_dialog import ManagePersonsDialog

class AttendanceRecognitionUI:
    """Giao diện Attendance System - Có Liveness Detection"""
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System with Anti-Spoofing")
        self.root.geometry("1280x720")
        self.root.configure(bg='#7c5ceb')
        
        self.db = DatabaseHelper()
        self.face_service = FaceRecognitionService()
        self.liveness_detector = LivenessDetector()  # Khởi tạo liveness detector
        
        self.cap = None
        self.is_camera_running = False
        self.current_person = None
        self.last_recognition_time = None
        self.recognition_cooldown = 30
        self.need_reload_embeddings = False
        self.embeddings = []
        self.is_camera_paused = False
        self.recognized_person = None
        
        # Liveness detection state
        self.liveness_enabled = True  # Bật/tắt liveness detection
        self.liveness_check_interval = 5  # Kiểm tra mỗi 5 frames
        self.liveness_status = None
        self.liveness_confidence = 0.0
        self.liveness_details = {}
        
        self.setup_ui()
        self.start_recognition()
    
    def setup_ui(self):
        # Main container
        main_container = tk.Frame(self.root, bg='#7c5ceb')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # LEFT PANEL - Camera
        left_panel = tk.Frame(main_container, bg='#7c5ceb')
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 20))
        
        # Header với nút admin và liveness toggle
        header_frame = tk.Frame(left_panel, bg='#5940c9', height=100)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title_container = tk.Frame(header_frame, bg='#5940c9')
        title_container.pack(side='left', padx=20, pady=20)
        
        tk.Label(title_container, text="🎓", font=("Arial", 40), 
                bg='#5940c9', fg='white').pack(side='left', padx=(0, 15))
        
        tk.Label(title_container, text="ATTENDANCE SYSTEM", 
                font=("Arial", 28, "bold"), bg='#5940c9', fg='white').pack(side='left')
        
        # Buttons container (bên phải)
        buttons_container = tk.Frame(header_frame, bg='#5940c9')
        buttons_container.pack(side='right', padx=20)
        
        # Nút Liveness Toggle
        self.btn_liveness = tk.Button(buttons_container, 
                                     text="🛡️ LIVENESS: ON", 
                                     font=("Arial", 10, "bold"), 
                                     bg='#27ae60', fg='white',
                                     relief='flat', cursor='hand2', 
                                     command=self.toggle_liveness,
                                     activebackground='#229954', 
                                     borderwidth=0,
                                     highlightthickness=0)
        self.btn_liveness.pack(side='left', padx=(0, 10), ipadx=12, ipady=6)
        
        # Nút Admin
        btn_admin = tk.Button(buttons_container, text="⚙️ ADMIN", 
                             font=("Arial", 11, "bold"), bg='#e74c3c', fg='white',
                             relief='flat', cursor='hand2', command=self.show_admin_login,
                             activebackground='#c0392b', borderwidth=0,
                             highlightthickness=0)
        btn_admin.pack(side='left', ipadx=15, ipady=8)
        
        # Camera Frame
        camera_container = tk.Frame(left_panel, bg='#5940c9', padx=15, pady=15)
        camera_container.pack(fill='both', expand=True)
        
        self.canvas_camera = tk.Canvas(camera_container, bg='black', 
                                      highlightthickness=0, bd=0)
        self.canvas_camera.pack(fill='both', expand=True)
        
        # RIGHT PANEL - Student Info
        right_panel = tk.Frame(main_container, bg='white', width=450, 
                              highlightbackground='#d0d0d0', highlightthickness=2)
        right_panel.pack(side='right', fill='both')
        right_panel.pack_propagate(False)
        
        # Info Card
        self.info_card = tk.Frame(right_panel, bg='white')
        self.info_card.pack(fill='both', expand=True, padx=30, pady=30)
        
        # Icon trạng thái
        self.status_icon_label = tk.Label(self.info_card, text="👤", 
                                         font=("Arial", 80), bg='white', fg='#e0e0e0')
        self.status_icon_label.pack(pady=(30, 20))
        
        # Name
        self.name_label = tk.Label(self.info_card, text="Chờ nhận diện...", 
                                  font=("Arial", 26, "bold"), bg='white', fg='#2c3e50',
                                  wraplength=400)
        self.name_label.pack(pady=(0, 25))
        
        # Info badges container
        badges_container = tk.Frame(self.info_card, bg='white')
        badges_container.pack(pady=10, fill='x')
        
        # Mã NV
        self.id_badge = tk.Frame(badges_container, bg='#c4b5fd', 
                                height=50, padx=25, pady=12)
        self.id_badge.pack(fill='x', pady=8)
        
        id_content = tk.Frame(self.id_badge, bg='#c4b5fd')
        id_content.pack()
        
        tk.Label(id_content, text="🆔", font=("Arial", 20), 
                bg='#c4b5fd').pack(side='left', padx=(0, 10))
        self.id_label = tk.Label(id_content, text="Mã: ------", 
                               font=("Arial", 14, "bold"), bg='#c4b5fd', 
                               fg='#5940c9', wraplength=350)
        self.id_label.pack(side='left')
        
        # Phòng ban
        self.dept_badge = tk.Frame(badges_container, bg='#c4b5fd', 
                                  height=50, padx=25, pady=12)
        self.dept_badge.pack(fill='x', pady=8)
        
        dept_content = tk.Frame(self.dept_badge, bg='#c4b5fd')
        dept_content.pack()
        
        tk.Label(dept_content, text="🏢", font=("Arial", 20), 
                bg='#c4b5fd').pack(side='left', padx=(0, 10))
        self.dept_label = tk.Label(dept_content, text="Phòng: ------", 
                                   font=("Arial", 14, "bold"), bg='#c4b5fd', 
                                   fg='#5940c9', wraplength=350)
        self.dept_label.pack(side='left')
        
        # Confidence
        self.conf_badge = tk.Frame(badges_container, bg='#e8f5e9', 
                                  height=50, padx=25, pady=12)
        self.conf_badge.pack(fill='x', pady=8)
        
        conf_content = tk.Frame(self.conf_badge, bg='#e8f5e9')
        conf_content.pack()
        
        tk.Label(conf_content, text="📊", font=("Arial", 20), 
                bg='#e8f5e9').pack(side='left', padx=(0, 10))
        self.conf_label = tk.Label(conf_content, text="Độ chính xác: ---%", 
                                   font=("Arial", 14, "bold"), bg='#e8f5e9', 
                                   fg='#27ae60', wraplength=350)
        self.conf_label.pack(side='left')
        
        # Liveness badge
        self.liveness_badge = tk.Frame(badges_container, bg='#fff3cd', 
                                      height=50, padx=25, pady=12)
        self.liveness_badge.pack(fill='x', pady=8)
        
        liveness_content = tk.Frame(self.liveness_badge, bg='#fff3cd')
        liveness_content.pack()
        
        tk.Label(liveness_content, text="🛡️", font=("Arial", 20), 
                bg='#fff3cd').pack(side='left', padx=(0, 10))
        self.liveness_label = tk.Label(liveness_content, text="Liveness: ---", 
                                       font=("Arial", 14, "bold"), bg='#fff3cd', 
                                       fg='#856404', wraplength=350)
        self.liveness_label.pack(side='left')
        
        # Status message
        self.status_label = tk.Label(self.info_card, text="", 
                                    font=("Arial", 13, "bold"), bg='white', 
                                    fg='#999', wraplength=400)
        self.status_label.pack(pady=(30, 0))
    
    def toggle_liveness(self):
        """Bật/tắt liveness detection"""
        self.liveness_enabled = not self.liveness_enabled
        
        if self.liveness_enabled:
            self.btn_liveness.config(text="🛡️ LIVENESS: ON", bg='#27ae60')
            messagebox.showinfo("Liveness Detection", 
                              "Đã BẬT chống giả mạo!\n\n"
                              "Hệ thống sẽ kiểm tra:\n"
                              "• Chuyển động tự nhiên\n"
                              "• Chớp mắt\n"
                              "• Texture khuôn mặt\n"
                              "• Phân tích tần số",
                              parent=self.root)
        else:
            self.btn_liveness.config(text="🛡️ LIVENESS: OFF", bg='#e74c3c')
            messagebox.showwarning("Liveness Detection", 
                                 "Đã TẮT chống giả mạo!\n\n"
                                 "⚠️ Cảnh báo: Hệ thống có thể chấp nhận ảnh/video giả",
                                 parent=self.root)
        
        # Reset liveness detector
        self.liveness_detector.reset()
        self.liveness_status = None
        self.liveness_confidence = 0.0
    
    def show_admin_login(self):
        """Hiển thị dialog đăng nhập admin"""
        def on_login_success(user_info):
            self.show_admin_menu(user_info)
        
        LoginDialog(self.root, on_login_success)
    
    def show_admin_menu(self, user_info):
        """Hiển thị menu admin"""
        menu = tk.Toplevel(self.root)
        menu.title(f"Admin Panel - {user_info['full_name']}")
        menu.geometry("450x600")
        menu.configure(bg='#ecf0f1')
        menu.transient(self.root)
        menu.resizable(False, False)
        
        # Center window
        menu.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 225
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 300
        menu.geometry(f"450x600+{x}+{y}")
        
        # Header
        header = tk.Frame(menu, bg='#7c5ceb')
        header.pack(fill='x')
        
        icon_frame = tk.Frame(header, bg='#7c5ceb')
        icon_frame.pack(pady=(30, 10))
        
        tk.Label(icon_frame, text="⚙️", font=("Arial", 50), 
                bg='#7c5ceb', fg='white').pack()
        
        tk.Label(header, text="ADMIN PANEL", font=("Arial", 20, "bold"), 
                bg='#7c5ceb', fg='white').pack(pady=(5, 10))
        
        # Menu buttons
        menu_frame = tk.Frame(menu, bg='#ecf0f1')
        menu_frame.pack(fill='both', expand=True, padx=40, pady=30)
        
        buttons = [
            ("➕ Thêm người mới", '#27ae60', self.add_person_action),
            ("📋 Quản lý đối tượng", '#3498db', self.manage_persons_action),
            ("📊 Xem thống kê", '#f39c12', lambda: messagebox.showinfo(
                "Thông báo", "Tính năng đang phát triển", parent=menu)),
            ("🔄 Tạo Embeddings", '#9b59b6', self.generate_embeddings_action),
            ("🚪 Đóng", '#95a5a6', menu.destroy),
        ]
        
        for text, color, command in buttons:
            btn_container = tk.Frame(menu_frame, bg='#ecf0f1')
            btn_container.pack(fill='x', pady=8)
            
            btn = tk.Button(btn_container, text=text, font=("Arial", 12, "bold"),
                           bg=color, fg='white', relief='flat', cursor='hand2',
                           command=command, activebackground=color,
                           borderwidth=0, highlightthickness=0)
            btn.pack(fill='x', ipady=15)
            
            def make_hover(btn, normal_color):
                def on_enter(e):
                    rgb = btn.winfo_rgb(normal_color)
                    dark = '#%02x%02x%02x' % tuple(int(c * 0.8 / 256) for c in rgb)
                    btn.config(bg=dark)
                
                def on_leave(e):
                    btn.config(bg=normal_color)
                
                btn.bind('<Enter>', on_enter)
                btn.bind('<Leave>', on_leave)
            
            make_hover(btn, color)
        
        # Footer
        footer = tk.Frame(menu, bg='#ecf0f1', height=40)
        footer.pack(fill='x', side='bottom')
        footer.pack_propagate(False)
        
        tk.Label(footer, text="Face Attendance System v1.0", 
                font=("Arial", 9), bg='#ecf0f1', fg='#7f8c8d').pack(pady=10)
    
    def add_person_action(self):
        """Mở dialog thêm người"""
        self.pause_camera()
        
        def on_success():
            self.need_reload_embeddings = True
        
        dialog = AddPersonDialog(self.root, self.face_service, self.db, on_success)
        
        original_on_close = dialog.on_close
        def enhanced_on_close():
            print("[INFO] Dialog đang đóng...")
            original_on_close()
            self.root.after(500, self.resume_camera)
        
        dialog.on_close = enhanced_on_close
        dialog.dialog.wait_window()
        
        print("[INFO] Dialog đã đóng, chuẩn bị khôi phục camera...")
        self.root.after(500, self.resume_camera)
    
    def manage_persons_action(self):
        """Mở dialog quản lý đối tượng"""
        ManagePersonsDialog(self.root, self.face_service, self.db, self)
    
    def generate_embeddings_action(self):
        """Tạo embeddings"""
        def run():
            success = self.face_service.generate_embeddings()
            if success:
                messagebox.showinfo("Thành công", "Đã tạo embeddings thành công!")
                self.need_reload_embeddings = True
        
        threading.Thread(target=run, daemon=True).start()
        messagebox.showinfo("Đang xử lý", "Đang tạo embeddings...")
    
    def start_recognition(self):
        """Bắt đầu nhận diện"""
        self.is_camera_running = True
        threading.Thread(target=self.recognition_loop, daemon=True).start()
    
    def pause_camera(self):
        """Tạm dừng camera"""
        print("[INFO] Yêu cầu tạm dừng camera...")
        self.is_camera_paused = True
        
        import time
        time.sleep(0.2)
        
        if self.cap:
            try:
                self.cap.release()
                print("[INFO] Đã release camera màn hình chính")
            except Exception as e:
                print(f"[LỖI] Lỗi khi release camera: {e}")
            finally:
                self.cap = None
        
        time.sleep(0.3)
        print("[INFO] Camera màn hình chính đã tạm dừng hoàn toàn")
    
    def resume_camera(self):
        """Tiếp tục camera"""
        print("[INFO] Yêu cầu khôi phục camera...")
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    print("[INFO] Đã mở lại camera thành công")
                    break
                else:
                    self.cap.release()
                    self.cap = None
                    retry_count += 1
                    print(f"[CẢNH BÁO] Thử lại mở camera lần {retry_count}/{max_retries}")
                    import time
                    time.sleep(0.5)
            except Exception as e:
                print(f"[LỖI] Không thể mở camera: {e}")
                retry_count += 1
                import time
                time.sleep(0.5)
        
        if self.cap is None or not self.cap.isOpened():
            print("[LỖI] Không thể khôi phục camera sau nhiều lần thử")
            messagebox.showerror("Lỗi", 
                "Không thể khôi phục camera!\n"
                "Vui lòng khởi động lại ứng dụng.")
            return
        
        self.is_camera_paused = False
        self.liveness_detector.reset()  # Reset liveness detector
        print("[INFO] Camera màn hình chính đã hoạt động trở lại")
    
    def recognition_loop(self):
        """Vòng lặp nhận diện với liveness detection - FIX HOÀN TOÀN"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        
        self.embeddings = self.face_service.load_embeddings()
        frame_count = 0
        last_results = []
        
        while self.is_camera_running:
            if self.is_camera_paused:
                import time
                time.sleep(0.1)
                frame_count = 0
                last_results = []
                continue
            
            if self.need_reload_embeddings:
                print("[INFO] Phát hiện cần reload embeddings...")
                self.need_reload_embeddings = False
                
                def reload_embeddings_thread():
                    try:
                        print("[INFO] Bắt đầu generate embeddings...")
                        self.face_service.generate_embeddings()
                        new_embeddings = self.face_service.load_embeddings()
                        self.embeddings = new_embeddings
                        print(f"[INFO] Đã reload embeddings: {len(self.embeddings)} người")
                    except Exception as e:
                        print(f"[LỖI] Không thể reload embeddings: {e}")
                        import traceback
                        traceback.print_exc()
                
                threading.Thread(target=reload_embeddings_thread, daemon=True).start()
            
            if self.cap is None or not self.cap.isOpened():
                print("[CẢNH BÁO] Camera không khả dụng, chờ...")
                import time
                time.sleep(0.5)
                continue
            
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("[CẢNH BÁO] Không đọc được frame")
                    import time
                    time.sleep(0.1)
                    continue
            except Exception as e:
                print(f"[LỖI] Lỗi khi đọc frame: {e}")
                import time
                time.sleep(0.1)
                continue
            
            try:
                frame_count += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_service.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0 and self.recognized_person is not None:
                    current_time = datetime.now()
                    if self.last_recognition_time and \
                    (current_time - self.last_recognition_time).total_seconds() > self.recognition_cooldown:
                        self.reset_to_waiting_state()
                
                # Nhận diện khuôn mặt mỗi 10 frames
                if frame_count % 10 == 0 and len(self.embeddings) > 0:
                    last_results = []
                    try:
                        from deepface import DeepFace
                        results = DeepFace.extract_faces(frame, 
                                                        detector_backend=self.face_service.detector_backend,
                                                        enforce_detection=False)
                        
                        for r in results:
                            fa = r["facial_area"]
                            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                            
                            face_img = frame[y:y+h, x:x+w]
                            if face_img.size > 0:
                                # ============================================
                                # LIVENESS DETECTION - FIX HOÀN TOÀN
                                # ============================================
                                allow_recognition = True  # MẶC ĐỊNH: CHO PHÉP nhận diện
                                is_real_face = True
                                liveness_conf = 1.0
                                
                                # Chỉ kiểm tra liveness khi được bật
                                if self.liveness_enabled:
                                    # Kiểm tra liveness mỗi N frames để tối ưu
                                    if frame_count % self.liveness_check_interval == 0:
                                        try:
                                            is_real_check, conf_check, details_check = \
                                                self.liveness_detector.check_liveness(frame, (x, y, w, h))
                                            
                                            # Lưu kết quả
                                            self.liveness_status = is_real_check
                                            self.liveness_confidence = conf_check
                                            self.liveness_details = details_check
                                            
                                            print(f"[DEBUG] Liveness check: is_real={is_real_check}, conf={conf_check:.2f}")
                                        except Exception as e:
                                            print(f"[LỖI] Liveness check failed: {e}")
                                            # Nếu lỗi, cho phép nhận diện
                                            self.liveness_status = None
                                    
                                    # Xác định có cho phép nhận diện không
                                    # CHỈ CHẶN KHI ĐÃ XÁC NHẬN LÀ FAKE
                                    if self.liveness_status == False:
                                        # Đã xác nhận FAKE - CHẶN nhận diện
                                        allow_recognition = False
                                        is_real_face = False
                                        liveness_conf = self.liveness_confidence
                                        
                                        # Cập nhật UI - FAKE
                                        self.liveness_label.config(
                                            text=f"✗ FAKE ({self.liveness_confidence*100:.0f}%)",
                                            fg='#e74c3c'
                                        )
                                        self.liveness_badge.config(bg='#f8d7da')
                                        print("[INFO] CHẶN nhận diện - Phát hiện FAKE")
                                        
                                    elif self.liveness_status == True:
                                        # Đã xác nhận REAL - CHO PHÉP nhận diện
                                        allow_recognition = True
                                        is_real_face = True
                                        liveness_conf = self.liveness_confidence
                                        
                                        # Cập nhật UI - REAL
                                        self.liveness_label.config(
                                            text=f"✓ REAL ({self.liveness_confidence*100:.0f}%)",
                                            fg='#27ae60'
                                        )
                                        self.liveness_badge.config(bg='#d4edda')
                                        
                                    else:
                                        # Chưa có kết quả (None) - VẪN CHO PHÉP nhận diện
                                        allow_recognition = True
                                        is_real_face = True
                                        
                                        # Cập nhật UI - Đang kiểm tra
                                        self.liveness_label.config(
                                            text="⏳ Đang kiểm tra...",
                                            fg='#f39c12'
                                        )
                                        self.liveness_badge.config(bg='#fff3cd')
                                
                                # ============================================
                                # NHẬN DIỆN - Chỉ khi được phép
                                # ============================================
                                if allow_recognition:
                                    print(f"[DEBUG] Bắt đầu nhận diện... (liveness_enabled={self.liveness_enabled}, status={self.liveness_status})")
                                    
                                    # Tạo embedding và nhận diện
                                    emb = DeepFace.represent(
                                        face_img,
                                        model_name=self.face_service.model_name,
                                        detector_backend="skip",
                                        enforce_detection=False
                                    )[0]["embedding"]
                                    
                                    emb_array = np.array(emb, dtype=np.float32)
                                    identity, confidence = self.face_service.find_best_match(
                                        emb_array, self.embeddings)
                                    
                                    print(f"[DEBUG] Nhận diện: {identity} - {confidence*100:.1f}%")
                                    
                                    # Lưu kết quả
                                    last_results.append((x, y, w, h, identity, confidence, is_real_face))
                                    
                                    # Xử lý nhận diện thành công
                                    if identity != "Unknown" and confidence > 0.6:
                                        current_time = datetime.now()
                                        
                                        if self.recognized_person is not None:
                                            if identity == self.recognized_person['name']:
                                                self.update_recognized_status()
                                            else:
                                                if (self.last_recognition_time is None or 
                                                    (current_time - self.last_recognition_time).total_seconds() > self.recognition_cooldown):
                                                    self.update_student_info(identity, confidence, face_img)
                                                    self.last_recognition_time = current_time
                                        else:
                                            self.update_student_info(identity, confidence, face_img)
                                            self.last_recognition_time = current_time
                                else:
                                    # Liveness failed - Vẽ warning
                                    print("[INFO] Vẽ warning FAKE/PHOTO")
                                    last_results.append((x, y, w, h, "FAKE/PHOTO", 0.0, False))
                    
                    except Exception as e:
                        print(f"[LỖI] DeepFace recognition: {e}")
                        import traceback
                        traceback.print_exc()
                
                # ============================================
                # VẼ KẾT QUẢ LÊN FRAME
                # ============================================
                pil_image = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(pil_image)
                
                for result in last_results:
                    if len(result) == 7:
                        x, y, w, h, identity, confidence, is_real = result
                    else:
                        x, y, w, h, identity, confidence = result
                        is_real = True
                    
                    # Màu dựa trên liveness và nhận diện
                    if not is_real:
                        color = (255, 0, 0)  # Đỏ - Fake
                    elif identity != "Unknown":
                        color = (0, 255, 0)  # Xanh lá - Real + Recognized
                    else:
                        color = (255, 255, 0)  # Vàng - Real but Unknown
                    
                    corner_length = 30
                    thickness = 4
                    
                    # Vẽ các góc của bounding box
                    draw.line([(x, y), (x + corner_length, y)], fill=color, width=thickness)
                    draw.line([(x, y), (x, y + corner_length)], fill=color, width=thickness)
                    draw.line([(x+w, y), (x+w - corner_length, y)], fill=color, width=thickness)
                    draw.line([(x+w, y), (x+w, y + corner_length)], fill=color, width=thickness)
                    draw.line([(x, y+h), (x + corner_length, y+h)], fill=color, width=thickness)
                    draw.line([(x, y+h), (x, y+h - corner_length)], fill=color, width=thickness)
                    draw.line([(x+w, y+h), (x+w - corner_length, y+h)], fill=color, width=thickness)
                    draw.line([(x+w, y+h), (x+w, y+h - corner_length)], fill=color, width=thickness)
                
                # Vẽ các khuôn mặt chưa nhận diện (chỉ detect)
                if len(last_results) == 0:
                    for (x, y, w, h) in faces:
                        color = (255, 255, 0)
                        corner_length = 30
                        thickness = 4
                        
                        draw.line([(x, y), (x + corner_length, y)], fill=color, width=thickness)
                        draw.line([(x, y), (x, y + corner_length)], fill=color, width=thickness)
                        draw.line([(x+w, y), (x+w - corner_length, y)], fill=color, width=thickness)
                        draw.line([(x+w, y), (x+w, y + corner_length)], fill=color, width=thickness)
                        draw.line([(x, y+h), (x + corner_length, y+h)], fill=color, width=thickness)
                        draw.line([(x, y+h), (x, y+h - corner_length)], fill=color, width=thickness)
                        draw.line([(x+w, y+h), (x+w - corner_length, y+h)], fill=color, width=thickness)
                        draw.line([(x+w, y+h), (x+w, y+h - corner_length)], fill=color, width=thickness)
                
                # Hiển thị lên canvas
                try:
                    if not self.canvas_camera.winfo_exists():
                        break
                    
                    canvas_width = self.canvas_camera.winfo_width()
                    canvas_height = self.canvas_camera.winfo_height()
                    if canvas_width > 1 and canvas_height > 1:
                        pil_image = pil_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(image=pil_image)
                    self.canvas_camera.create_image(0, 0, image=photo, anchor='nw')
                    self.canvas_camera.image = photo
                except tk.TclError:
                    break
            
            except Exception as e:
                print(f"[LỖI] recognition_loop: {e}")
                import traceback
                traceback.print_exc()
            
            try:
                if self.root.winfo_exists():
                    self.root.update_idletasks()
            except:
                break
            
            import time
            time.sleep(0.03)
    
    def reset_to_waiting_state(self):
        """Reset về trạng thái chờ"""
        self.recognized_person = None
        self.liveness_detector.reset()
        self.status_icon_label.config(text="👤", fg='#e0e0e0')
        self.name_label.config(text="Chờ nhận diện...", fg='#2c3e50')
        self.id_label.config(text="Mã: ------")
        self.dept_label.config(text="Phòng: ------")
        self.conf_label.config(text="Độ chính xác: ---%")
        self.liveness_label.config(text="Liveness: ---", fg='#856404')
        self.liveness_badge.config(bg='#fff3cd')
        self.status_label.config(text="", fg='#999')
    
    def update_recognized_status(self):
        """Cập nhật trạng thái đã nhận diện"""
        self.status_label.config(
            text="✓ Đã nhận diện - Vẫn trong khung hình",
            fg='#27ae60'
        )
    
    def update_student_info(self, identity, confidence, face_img):
        """Cập nhật thông tin sinh viên"""
        try:
            persons = self.db.get_all_persons()
            person = next((p for p in persons if p['full_name'] == identity), None)
            
            if person:
                self.recognized_person = {
                    'name': identity,
                    'person_id': person['id']
                }
                
                self.status_icon_label.config(text="✓", fg='#27ae60')
                self.name_label.config(text=identity, fg='#27ae60')
                
                emp_id = person.get('employee_id') or 'Chưa cập nhật'
                self.id_label.config(text=f"Mã: {emp_id}")
                
                dept = person.get('department') or 'Chưa cập nhật'
                self.dept_label.config(text=f"Phòng: {dept}")
                
                self.conf_label.config(text=f"Độ chính xác: {confidence*100:.1f}%")
                
                self.status_label.config(
                    text=f"✓ Đã xác nhận nhận diện thành công!",
                    fg='#27ae60'
                )
                
                self.db.add_recognition_log(
                    person_id=person['id'],
                    identified_name=identity,
                    confidence=confidence
                )
                
                print(f"[INFO] Attendance logged: {identity} - {confidence*100:.1f}%")
        
        except Exception as e:
            print(f"[LỖI] update_student_info: {e}")
    
    def __del__(self):
        """Cleanup"""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceRecognitionUI(root)
    root.mainloop()