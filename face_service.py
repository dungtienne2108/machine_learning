import os
import cv2
import numpy as np
import pickle
import shutil
import unicodedata
from deepface import DeepFace
from numpy.linalg import norm
from config import APP_CONFIG
from database_helper import DatabaseHelper

class FaceRecognitionService:
    def __init__(self):
        self.dataset_folder = APP_CONFIG['dataset_folder']
        self.embedding_file = APP_CONFIG['embedding_file']
        self.model_name = APP_CONFIG['model_name']
        self.detector_backend = APP_CONFIG['detector_backend']
        self.confidence_threshold = APP_CONFIG['confidence_threshold']
        self.db = DatabaseHelper()
        
        # Load OpenCV face cascade for quick detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def normalize_folder_name(self, name):
        """
        Chuẩn hóa tên thư mục để tránh lỗi với tên có dấu
        Giữ nguyên tên có dấu tiếng Việt
        """
        # Loại bỏ các ký tự không hợp lệ cho tên thư mục/file
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        
        # Loại bỏ khoảng trắng thừa
        name = ' '.join(name.split())
        
        return name
    
    def imread_unicode(self, img_path):
        """
        Đọc ảnh với đường dẫn Unicode (hỗ trợ tiếng Việt)
        """
        try:
            # Đọc file dưới dạng numpy array
            with open(img_path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[LỖI] imread_unicode: {e}")
            return None
    
    def imwrite_unicode(self, img_path, img):
        """
        Ghi ảnh với đường dẫn Unicode (hỗ trợ tiếng Việt)
        """
        try:
            # Encode ảnh
            is_success, buffer = cv2.imencode('.jpg', img)
            if is_success:
                # Ghi file
                with open(img_path, 'wb') as f:
                    f.write(buffer)
                return True
            return False
        except Exception as e:
            print(f"[LỖI] imwrite_unicode: {e}")
            return False
    
    def check_image_quality(self, img_path, min_size=80):
        """
        Kiểm tra chất lượng ảnh
        Returns: (is_valid, quality_score, message)
        """
        try:
            # Sử dụng imread_unicode thay vì cv2.imread
            img = self.imread_unicode(img_path)
            if img is None:
                return False, 0.0, "Không đọc được ảnh"
            
            height, width = img.shape[:2]
            print(f"[DEBUG] Kiểm tra ảnh: {img_path}, kích thước: {width}x{height}")
            
            if height < min_size or width < min_size:
                return False, 0.0, f"Ảnh quá nhỏ ({width}x{height})"
            
            # Kiểm tra độ mờ (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score < 50:
                return False, blur_score / 500, "Ảnh bị mờ, vui lòng chụp lại"
            
            # Tính điểm chất lượng
            size_score = min(1.0, (width * height) / (160 * 160))
            blur_normalized = min(1.0, blur_score / 500)
            quality_score = (blur_normalized * 0.6 + size_score * 0.4)
            
            print(f"[DEBUG] Ảnh hợp lệ: blur_score={blur_score:.2f}, quality={quality_score:.2f}")
            return True, quality_score, "Ảnh đạt chất lượng tốt"
                
        except Exception as e:
            print(f"[LỖI] check_image_quality: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, f"Lỗi kiểm tra ảnh: {str(e)}"
    
    def add_person(self, person_info, image_paths):
        """
        Thêm người mới vào hệ thống
        person_info: dict chứa thông tin người
        image_paths: list đường dẫn ảnh
        Returns: (success, message, person_id)
        """
        try:
            full_name = person_info['full_name']
            print(f"[INFO] Bắt đầu thêm người: {full_name}")
            print(f"[INFO] Số ảnh cần xử lý: {len(image_paths)}")
            
            # Kiểm tra chất lượng tất cả ảnh
            valid_images = []
            quality_results = []
            
            for idx, img_path in enumerate(image_paths):
                print(f"[INFO] Kiểm tra ảnh {idx+1}/{len(image_paths)}: {img_path}")
                
                if not os.path.exists(img_path):
                    print(f"[CẢNH BÁO] File không tồn tại: {img_path}")
                    quality_results.append({
                        'path': img_path,
                        'valid': False,
                        'quality': 0.0,
                        'message': 'File không tồn tại'
                    })
                    continue
                
                is_valid, quality, message = self.check_image_quality(img_path)
                quality_results.append({
                    'path': img_path,
                    'valid': is_valid,
                    'quality': quality,
                    'message': message
                })
                
                if is_valid:
                    valid_images.append((img_path, quality))
                    print(f"[INFO] Ảnh hợp lệ: {img_path} (quality: {quality:.2f})")
                else:
                    print(f"[CẢNH BÁO] Ảnh không hợp lệ: {img_path} - {message}")
            
            print(f"[INFO] Tổng số ảnh hợp lệ: {len(valid_images)}/{len(image_paths)}")
            
            if len(valid_images) < APP_CONFIG['min_images_per_person']:
                error_msg = f"Cần ít nhất {APP_CONFIG['min_images_per_person']} ảnh hợp lệ (hiện có {len(valid_images)}/{len(image_paths)}). "
                error_msg += "Chi tiết: " + "; ".join([f"{r['path']}: {r['message']}" for r in quality_results if not r['valid']])
                return False, error_msg, None
            
            # Thêm vào database
            print(f"[INFO] Thêm vào database...")
            person_id = self.db.add_person(
                full_name=full_name,
                employee_id=person_info.get('employee_id'),
                department=person_info.get('department'),
                position=person_info.get('position'),
                email=person_info.get('email'),
                phone=person_info.get('phone'),
                notes=person_info.get('notes')
            )
            print(f"[INFO] Đã thêm vào database với ID: {person_id}")
            
            # Chuẩn hóa tên thư mục (giữ nguyên dấu tiếng Việt)
            normalized_name = self.normalize_folder_name(full_name)
            person_folder = os.path.join(self.dataset_folder, normalized_name)
            
            # Tạo thư mục với encoding UTF-8
            os.makedirs(person_folder, exist_ok=True)
            print(f"[INFO] Đã tạo thư mục: {person_folder}")
            
            # Copy ảnh vào dataset và lưu vào DB
            saved_count = 0
            for i, (img_path, quality) in enumerate(valid_images):
                # Tạo tên file an toàn
                safe_filename = f"{normalized_name}_{i+1}.jpg"
                dest_path = os.path.join(person_folder, safe_filename)
                print(f"[INFO] Đang lưu ảnh {i+1}/{len(valid_images)}: {dest_path}")
                
                # Đọc ảnh bằng imread_unicode
                img = self.imread_unicode(img_path)
                if img is None:
                    print(f"[LỖI] Không đọc được ảnh: {img_path}")
                    continue
                
                print(f"[INFO] Ảnh đọc được, kích thước: {img.shape}")
                
                # Ghi ảnh vào dataset bằng imwrite_unicode
                success = self.imwrite_unicode(dest_path, img)
                if not success:
                    print(f"[LỖI] Không ghi được ảnh: {dest_path}")
                    continue
                
                # Kiểm tra ảnh đã được ghi
                if not os.path.exists(dest_path):
                    print(f"[LỖI] Ảnh không tồn tại sau khi ghi: {dest_path}")
                    continue
                
                file_size = os.path.getsize(dest_path)
                print(f"[INFO] Ảnh đã được ghi, kích thước file: {file_size} bytes")
                
                # Lưu vào database
                try:
                    image_id = self.db.add_person_image(person_id, dest_path, quality)
                    print(f"[INFO] Đã lưu vào database với image_id: {image_id}")
                    saved_count += 1
                except Exception as e:
                    print(f"[LỖI] Không lưu được vào database: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"[INFO] Đã lưu {saved_count}/{len(valid_images)} ảnh vào dataset và database")
            
            if saved_count == 0:
                return False, "Không thể lưu bất kỳ ảnh nào vào dataset", None
            
            return True, f"Đã thêm thành công {saved_count} ảnh", person_id
            
        except Exception as e:
            print(f"[LỖI] add_person: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Lỗi: {str(e)}", None
    
    def generate_embeddings(self):
        """Tạo embeddings cho toàn bộ dataset"""
        embeddings = []
        
        if os.path.exists(self.embedding_file):
            with open(self.embedding_file, "rb") as f:
                embeddings = pickle.load(f)
        
        existing_identities = set(e["identity"] for e in embeddings)
        print(f"[INFO] Đã có {len(existing_identities)} nhãn trong embeddings")
        
        # Đếm số ảnh cần xử lý
        total_images = 0
        processed_images = 0
        
        for root, dirs, files in os.walk(self.dataset_folder):
            label = os.path.basename(root)
            
            # Bỏ qua thư mục gốc
            if label == os.path.basename(self.dataset_folder) or not label:
                continue
            
            # Lấy tất cả ảnh của người này
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(image_files)
            
            # Nếu người này đã có embeddings, bỏ qua
            if label in existing_identities:
                print(f"[INFO] Bỏ qua {label} (đã có embeddings)")
                continue
            
            # Xử lý tất cả ảnh của người này
            for file in image_files:
                img_path = os.path.join(root, file)
                
                # Kiểm tra file tồn tại
                if not os.path.exists(img_path):
                    print(f"[CẢNH BÁO] File không tồn tại: {img_path}")
                    continue
                
                try:
                    # Sử dụng imread_unicode để đọc ảnh
                    img = self.imread_unicode(img_path)
                    if img is None:
                        print(f"[CẢNH BÁO] Không đọc được ảnh: {img_path}")
                        continue
                    
                    # Tạo embedding từ numpy array thay vì đường dẫn
                    embedding = DeepFace.represent(
                        img_path=img,  # Truyền numpy array thay vì path
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False
                    )[0]["embedding"]
                    
                    embeddings.append({
                        "identity": label,
                        "embedding": np.array(embedding, dtype=np.float32)
                    })
                    
                    processed_images += 1
                    print(f"[INFO] Đã thêm embedding cho {label} ({file}) [{processed_images}/{total_images}]")
                    
                except Exception as e:
                    print(f"[CẢNH BÁO] Không trích xuất được embedding cho {img_path}: {e}")
        
        # Lưu embeddings với encoding UTF-8
        with open(self.embedding_file, "wb") as f:
            pickle.dump(embeddings, f)
        
        print(f"[INFO] Hoàn tất: {len(embeddings)} embedding(s) từ {processed_images}/{total_images} ảnh")
        return True
    
    def cosine_distance(self, a, b):
        """Tính khoảng cách cosine"""
        return 1 - np.dot(a, b) / (norm(a) * norm(b))
    
    def find_best_match(self, face_emb, embeddings):
        """Tìm người khớp nhất"""
        min_dist = float("inf")
        identity = "Unknown"
        
        for e in embeddings:
            dist = self.cosine_distance(face_emb, e["embedding"])
            if dist < min_dist:
                min_dist = dist
                identity = e["identity"]
        
        confidence = max(0, 1 - min_dist)
        
        if min_dist > self.confidence_threshold:
            identity = "Unknown"
        
        return identity, confidence
    
    def load_embeddings(self):
        """Load embeddings từ file"""
        if os.path.exists(self.embedding_file):
            with open(self.embedding_file, "rb") as f:
                return pickle.load(f)
        return []
    
    def delete_person_data(self, person_name):
        """Xóa dữ liệu người (thư mục ảnh và embeddings)"""
        try:
            # Chuẩn hóa tên để tìm thư mục đúng
            normalized_name = self.normalize_folder_name(person_name)
            person_folder = os.path.join(self.dataset_folder, normalized_name)
            
            # Xóa thư mục ảnh
            if os.path.exists(person_folder):
                shutil.rmtree(person_folder)
                print(f"[INFO] Đã xóa thư mục: {person_folder}")
            
            # Xóa embeddings của người này
            if os.path.exists(self.embedding_file):
                embeddings = self.load_embeddings()
                # Xóa tất cả embeddings có identity trùng với tên (cả tên gốc và tên chuẩn hóa)
                embeddings = [e for e in embeddings 
                            if e["identity"] not in [person_name, normalized_name]]
                
                with open(self.embedding_file, "wb") as f:
                    pickle.dump(embeddings, f)
                
                print(f"[INFO] Đã xóa embeddings của: {person_name}")
            
            return True
        except Exception as e:
            print(f"[LỖI] delete_person_data: {e}")
            import traceback
            traceback.print_exc()
            return False