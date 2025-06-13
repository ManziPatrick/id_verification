import torch
import numpy as np
import cv2
import dlib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Import face recognition libraries
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[WARNING] face_recognition not available. Install with: pip install face-recognition")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARNING] DeepFace not available. Install with: pip install deepface")

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("[WARNING] FaceNet PyTorch not available. Install with: pip install facenet-pytorch")

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[WARNING] InsightFace not available. Install with: pip install insightface")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[WARNING] SentenceTransformers not available. Install with: pip install sentence-transformers")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] Scikit-learn not available. Install with: pip install scikit-learn")


class AdvancedFaceVerifier:
    """Advanced multi-model face verification system with visual analysis"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        """Initialize all available face recognition models"""
        self.confidence_threshold = confidence_threshold
        self.models_status = {}
        self.face_locations = {}
        
        # Initialize all models
        self._init_dlib()
        self._init_facenet()
        self._init_insightface()
        self._init_additional_models()
        
        # DeepFace and face_recognition don't need initialization
        self.models_status['deepface'] = DEEPFACE_AVAILABLE
        self.models_status['face_recognition'] = FACE_RECOGNITION_AVAILABLE
        
        print(f"[INFO] Model initialization status: {self.models_status}")
        
    def _init_dlib(self):
        """Initialize dlib models"""
        try:
            self.dlib_detector = dlib.get_frontal_face_detector()
            
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            face_model_path = "dlib_face_recognition_resnet_model_v1.dat"
            
            if os.path.exists(predictor_path) and os.path.exists(face_model_path):
                self.dlib_predictor = dlib.shape_predictor(predictor_path)
                self.dlib_face_rec = dlib.face_recognition_model_v1(face_model_path)
                self.models_status['dlib'] = True
                print("[INFO] Dlib models loaded successfully")
            else:
                self.models_status['dlib'] = False
                print("[WARNING] Dlib model files not found")
                
        except Exception as e:
            self.models_status['dlib'] = False
            print(f"[WARNING] Dlib initialization failed: {e}")
    
    def _init_facenet(self):
        """Initialize FaceNet models"""
        try:
            if FACENET_AVAILABLE:
                self.mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)
                self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
                self.models_status['facenet'] = True
                print("[INFO] FaceNet models loaded successfully")
            else:
                self.models_status['facenet'] = False
        except Exception as e:
            self.models_status['facenet'] = False
            print(f"[WARNING] FaceNet initialization failed: {e}")
    
    def _init_insightface(self):
        """Initialize InsightFace models"""
        try:
            if INSIGHTFACE_AVAILABLE:
                self.insightface_app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
                self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
                self.models_status['insightface'] = True
                print("[INFO] InsightFace models loaded successfully")
            else:
                self.models_status['insightface'] = False
        except Exception as e:
            self.models_status['insightface'] = False
            print(f"[WARNING] InsightFace initialization failed: {e}")
    
    def _init_additional_models(self):
        """Initialize additional models for better accuracy"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use CLIP for additional face similarity
                self.clip_model = SentenceTransformer('clip-ViT-B-32')
                self.models_status['clip'] = True
                print("[INFO] CLIP model loaded successfully")
            else:
                self.models_status['clip'] = False
        except Exception as e:
            self.models_status['clip'] = False
            print(f"[WARNING] CLIP initialization failed: {e}")
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess image"""
        try:
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not read image from {image_path}")
            else:
                image = image_path
            
            # Ensure image is in RGB format for some models
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, image_rgb
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def detect_and_extract_faces(self, image_path: str) -> Dict[str, Any]:
        """Detect faces using multiple methods and return best detection"""
        detections = {}
        
        try:
            img_bgr, img_rgb = self.preprocess_image(image_path)
            
            # Method 1: Dlib detection
            if self.models_status.get('dlib'):
                faces = self.dlib_detector(img_rgb, 1)
                if faces:
                    detections['dlib'] = [(face.left(), face.top(), face.right(), face.bottom()) 
                                        for face in faces]
            
            # Method 2: face_recognition detection
            if self.models_status.get('face_recognition'):
                face_locations = face_recognition.face_locations(img_rgb)
                if face_locations:
                    detections['face_recognition'] = [(left, top, right, bottom) 
                                                    for (top, right, bottom, left) in face_locations]
            
            # Method 3: MTCNN detection
            if self.models_status.get('facenet'):
                pil_image = Image.fromarray(img_rgb)
                boxes, _ = self.mtcnn.detect(pil_image)
                if boxes is not None:
                    detections['mtcnn'] = [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) 
                                         for box in boxes]
            
            # Method 4: InsightFace detection
            if self.models_status.get('insightface'):
                faces = self.insightface_app.get(img_bgr)
                if faces:
                    detections['insightface'] = [(int(face.bbox[0]), int(face.bbox[1]), 
                                                int(face.bbox[2]), int(face.bbox[3])) 
                                               for face in faces]
            
            return {
                'detections': detections,
                'image_shape': img_rgb.shape,
                'total_faces': sum(len(faces) for faces in detections.values())
            }
            
        except Exception as e:
            return {'error': f"Face detection failed: {str(e)}", 'detections': {}}
    
    def extract_face_features(self, image_path: str) -> Dict[str, Any]:
        """Extract comprehensive face features using multiple models"""
        features = {}
        
        try:
            # Get face detections first
            detection_result = self.detect_and_extract_faces(image_path)
            if 'error' in detection_result:
                return detection_result
            
            img_bgr, img_rgb = self.preprocess_image(image_path)
            
            # Extract features using different models
            if self.models_status.get('dlib'):
                features['dlib'] = self._extract_dlib_features(img_rgb)
            
            if self.models_status.get('face_recognition'):
                features['face_recognition'] = self._extract_face_recognition_features(img_rgb)
            
            if self.models_status.get('facenet'):
                features['facenet'] = self._extract_facenet_features(image_path)
            
            if self.models_status.get('insightface'):
                features['insightface'] = self._extract_insightface_features(img_bgr)
            
            if self.models_status.get('deepface'):
                features['deepface'] = self._extract_deepface_features(image_path)
            
            features['detections'] = detection_result
            
            return features
            
        except Exception as e:
            return {'error': f"Feature extraction failed: {str(e)}"}
    
    def _extract_dlib_features(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """Extract Dlib features"""
        try:
            faces = self.dlib_detector(img_rgb, 1)
            if not faces:
                return {'error': 'No face detected'}
            
            face = faces[0]
            shape = self.dlib_predictor(img_rgb, face)
            descriptor = np.array(self.dlib_face_rec.compute_face_descriptor(img_rgb, shape))
            
            # Extract landmarks
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            return {
                'descriptor': descriptor,
                'landmarks': landmarks,
                'bbox': (face.left(), face.top(), face.right(), face.bottom()),
                'feature_dim': len(descriptor)
            }
        except Exception as e:
            return {'error': f"Dlib feature extraction failed: {str(e)}"}
    
    def _extract_face_recognition_features(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """Extract face_recognition features"""
        try:
            encodings = face_recognition.face_encodings(img_rgb)
            if not encodings:
                return {'error': 'No face detected'}
            
            landmarks = face_recognition.face_landmarks(img_rgb)
            locations = face_recognition.face_locations(img_rgb)
            
            return {
                'encoding': encodings[0],
                'landmarks': landmarks[0] if landmarks else None,
                'location': locations[0] if locations else None,
                'feature_dim': len(encodings[0])
            }
        except Exception as e:
            return {'error': f"face_recognition feature extraction failed: {str(e)}"}
    
    def _extract_facenet_features(self, image_path: str) -> Dict[str, Any]:
        """Extract FaceNet features"""
        try:
            img = Image.open(image_path).convert('RGB')
            face_tensor = self.mtcnn(img)
            
            if face_tensor is None:
                return {'error': 'No face detected'}
            
            with torch.no_grad():
                embedding = self.facenet_model(face_tensor.unsqueeze(0))
            
            return {
                'embedding': embedding.numpy().flatten(),
                'feature_dim': embedding.shape[1]
            }
        except Exception as e:
            return {'error': f"FaceNet feature extraction failed: {str(e)}"}
    
    def _extract_insightface_features(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Extract InsightFace features"""
        try:
            faces = self.insightface_app.get(img_bgr)
            if not faces:
                return {'error': 'No face detected'}
            
            face = faces[0]
            return {
                'embedding': face.embedding,
                'bbox': face.bbox,
                'kps': face.kps,
                'age': getattr(face, 'age', None),
                'gender': getattr(face, 'gender', None),
                'feature_dim': len(face.embedding)
            }
        except Exception as e:
            return {'error': f"InsightFace feature extraction failed: {str(e)}"}
    
    def _extract_deepface_features(self, image_path: str) -> Dict[str, Any]:
        """Extract DeepFace features using multiple models"""
        try:
            models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'ArcFace']
            features = {}
            
            for model in models:
                try:
                    embedding = DeepFace.represent(image_path, model_name=model, enforce_detection=False)
                    features[model] = {
                        'embedding': np.array(embedding[0]['embedding']),
                        'feature_dim': len(embedding[0]['embedding'])
                    }
                except:
                    continue
            
            return features if features else {'error': 'No DeepFace models worked'}
        except Exception as e:
            return {'error': f"DeepFace feature extraction failed: {str(e)}"}
    
    def calculate_similarity_metrics(self, features1: Dict, features2: Dict) -> Dict[str, Any]:
        """Calculate comprehensive similarity metrics"""
        similarities = {}
        
        # Dlib similarity
        if ('dlib' in features1 and 'dlib' in features2 and 
            'error' not in features1['dlib'] and 'error' not in features2['dlib']):
            desc1 = features1['dlib']['descriptor']
            desc2 = features2['dlib']['descriptor']
            distance = np.linalg.norm(desc1 - desc2)
            similarities['dlib'] = {
                'distance': float(distance),
                'similarity': max(0.0, 1.0 - distance / 2.0),
                'match': distance < 0.6
            }
        
        # face_recognition similarity
        if ('face_recognition' in features1 and 'face_recognition' in features2 and
            'error' not in features1['face_recognition'] and 'error' not in features2['face_recognition']):
            enc1 = features1['face_recognition']['encoding']
            enc2 = features2['face_recognition']['encoding']
            distance = np.linalg.norm(enc1 - enc2)
            similarities['face_recognition'] = {
                'distance': float(distance),
                'similarity': max(0.0, 1.0 - distance),
                'match': distance < 0.6
            }
        
        # FaceNet similarity
        if ('facenet' in features1 and 'facenet' in features2 and
            'error' not in features1['facenet'] and 'error' not in features2['facenet']):
            emb1 = features1['facenet']['embedding']
            emb2 = features2['facenet']['embedding']
            
            # Cosine similarity
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            euclidean_dist = np.linalg.norm(emb1 - emb2)
            
            similarities['facenet'] = {
                'cosine_similarity': float(cos_sim),
                'euclidean_distance': float(euclidean_dist),
                'similarity': float(cos_sim),
                'match': cos_sim > 0.5
            }
        
        # InsightFace similarity
        if ('insightface' in features1 and 'insightface' in features2 and
            'error' not in features1['insightface'] and 'error' not in features2['insightface']):
            emb1 = features1['insightface']['embedding']
            emb2 = features2['insightface']['embedding']
            
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities['insightface'] = {
                'cosine_similarity': float(cos_sim),
                'similarity': float(cos_sim),
                'match': cos_sim > 0.4
            }
        
        # DeepFace similarities
        if ('deepface' in features1 and 'deepface' in features2 and
            'error' not in features1['deepface'] and 'error' not in features2['deepface']):
            deepface_sims = {}
            
            for model in features1['deepface']:
                if model in features2['deepface']:
                    emb1 = features1['deepface'][model]['embedding']
                    emb2 = features2['deepface'][model]['embedding']
                    
                    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    deepface_sims[model] = {
                        'cosine_similarity': float(cos_sim),
                        'similarity': float(cos_sim),
                        'match': cos_sim > 0.5
                    }
            
            similarities['deepface'] = deepface_sims
        
        return similarities
    
    def visualize_face_comparison(self, img1_path: str, img2_path: str, 
                                features1: Dict, features2: Dict, 
                                similarities: Dict, save_path: str = None) -> None:
        """Create comprehensive visual comparison of faces"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Face Verification Analysis', fontsize=16, fontweight='bold')
        
        # Load images
        img1_bgr, img1_rgb = self.preprocess_image(img1_path)
        img2_bgr, img2_rgb = self.preprocess_image(img2_path)
        
        # Plot original images
        axes[0, 0].imshow(img1_rgb)
        axes[0, 0].set_title('Image 1 (ID Card)', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(img2_rgb)
        axes[1, 0].set_title('Image 2 (Selfie)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Draw face detections
        img1_with_faces = self._draw_face_detections(img1_rgb.copy(), features1)
        img2_with_faces = self._draw_face_detections(img2_rgb.copy(), features2)
        
        axes[0, 1].imshow(img1_with_faces)
        axes[0, 1].set_title('Face Detection Results', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 1].imshow(img2_with_faces)
        axes[1, 1].set_title('Face Detection Results', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Plot similarity scores
        self._plot_similarity_scores(axes[0, 2], similarities)
        
        # Plot feature comparison
        self._plot_feature_comparison(axes[1, 2], features1, features2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[INFO] Visualization saved to {save_path}")
        
        plt.show()
    
    def _draw_face_detections(self, img: np.ndarray, features: Dict) -> np.ndarray:
        """Draw face detection boxes and landmarks on image"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        # Draw bounding boxes from different models
        model_idx = 0
        for model_name, model_features in features.items():
            if model_name == 'detections' or 'error' in model_features:
                continue
                
            color = colors[model_idx % len(colors)]
            
            if 'bbox' in model_features:
                bbox = model_features['bbox']
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(img, model_name, (int(bbox[0]), int(bbox[1]-10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks if available
            if 'landmarks' in model_features and model_features['landmarks'] is not None:
                landmarks = model_features['landmarks']
                if isinstance(landmarks, dict):  # face_recognition format
                    for feature_name, points in landmarks.items():
                        for point in points:
                            cv2.circle(img, tuple(point), 1, color, -1)
                elif isinstance(landmarks, np.ndarray):  # dlib format
                    for point in landmarks:
                        cv2.circle(img, (int(point[0]), int(point[1])), 1, color, -1)
            
            model_idx += 1
        
        return img
    
    def _plot_similarity_scores(self, ax, similarities: Dict) -> None:
        """Plot similarity scores as a bar chart"""
        model_names = []
        similarity_scores = []
        match_status = []
        
        for model, sim_data in similarities.items():
            if model == 'deepface':
                for sub_model, sub_data in sim_data.items():
                    model_names.append(f"{model}_{sub_model}")
                    similarity_scores.append(sub_data['similarity'])
                    match_status.append(sub_data['match'])
            else:
                model_names.append(model)
                similarity_scores.append(sim_data['similarity'])
                match_status.append(sim_data['match'])
        
        colors = ['green' if match else 'red' for match in match_status]
        
        bars = ax.bar(range(len(model_names)), similarity_scores, color=colors, alpha=0.7)
        ax.set_xlabel('Models')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Similarity Scores by Model', fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, similarity_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_feature_comparison(self, ax, features1: Dict, features2: Dict) -> None:
        """Plot feature dimension comparison"""
        model_names = []
        feature_dims = []
        
        for model in features1:
            if model in features2 and model != 'detections':
                if 'error' not in features1[model] and 'error' not in features2[model]:
                    if model == 'deepface':
                        for sub_model in features1[model]:
                            if sub_model in features2[model]:
                                model_names.append(f"{model}_{sub_model}")
                                feature_dims.append(features1[model][sub_model]['feature_dim'])
                    else:
                        model_names.append(model)
                        feature_dims.append(features1[model]['feature_dim'])
        
        bars = ax.bar(range(len(model_names)), feature_dims, 
                     color='skyblue', alpha=0.7, edgecolor='navy')
        ax.set_xlabel('Models')
        ax.set_ylabel('Feature Dimensions')
        ax.set_title('Feature Dimensions by Model', fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, dim in zip(bars, feature_dims):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(feature_dims)*0.01,
                   f'{dim}', ha='center', va='bottom', fontsize=8)
    
    def comprehensive_face_verification(self, img1_path: str, img2_path: str, 
                                      visualize: bool = True) -> Dict[str, Any]:
        """Perform comprehensive face verification with visual analysis"""
        
        print(f"[INFO] Starting comprehensive face verification...")
        print(f"  Image 1: {img1_path}")
        print(f"  Image 2: {img2_path}")
        
        # Extract features from both images
        print("[INFO] Extracting features from Image 1...")
        features1 = self.extract_face_features(img1_path)
        
        print("[INFO] Extracting features from Image 2...")
        features2 = self.extract_face_features(img2_path)
        
        if 'error' in features1 or 'error' in features2:
            return {
                'error': 'Feature extraction failed',
                'features1': features1,
                'features2': features2
            }
        
        # Calculate similarities
        print("[INFO] Calculating similarity metrics...")
        similarities = self.calculate_similarity_metrics(features1, features2)
        
        # Aggregate results
        valid_similarities = {}
        for model, sim_data in similarities.items():
            if model == 'deepface':
                for sub_model, sub_data in sim_data.items():
                    valid_similarities[f"{model}_{sub_model}"] = sub_data
            else:
                valid_similarities[model] = sim_data
        
        # Calculate ensemble metrics
        similarity_scores = [data['similarity'] for data in valid_similarities.values()]
        match_results = [data['match'] for data in valid_similarities.values()]
        
        if not similarity_scores:
            return {
                'error': 'No valid similarity calculations',
                'features1': features1,
                'features2': features2
            }
        
        # Weighted ensemble
        model_weights = {
            'insightface': 0.25,
            'facenet': 0.20,
            'dlib': 0.15,
            'face_recognition': 0.10,
            'deepface_VGG-Face': 0.10,
            'deepface_Facenet': 0.08,
            'deepface_ArcFace': 0.07,
            'deepface_OpenFace': 0.03,
            'deepface_DeepFace': 0.02
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for model_key, sim_data in valid_similarities.items():
            weight = model_weights.get(model_key, 0.05)
            weighted_score += weight * sim_data['similarity']
            total_weight += weight
        
        final_confidence = weighted_score / total_weight if total_weight > 0 else 0.0
        match_ratio = np.mean(match_results)
        final_match = (final_confidence >= self.confidence_threshold) and (match_ratio >= 0.5)
        
        # Quality assessment
        quality_score = self._assess_quality(valid_similarities, features1, features2)
        
        result = {
            'match': final_match,
            'confidence': float(final_confidence),
            'match_ratio': float(match_ratio),
            'quality_score': quality_score,
            'individual_similarities': valid_similarities,
            'features1': features1,
            'features2': features2,
            'models_used': len(valid_similarities),
            'recommendation': self._get_recommendation(final_confidence, match_ratio, quality_score)
        }
        
        # Create visualization
        if visualize:
            try:
                self.visualize_face_comparison(img1_path, img2_path, features1, features2, similarities)
            except Exception as e:
                print(f"[WARNING] Visualization failed: {e}")
        
        print(f"[INFO] Verification complete:")
        print(f"  Final Match: {result['match']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Quality Score: {result['quality_score']:.3f}")
        print(f"  Models Used: {result['models_used']}")
        print(f"  Recommendation: {result['recommendation']}")
        
        return result
    
    def _assess_quality(self, similarities: Dict, features1: Dict, features2: Dict) -> float:
        """Assess verification quality"""
        # Model agreement
        matches = [sim['match'] for sim in similarities.values()]
        agreement = 1.0 if (all(matches) or not any(matches)) else (sum(matches) / len(matches))
        
        # Confidence consistency
        confidences = [sim['similarity'] for sim in similarities.values()]
        conf_std = np.std(confidences) if len(confidences) > 1 else 0.0
        consistency = max(0.0, 1.0 - conf_std)
        
        # Model coverage
        coverage = min(1.0, len(similarities) / 5.0)
        
        # Face detection quality
        det_quality1 = features1.get('detections', {}).get('total_faces', 0)
        det_quality2 = features2.get('detections', {}).get('total_faces', 0)
        detection_quality = 1.0 if (det_quality1 > 0 and det_quality2 > 0) else 0.0
        
        # Combine quality metrics
        quality = (agreement * 0.3 + consistency * 0.3 + coverage * 0.2 + detection_quality * 0.2)
        return float(quality)
    
    def _get_recommendation(self, confidence: float, match_ratio: float, quality: float) -> str:
        """Get verification recommendation"""
        if quality < 0.3:
            return "LOW_QUALITY - Manual review recommended"
        elif confidence >= 0.9 and match_ratio >= 0.8:
            return "HIGH_CONFIDENCE - Verification passed"
        elif confidence >= 0.7 and match_ratio >= 0.6:
            return "MEDIUM_CONFIDENCE - Verification likely passed"
        elif confidence <= 0.3 and match_ratio <= 0.4:
            return "HIGH_CONFIDENCE - Verification failed"
        else:
            return "UNCERTAIN - Manual review recommended"
    
    def batch_verification(self, image_pairs: List[Tuple[str, str]], 
                         save_results: bool = True) -> Dict[str, Any]:
        """Perform batch face verification on multiple image pairs"""
        results = {}
        summary_stats = {
            'total_pairs': len(image_pairs),
            'matches': 0,
            'non_matches': 0,
            'errors': 0,
            'avg_confidence': 0.0,
            'avg_quality': 0.0
        }
        
        confidences = []
        qualities = []
        
        for i, (img1, img2) in enumerate(image_pairs):
            print(f"[INFO] Processing pair {i+1}/{len(image_pairs)}: {img1} vs {img2}")
            
            result = self.comprehensive_face_verification(img1, img2, visualize=False)
            
            pair_key = f"pair_{i+1}_{os.path.basename(img1)}_vs_{os.path.basename(img2)}"
            results[pair_key] = result
            
            if 'error' in result:
                summary_stats['errors'] += 1
            else:
                if result['match']:
                    summary_stats['matches'] += 1
                else:
                    summary_stats['non_matches'] += 1
                
                confidences.append(result['confidence'])
                qualities.append(result['quality_score'])
        
        # Calculate summary statistics
        if confidences:
            summary_stats['avg_confidence'] = float(np.mean(confidences))
            summary_stats['avg_quality'] = float(np.mean(qualities))
            summary_stats['confidence_std'] = float(np.std(confidences))
            summary_stats['quality_std'] = float(np.std(qualities))
        
        batch_result = {
            'summary': summary_stats,
            'individual_results': results,
            'processing_timestamp': np.datetime64('now').astype(str)
        }
        
        # Save results if requested
        if save_results:
            import json
            timestamp = str(np.datetime64('now')).replace(':', '-')
            filename = f"batch_verification_results_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_safe_results = self._make_json_safe(batch_result)
            
            with open(filename, 'w') as f:
                json.dump(json_safe_results, f, indent=2)
            print(f"[INFO] Batch results saved to {filename}")
        
        return batch_result
    
    def _make_json_safe(self, obj):
        """Convert numpy arrays and other non-JSON serializable objects"""
        if isinstance(obj, dict):
            return {key: self._make_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def create_verification_report(self, img1_path: str, img2_path: str, 
                                 result: Dict[str, Any] = None) -> str:
        """Create a detailed verification report"""
        if result is None:
            result = self.comprehensive_face_verification(img1_path, img2_path, visualize=False)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            FACE VERIFICATION REPORT                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 VERIFICATION DETAILS:
   • Image 1: {os.path.basename(img1_path)}
   • Image 2: {os.path.basename(img2_path)}
   • Processing Time: {np.datetime64('now').astype(str)}

🎯 OVERALL RESULT:
   • MATCH: {'✅ YES' if result.get('match', False) else '❌ NO'}
   • Confidence: {result.get('confidence', 0.0):.3f} ({result.get('confidence', 0.0)*100:.1f}%)
   • Quality Score: {result.get('quality_score', 0.0):.3f} ({result.get('quality_score', 0.0)*100:.1f}%)
   • Match Ratio: {result.get('match_ratio', 0.0):.3f} ({result.get('match_ratio', 0.0)*100:.1f}%)
   • Models Used: {result.get('models_used', 0)}

💡 RECOMMENDATION: {result.get('recommendation', 'Unknown')}

📊 INDIVIDUAL MODEL RESULTS:
"""
        
        if 'individual_similarities' in result:
            for model, sim_data in result['individual_similarities'].items():
                match_icon = '✅' if sim_data.get('match', False) else '❌'
                report += f"   • {model:20s}: {match_icon} Similarity: {sim_data.get('similarity', 0.0):.3f}\n"
        
        report += f"""
🔍 TECHNICAL DETAILS:
   • Threshold Used: {self.confidence_threshold}
   • Available Models: {list(self.models_status.keys())}
   • Active Models: {[k for k, v in self.models_status.items() if v]}

📈 QUALITY ASSESSMENT:
"""
        
        # Add quality breakdown if available
        quality = result.get('quality_score', 0.0)
        if quality >= 0.8:
            report += "   • Quality Level: HIGH ⭐⭐⭐\n"
        elif quality >= 0.6:
            report += "   • Quality Level: MEDIUM ⭐⭐\n"
        elif quality >= 0.4:
            report += "   • Quality Level: LOW ⭐\n"
        else:
            report += "   • Quality Level: VERY LOW ⚠️\n"
        
        report += f"""
🚀 PERFORMANCE METRICS:
   • Face Detection Success: {'Yes' if result.get('features1', {}).get('detections', {}).get('total_faces', 0) > 0 else 'No'}
   • Feature Extraction: {'Success' if 'error' not in result else 'Failed'}
   • Multi-Model Agreement: {result.get('match_ratio', 0.0)*100:.1f}%

═══════════════════════════════════════════════════════════════════════════════
        """
        
        return report


class AdvancedIDCardProcessor:
    """Enhanced ID Card Processor with advanced face verification"""
    
    def __init__(self, 
                 mongo_uri: str = "mongodb://localhost:27017/",
                 db_name: str = "id_verification",
                 face_confidence_threshold: float = 0.8):
        
        # MongoDB setup
        try:
            from pymongo import MongoClient
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db.id_cards
            self.mongodb_available = True
            print("[INFO] MongoDB connection established")
        except ImportError:
            self.mongodb_available = False
            print("[WARNING] MongoDB not available. Install with: pip install pymongo")
        except Exception as e:
            self.mongodb_available = False
            print(f"[WARNING] MongoDB connection failed: {e}")
        
        # Initialize advanced face verifier
        self.face_verifier = AdvancedFaceVerifier(confidence_threshold=face_confidence_threshold)
        
        print("[INFO] Advanced ID Card Processor initialized")
    
    def process_id_verification(self, id_card_path: str, selfie_path: str, 
                              user_id: str = None, save_report: bool = True) -> Dict[str, Any]:
        """Complete ID verification process with enhanced face matching"""
        
        verification_id = f"verification_{np.datetime64('now').astype(str).replace(':', '-')}"
        
        print(f"[INFO] Starting ID verification process (ID: {verification_id})")
        
        # Perform face verification
        face_result = self.face_verifier.comprehensive_face_verification(
            id_card_path, selfie_path, visualize=True
        )
        
        # Generate detailed report
        report = self.face_verifier.create_verification_report(
            id_card_path, selfie_path, face_result
        )
        
        # Compile final result
        final_result = {
            'verification_id': verification_id,
            'user_id': user_id,
            'id_card_path': id_card_path,
            'selfie_path': selfie_path,
            'face_verification': face_result,
            'overall_status': 'PASSED' if face_result.get('match', False) else 'FAILED',
            'confidence_score': face_result.get('confidence', 0.0),
            'quality_score': face_result.get('quality_score', 0.0),
            'recommendation': face_result.get('recommendation', 'Manual review required'),
            'processing_timestamp': str(np.datetime64('now')),
            'detailed_report': report
        }
        
        # Save to database if available
        if self.mongodb_available:
            try:
                # Convert numpy arrays for MongoDB storage
                mongo_result = self.face_verifier._make_json_safe(final_result)
                self.collection.insert_one(mongo_result)
                print(f"[INFO] Results saved to MongoDB with ID: {verification_id}")
            except Exception as e:
                print(f"[WARNING] Failed to save to MongoDB: {e}")
        
        # Save report to file if requested
        if save_report:
            report_filename = f"verification_report_{verification_id}.txt"
            with open(report_filename, 'w') as f:
                f.write(report)
            print(f"[INFO] Detailed report saved to {report_filename}")
        
        # Print summary
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        return final_result
    
    def batch_process_verifications(self, verification_pairs: List[Dict]) -> Dict[str, Any]:
        """Process multiple ID verifications in batch"""
        batch_id = f"batch_{np.datetime64('now').astype(str).replace(':', '-')}"
        results = []
        
        print(f"[INFO] Starting batch processing (Batch ID: {batch_id})")
        print(f"[INFO] Processing {len(verification_pairs)} verification pairs")
        
        for i, pair_info in enumerate(verification_pairs):
            print(f"\n[INFO] Processing verification {i+1}/{len(verification_pairs)}")
            
            id_card = pair_info.get('id_card_path')
            selfie = pair_info.get('selfie_path')
            user_id = pair_info.get('user_id', f'user_{i+1}')
            
            result = self.process_id_verification(
                id_card, selfie, user_id, save_report=False
            )
            results.append(result)
        
        # Create batch summary
        passed = sum(1 for r in results if r['overall_status'] == 'PASSED')
        failed = len(results) - passed
        avg_confidence = np.mean([r['confidence_score'] for r in results])
        avg_quality = np.mean([r['quality_score'] for r in results])
        
        batch_summary = {
            'batch_id': batch_id,
            'total_verifications': len(results),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(results) * 100,
            'average_confidence': float(avg_confidence),
            'average_quality': float(avg_quality),
            'individual_results': results,
            'processing_completed': str(np.datetime64('now'))
        }
        
        # Save batch results
        batch_filename = f"batch_results_{batch_id}.json"
        import json
        json_safe_summary = self.face_verifier._make_json_safe(batch_summary)
        
        with open(batch_filename, 'w') as f:
            json.dump(json_safe_summary, f, indent=2)
        
        print(f"\n[INFO] Batch processing completed!")
        print(f"[INFO] Results: {passed}/{len(results)} passed ({batch_summary['success_rate']:.1f}%)")
        print(f"[INFO] Batch results saved to {batch_filename}")
        
        return batch_summary


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Initializing Advanced Face Verification System...")
    
    # Initialize the advanced verifier
    verifier = AdvancedFaceVerifier(confidence_threshold=0.8)
    
    # Example single verification
    print("\n📋 Example 1: Single Face Verification")
    id_card_path = "diffid.jpeg"
    selfie_path = "nyagaimaga.jpeg"
    
    if os.path.exists(id_card_path) and os.path.exists(selfie_path):
        result = verifier.comprehensive_face_verification(id_card_path, selfie_path)
        
        # Generate and display report
        report = verifier.create_verification_report(id_card_path, selfie_path, result)
        print(report)
        
        # Example with ID Card Processor
        print("\n📋 Example 2: Complete ID Verification Process")
        processor = AdvancedIDCardProcessor()
        verification_result = processor.process_id_verification(
            id_card_path, selfie_path, user_id="test_user_001"
        )
        
    else:
        print(f"[WARNING] Example images not found:")
        print(f"  - {id_card_path}: {'Found' if os.path.exists(id_card_path) else 'Not found'}")
        print(f"  - {selfie_path}: {'Found' if os.path.exists(selfie_path) else 'Not found'}")
        
        # Create example batch processing template
        print("\n📋 Example 3: Batch Processing Template")
        verification_pairs = [
            {
                'id_card_path': 'id_card_1.jpg',
                'selfie_path': 'selfie_1.jpg',
                'user_id': 'user_001'
            },
            {
                'id_card_path': 'id_card_2.jpg', 
                'selfie_path': 'selfie_2.jpg',
                'user_id': 'user_002'
            }
        ]
        
        print("Example batch processing configuration:")
        import json
        print(json.dumps(verification_pairs, indent=2))
    
    print("\n✅ Advanced Face Verification System ready!")
    print(f"📊 Available models: {[k for k, v in verifier.models_status.items() if v]}")
    print(f"🎯 Confidence threshold: {verifier.confidence_threshold}")
    
    # Print installation instructions for missing dependencies
    missing_deps = [k for k, v in verifier.models_status.items() if not v]
    if missing_deps:
        print(f"\n⚠️  Missing optional dependencies: {missing_deps}")
        print("Install them for better accuracy:")
        print("  pip install face-recognition deepface facenet-pytorch insightface")
        print("  pip install sentence-transformers scikit-learn")
        print("  # For dlib models, download:")
        print("  # shape_predictor_68_face_landmarks.dat")
        print("  # dlib_face_recognition_resnet_model_v1.dat")