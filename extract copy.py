import torch
import numpy as np
import cv2
import dlib
import os
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import face recognition libraries
try:
    import face_recognition  # For face_recognition library
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


class EnhancedFaceVerifier:
    """Enhanced multi-model face verification system"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        """Initialize all available face recognition models"""
        self.confidence_threshold = confidence_threshold
        self.models_status = {}
        
        # Initialize Dlib
        self._init_dlib()
        
        # Initialize FaceNet
        self._init_facenet()
        
        # Initialize InsightFace
        self._init_insightface()
        
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
                self.mtcnn = MTCNN(keep_all=False, device='cpu')
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
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
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
    
    def verify_with_dlib(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """Face verification using Dlib"""
        if not self.models_status.get('dlib', False):
            return {"error": "Dlib not available", "distance": None, "match": False}
        
        try:
            img1_bgr, img1_rgb = self.preprocess_image(img1_path)
            img2_bgr, img2_rgb = self.preprocess_image(img2_path)
            
            # Detect faces
            faces1 = self.dlib_detector(img1_rgb, 1)
            faces2 = self.dlib_detector(img2_rgb, 1)
            
            if len(faces1) == 0 or len(faces2) == 0:
                return {"error": "Face not detected", "distance": None, "match": False}
            
            # Get face descriptors
            shape1 = self.dlib_predictor(img1_rgb, faces1[0])
            shape2 = self.dlib_predictor(img2_rgb, faces2[0])
            
            desc1 = np.array(self.dlib_face_rec.compute_face_descriptor(img1_rgb, shape1))
            desc2 = np.array(self.dlib_face_rec.compute_face_descriptor(img2_rgb, shape2))
            
            # Calculate distance
            distance = np.linalg.norm(desc1 - desc2)
            match = distance < 0.6  # Dlib threshold
            
            return {
                "distance": float(distance),
                "match": match,
                "threshold": 0.6,
                "confidence": max(0.0, 1.0 - distance)
            }
            
        except Exception as e:
            return {"error": f"Dlib verification failed: {str(e)}", "distance": None, "match": False}
    
    def verify_with_face_recognition(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """Face verification using face_recognition library"""
        if not self.models_status.get('face_recognition', False):
            return {"error": "face_recognition not available", "distance": None, "match": False}
        
        try:
            # Load images
            img1 = face_recognition.load_image_file(img1_path)
            img2 = face_recognition.load_image_file(img2_path)
            
            # Get face encodings
            encodings1 = face_recognition.face_encodings(img1)
            encodings2 = face_recognition.face_encodings(img2)
            
            if len(encodings1) == 0 or len(encodings2) == 0:
                return {"error": "Face not detected", "distance": None, "match": False}
            
            # Compare faces
            distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
            match = distance < 0.6  # face_recognition threshold
            
            return {
                "distance": float(distance),
                "match": match,
                "threshold": 0.6,
                "confidence": max(0.0, 1.0 - distance)
            }
            
        except Exception as e:
            return {"error": f"face_recognition failed: {str(e)}", "distance": None, "match": False}
    
    def verify_with_facenet(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """Face verification using FaceNet"""
        if not self.models_status.get('facenet', False):
            return {"error": "FaceNet not available", "distance": None, "match": False}
        
        try:
            # Load and preprocess images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Detect and crop faces
            face1 = self.mtcnn(img1)
            face2 = self.mtcnn(img2)
            
            if face1 is None or face2 is None:
                return {"error": "Face not detected", "distance": None, "match": False}
            
            # Get embeddings
            with torch.no_grad():
                emb1 = self.facenet_model(face1.unsqueeze(0))
                emb2 = self.facenet_model(face2.unsqueeze(0))
            
            # Calculate distance
            distance = torch.dist(emb1, emb2).item()
            match = distance < 1.0  # FaceNet threshold
            
            return {
                "distance": float(distance),
                "match": match,
                "threshold": 1.0,
                "confidence": max(0.0, 1.0 - (distance / 2.0))
            }
            
        except Exception as e:
            return {"error": f"FaceNet verification failed: {str(e)}", "distance": None, "match": False}
    
    def verify_with_insightface(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """Face verification using InsightFace"""
        if not self.models_status.get('insightface', False):
            return {"error": "InsightFace not available", "distance": None, "match": False}
        
        try:
            # Load images
            img1_bgr, img1_rgb = self.preprocess_image(img1_path)
            img2_bgr, img2_rgb = self.preprocess_image(img2_path)
            
            # Get face embeddings
            faces1 = self.insightface_app.get(img1_bgr)
            faces2 = self.insightface_app.get(img2_bgr)
            
            if len(faces1) == 0 or len(faces2) == 0:
                return {"error": "Face not detected", "distance": None, "match": False}
            
            # Get embeddings
            emb1 = faces1[0].embedding
            emb2 = faces2[0].embedding
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            distance = 1.0 - similarity
            match = similarity > 0.4  # InsightFace threshold
            
            return {
                "distance": float(distance),
                "similarity": float(similarity),
                "match": match,
                "threshold": 0.4,
                "confidence": float(similarity)
            }
            
        except Exception as e:
            return {"error": f"InsightFace verification failed: {str(e)}", "distance": None, "match": False}
    
    def verify_with_deepface(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """Face verification using DeepFace"""
        if not self.models_status.get('deepface', False):
            return {"error": "DeepFace not available", "distance": None, "match": False}
        
        try:
            # Use multiple models available in DeepFace
            models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
            results = []
            
            for model in models:
                try:
                    result = DeepFace.verify(img1_path, img2_path, model_name=model, enforce_detection=False)
                    results.append({
                        'model': model,
                        'distance': result['distance'],
                        'verified': result['verified'],
                        'threshold': result['threshold']
                    })
                except:
                    continue
            
            if not results:
                return {"error": "DeepFace verification failed", "distance": None, "match": False}
            
            # Aggregate results
            distances = [r['distance'] for r in results if 'distance' in r]
            matches = [r['verified'] for r in results if 'verified' in r]
            
            avg_distance = np.mean(distances) if distances else None
            match_ratio = np.mean(matches) if matches else 0
            final_match = match_ratio >= 0.5  # Majority vote
            
            return {
                "distance": float(avg_distance) if avg_distance else None,
                "match": final_match,
                "match_ratio": float(match_ratio),
                "model_results": results,
                "confidence": 1.0 - float(avg_distance) if avg_distance else 0.0
            }
            
        except Exception as e:
            return {"error": f"DeepFace verification failed: {str(e)}", "distance": None, "match": False}
    
    def ensemble_verification(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """Ensemble verification using all available models"""
        
        print(f"[INFO] Starting ensemble face verification...")
        
        # Get results from all models
        model_results = {}
        
        if self.models_status.get('dlib'):
            print("[INFO] Running Dlib verification...")
            model_results['dlib'] = self.verify_with_dlib(img1_path, img2_path)
        
        if self.models_status.get('face_recognition'):
            print("[INFO] Running face_recognition verification...")
            model_results['face_recognition'] = self.verify_with_face_recognition(img1_path, img2_path)
        
        if self.models_status.get('facenet'):
            print("[INFO] Running FaceNet verification...")
            model_results['facenet'] = self.verify_with_facenet(img1_path, img2_path)
        
        if self.models_status.get('insightface'):
            print("[INFO] Running InsightFace verification...")
            model_results['insightface'] = self.verify_with_insightface(img1_path, img2_path)
        
        if self.models_status.get('deepface'):
            print("[INFO] Running DeepFace verification...")
            model_results['deepface'] = self.verify_with_deepface(img1_path, img2_path)
        
        # Aggregate results
        valid_results = {k: v for k, v in model_results.items() if 'error' not in v}
        
        if not valid_results:
            return {
                "error": "No models could process the images",
                "match": False,
                "confidence": 0.0,
                "model_results": model_results
            }
        
        # Calculate ensemble metrics
        matches = [result.get('match', False) for result in valid_results.values()]
        confidences = [result.get('confidence', 0.0) for result in valid_results.values()]
        
        # Weighted voting (give more weight to more reliable models)
        model_weights = {
            'insightface': 0.3,
            'facenet': 0.25,
            'dlib': 0.2,
            'deepface': 0.15,
            'face_recognition': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for model_name, result in valid_results.items():
            weight = model_weights.get(model_name, 0.1)
            confidence = result.get('confidence', 0.0)
            weighted_score += weight * confidence
            total_weight += weight
        
        final_confidence = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Final decision
        match_ratio = np.mean(matches)
        final_match = (final_confidence >= self.confidence_threshold) and (match_ratio >= 0.5)
        
        # Quality assessment
        quality_score = self.assess_verification_quality(model_results)
        
        return {
            "match": final_match,
            "confidence": float(final_confidence),
            "match_ratio": float(match_ratio),
            "quality_score": quality_score,
            "total_models_used": len(valid_results),
            "model_results": model_results,
            "ensemble_method": "weighted_voting",
            "threshold_used": self.confidence_threshold,
            "recommendation": self.get_verification_recommendation(final_confidence, match_ratio, quality_score)
        }
    
    def assess_verification_quality(self, model_results: Dict[str, Any]) -> float:
        """Assess the quality of verification based on model agreement"""
        valid_results = {k: v for k, v in model_results.items() if 'error' not in v}
        
        if len(valid_results) < 2:
            return 0.5  # Low quality if only one model works
        
        matches = [result.get('match', False) for result in valid_results.values()]
        confidences = [result.get('confidence', 0.0) for result in valid_results.values()]
        
        # Agreement score
        agreement = 1.0 if all(matches) or not any(matches) else 0.5
        
        # Confidence consistency
        conf_std = np.std(confidences) if len(confidences) > 1 else 0.0
        conf_consistency = max(0.0, 1.0 - conf_std)
        
        # Model coverage (more models = higher quality)
        coverage = min(1.0, len(valid_results) / 3.0)
        
        quality_score = (agreement * 0.4 + conf_consistency * 0.4 + coverage * 0.2)
        
        return float(quality_score)
    
    def get_verification_recommendation(self, confidence: float, match_ratio: float, quality: float) -> str:
        """Get human-readable recommendation"""
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
    
    def verify_faces(self, id_card_path: str, selfie_path: str) -> Dict[str, Any]:
        """Main verification method - uses ensemble approach"""
        try:
            print(f"[INFO] Verifying faces:")
            print(f"  ID Card: {id_card_path}")
            print(f"  Selfie: {selfie_path}")
            
            # Run ensemble verification
            result = self.ensemble_verification(id_card_path, selfie_path)
            
            print(f"[INFO] Verification complete:")
            print(f"  Match: {result.get('match', False)}")
            print(f"  Confidence: {result.get('confidence', 0.0):.3f}")
            print(f"  Quality: {result.get('quality_score', 0.0):.3f}")
            print(f"  Recommendation: {result.get('recommendation', 'Unknown')}")
            
            return result
            
        except Exception as e:
            return {
                "error": f"Face verification error: {str(e)}",
                "match": False,
                "confidence": 0.0,
                "quality_score": 0.0,
                "recommendation": "ERROR - Verification failed"
            }


# Integration with your existing IDCardProcessor
class IDCardProcessor:
    """Updated IDCardProcessor with enhanced face verification"""
    
    def __init__(self, 
                 mongo_uri: str = "mongodb://localhost:27017/",
                 db_name: str = "id_verification",
                 face_confidence_threshold: float = 0.8):
        
        # MongoDB setup (keeping your existing code)
        from pymongo import MongoClient
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db.id_cards
        
        # Initialize enhanced face verifier
        self.face_verifier = EnhancedFaceVerifier(confidence_threshold=face_confidence_threshold)
        
        # Keep your existing back extractor and OCR model initialization
        # ... (rest of your existing __init__ code)
    
    def verify_face(self, id_card_path: str, selfie_path: str) -> Dict[str, Any]:
        """Enhanced face verification using ensemble approach"""
        return self.face_verifier.verify_faces(id_card_path, selfie_path)
    
    # ... (rest of your existing methods remain the same)


# Example usage
if __name__ == "__main__":
    # Test the enhanced face verifier
    verifier = EnhancedFaceVerifier(confidence_threshold=0.8)
    
    # Example paths
    id_card_path = "path/to/id_card.jpg"
    selfie_path = "path/to/selfie.jpg"
    
    # Run verification
    result = verifier.verify_faces(id_card_path, selfie_path)
    
    print("\n🎯 Enhanced Face Verification Results:")
    print(f"Match: {result.get('match', False)}")
    print(f"Confidence: {result.get('confidence', 0.0):.3f}")
    print(f"Quality Score: {result.get('quality_score', 0.0):.3f}")
    print(f"Recommendation: {result.get('recommendation', 'Unknown')}")
    print(f"Models Used: {result.get('total_models_used', 0)}")