import re
import base64
from io import BytesIO
from PIL import Image
import pymongo
from pymongo import MongoClient
from datetime import datetime
from paddleocr import PaddleOCR
from pprint import pprint
from typing import Optional, Dict, Any, List
import json
import os
import cv2
import numpy as np
import dlib
import face_recognition
from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1
import warnings
from collections import Counter
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class MultiOCRProcessor:
    """Handles OCR processing using only PaddleOCR"""
    
    def __init__(self):
        self.engines = {
            'paddle': self._init_paddle(),
            'easyocr': None,
            'doctr': None,
            'tesseract': None
        }
        print("\n[OCR ENGINE INITIALIZATION]")
        for engine, status in self.engines.items():
            print(f"- {engine.upper()}: {'✅ Ready' if status else '❌ Disabled'}")
        
    def _init_paddle(self):
        try:
            print("Initializing PaddleOCR...")
            paddle = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                enable_mkldnn=False,
                rec_algorithm='SVTR_LCNet'
            )
            print("PaddleOCR initialized successfully")
            return paddle
        except Exception as e:
            print(f"[WARNING] PaddleOCR init failed: {e}")
            return None
    
    def _init_easyocr(self):
        return None
    
    def _init_doctr(self):
        return None
    
    def _init_tesseract(self):
        return False
    
    def preprocess_image(self, img_path: str) -> np.ndarray:
        """Standard image preprocessing for all OCR engines"""
        print(f"\nPreprocessing image: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded
    
    def extract_with_paddle(self, img_path: str) -> List[str]:
        if not self.engines['paddle']:
            return []
        try:
            print("\n[PADDLEOCR EXTRACTION]")
            result = self.engines['paddle'].ocr(img_path, cls=True)
            extracted = [line[1][0] for line in result[0]] if result else []
            print(f"Extracted {len(extracted)} items: {extracted}")
            return extracted
        except Exception as e:
            print(f"[ERROR] PaddleOCR extraction failed: {e}")
            return []
    
    def extract_with_easyocr(self, img_path: str) -> List[str]:
        return []
    
    def extract_with_doctr(self, img_path: str) -> List[str]:
        return []
    
    def extract_with_tesseract(self, img_path: str) -> List[str]:
        return []
    
    def extract_consensus_text(self, img_path: str) -> Dict[str, Any]:
        """Get text from PaddleOCR only"""
        print(f"\n[STARTING EXTRACTION FOR {img_path}]")
        
        results = {
            'paddle': self.extract_with_paddle(img_path),
            'easyocr': [],
            'doctr': [],
            'tesseract': []
        }
        
        # Since we're only using PaddleOCR, all extracted text is considered consensus
        text_scores = {}
        for text in results['paddle']:
            if text not in text_scores:
                text_scores[text] = {'count': 1, 'engines': ['paddle']}
        
        consensus = [
            {'text': text, 'confidence': 1.0, 'engines': data['engines']}
            for text, data in text_scores.items()
        ]
        
        # Sort by confidence (highest first)
        consensus.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_result = {
            'consensus_text': [item['text'] for item in consensus],
            'confidence_scores': consensus,
            'all_results': results,
            'engine_availability': {k: v is not None for k, v in self.engines.items()}
        }
        
        print("\n[FINAL EXTRACTION RESULTS]")
        pprint(final_result)
        
        return final_result

class MultiModelFaceVerifier:
    """Integrated face verification using multiple models"""
    
    def __init__(self):
        """Initialize all face verification models"""
        self.models = {
            "dlib": self._init_dlib(),
            "face_recognition": self._init_face_recognition(),
            "deepface_vgg": {"model": "VGG-Face", "enabled": True},
            "deepface_facenet": {"model": "Facenet", "enabled": True},
            "facenet_pytorch": self._init_facenet_pytorch(),
            "deepface_arcface": {"model": "ArcFace", "enabled": True},
            "deepface_openface": {"model": "OpenFace", "enabled": True}
        }
        
        # Model weights for final decision
        self.model_weights = {
            "dlib": 0.15,
            "face_recognition": 0.15,
            "deepface_vgg": 0.15,
            "deepface_facenet": 0.15,
            "facenet_pytorch": 0.15,
            "deepface_arcface": 0.15,
            "deepface_openface": 0.10
        }
        
        # Minimum required similarity for each model
        self.minimum_similarity = 0.40
        
        print("\n[FACE VERIFICATION MODELS INITIALIZED]")
        for model, status in self.models.items():
            print(f"- {model.upper()}: {'✅ Ready' if status.get('enabled', False) else '❌ Disabled'}")
    
    def _init_dlib(self) -> Dict[str, Any]:
        """Initialize dlib models"""
        try:
            detector = dlib.get_frontal_face_detector()
            
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            face_model_path = "dlib_face_recognition_resnet_model_v1.dat"
            
            if os.path.exists(predictor_path) and os.path.exists(face_model_path):
                predictor = dlib.shape_predictor(predictor_path)
                face_rec_model = dlib.face_recognition_model_v1(face_model_path)
                return {
                    "detector": detector,
                    "predictor": predictor,
                    "face_rec_model": face_rec_model,
                    "enabled": True
                }
            else:
                print(f"[WARNING] Dlib model files not found at {predictor_path} or {face_model_path}")
                return {"enabled": False}
        except Exception as e:
            print(f"[WARNING] Failed to initialize dlib: {e}")
            return {"enabled": False}
    
    def _init_face_recognition(self) -> Dict[str, Any]:
        """Initialize face_recognition model"""
        try:
            # Just test loading a small image to verify the model works
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = face_recognition.face_encodings(img)
            return {"enabled": True}
        except Exception as e:
            print(f"[WARNING] Failed to initialize face_recognition: {e}")
            return {"enabled": False}
    
    def _init_facenet_pytorch(self) -> Dict[str, Any]:
        """Initialize FaceNet PyTorch model"""
        try:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            mtcnn = MTCNN(keep_all=True, device=device)
            resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            return {
                "mtcnn": mtcnn,
                "resnet": resnet,
                "device": device,
                "enabled": True
            }
        except Exception as e:
            print(f"[WARNING] Failed to initialize facenet_pytorch: {e}")
            return {"enabled": False}
    
    def read_image(self, image_path: str) -> np.ndarray:
        """Read image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            return image
        except Exception as e:
            raise ValueError(f"Error reading image: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for face detection"""
        try:
            # Convert to RGB (most models expect RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return rgb_image
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def detect_faces_dlib(self, image: np.ndarray) -> List[Any]:
        """Detect faces using dlib"""
        if not self.models["dlib"]["enabled"]:
            return []
            
        try:
            dets = self.models["dlib"]["detector"](image, 1)
            return dets
        except Exception as e:
            print(f"[ERROR] dlib face detection failed: {e}")
            return []
    
    def get_face_descriptor_dlib(self, image: np.ndarray, face) -> Optional[np.ndarray]:
        """Get face descriptor using dlib"""
        if not self.models["dlib"]["enabled"]:
            return None
            
        try:
            shape = self.models["dlib"]["predictor"](image, face)
            face_descriptor = self.models["dlib"]["face_rec_model"].compute_face_descriptor(image, shape)
            return np.array(face_descriptor)
        except Exception as e:
            print(f"[ERROR] dlib face descriptor failed: {e}")
            return None
    
    def compare_faces_dlib(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """Compare faces using dlib"""
        if not self.models["dlib"]["enabled"]:
            return None
            
        try:
            # Detect faces
            dets1 = self.detect_faces_dlib(img1)
            dets2 = self.detect_faces_dlib(img2)
            
            if not dets1 or not dets2:
                return None
                
            # Get descriptors
            desc1 = self.get_face_descriptor_dlib(img1, dets1[0])
            desc2 = self.get_face_descriptor_dlib(img2, dets2[0])
            
            if desc1 is None or desc2 is None:
                return None
                
            # Calculate distance (lower is more similar)
            dist = np.linalg.norm(desc1 - desc2)
            # Convert to similarity score (0-1)
            similarity = 1 - (dist / 1.5)  # Normalize with max expected distance
            return max(0, min(1, similarity))
            
        except Exception as e:
            print(f"[ERROR] dlib face comparison failed: {e}")
            return None
    
    def compare_faces_face_recognition(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """Compare faces using face_recognition library"""
        if not self.models["face_recognition"]["enabled"]:
            return None
            
        try:
            # Get face encodings
            encodings1 = face_recognition.face_encodings(img1)
            encodings2 = face_recognition.face_encodings(img2)
            
            if not encodings1 or not encodings2:
                return None
                
            # Calculate face distance
            distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
            # Convert to similarity (0-1)
            similarity = 1 - distance
            return max(0, min(1, similarity))
            
        except Exception as e:
            print(f"[ERROR] face_recognition comparison failed: {e}")
            return None
    
    def compare_faces_deepface(self, img1: np.ndarray, img2: np.ndarray, model_name: str) -> Optional[float]:
        """Compare faces using DeepFace with specified model"""
        model_info = self.models[f"deepface_{model_name.lower()}"]
        if not model_info["enabled"]:
            return None
            
        try:
            # DeepFace expects BGR images
            img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            
            result = DeepFace.verify(
                img1_path=img1_bgr,
                img2_path=img2_bgr,
                model_name=model_info["model"],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            
            # Different DeepFace models return different similarity metrics
            if 'distance' in result:
                # For models that return distance (smaller is better)
                distance = result['distance']
                # Convert to similarity (0-1)
                if model_info["model"] == "VGG-Face":
                    similarity = 1 - (distance / 1.2)  # VGG-Face max distance ~1.2
                elif model_info["model"] == "Facenet":
                    similarity = 1 - (distance / 1.5)  # Facenet max distance ~1.5
                elif model_info["model"] == "ArcFace":
                    similarity = 1 - (distance / 1.5)  # ArcFace max distance ~1.5
                else:
                    similarity = 1 - distance  # Default
            elif 'similarity_metric' in result:
                # For models that return similarity directly
                similarity = result['similarity_metric']
            else:
                similarity = result['verified'] * 1.0  # Boolean to 0 or 1
                
            return max(0, min(1, similarity))
            
        except Exception as e:
            print(f"[ERROR] DeepFace {model_info['model']} comparison failed: {e}")
            return None
    
    def compare_faces_facenet_pytorch(self, img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
        """Compare faces using FaceNet PyTorch"""
        if not self.models["facenet_pytorch"]["enabled"]:
            return None
            
        try:
            model = self.models["facenet_pytorch"]
            device = model["device"]
            mtcnn = model["mtcnn"]
            resnet = model["resnet"]
            
            # Detect faces
            faces1 = mtcnn(img1)
            faces2 = mtcnn(img2)
            
            if faces1 is None or faces2 is None:
                return None
                
            # Get embeddings
            emb1 = resnet(faces1.unsqueeze(0).to(device)).detach().cpu()
            emb2 = resnet(faces2.unsqueeze(0).to(device)).detach().cpu()
            
            # Calculate distance (Euclidean)
            dist = torch.nn.functional.pairwise_distance(emb1, emb2).item()
            # Convert to similarity (0-1)
            similarity = 1 - (dist / 1.5)  # Normalize with max expected distance
            return max(0, min(1, similarity))
            
        except Exception as e:
            print(f"[ERROR] FaceNet PyTorch comparison failed: {e}")
            return None
    
    def compare_faces_all_models(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """Compare faces using all available models"""
        results = {}
        
        try:
            print("\n[FACE VERIFICATION PROCESS STARTED]")
            print(f"- ID Image: {img1_path}")
            print(f"- Selfie Image: {img2_path}")
            
            # Read and preprocess images
            img1 = self.preprocess_image(self.read_image(img1_path))
            img2 = self.preprocess_image(self.read_image(img2_path))
            
            print("\n[RUNNING FACE COMPARISONS]")
            
            # Dlib comparison
            if self.models["dlib"]["enabled"]:
                print("- Running dlib comparison...")
                dlib_sim = self.compare_faces_dlib(img1, img2)
                results["dlib"] = dlib_sim
                print(f"  dlib similarity: {dlib_sim or 'N/A'}")
            
            # face_recognition comparison
            if self.models["face_recognition"]["enabled"]:
                print("- Running face_recognition comparison...")
                fr_sim = self.compare_faces_face_recognition(img1, img2)
                results["face_recognition"] = fr_sim
                print(f"  face_recognition similarity: {fr_sim or 'N/A'}")
            
            # DeepFace comparisons
            for model in ["vgg", "facenet", "arcface", "openface"]:
                model_key = f"deepface_{model}"
                if self.models[model_key]["enabled"]:
                    print(f"- Running DeepFace {model} comparison...")
                    sim = self.compare_faces_deepface(img1, img2, model)
                    results[model_key] = sim
                    print(f"  DeepFace {model} similarity: {sim or 'N/A'}")
            
            # FaceNet PyTorch comparison
            if self.models["facenet_pytorch"]["enabled"]:
                print("- Running FaceNet PyTorch comparison...")
                fn_sim = self.compare_faces_facenet_pytorch(img1, img2)
                results["facenet_pytorch"] = fn_sim
                print(f"  FaceNet PyTorch similarity: {fn_sim or 'N/A'}")
            
            # Calculate weighted score
            weighted_score = 0.0
            total_weight = 0.0
            valid_models = 0
            
            for model_name, score in results.items():
                if score is not None:
                    weight = self.model_weights.get(model_name, 0.1)
                    weighted_score += score * weight
                    total_weight += weight
                    valid_models += 1
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.0
            
            # Determine verification status
            verification_status = False
            if valid_models >= 3:  # At least 3 models must agree
                # Check if final score meets threshold and no model is below minimum
                below_minimum = any(
                    score is not None and score < self.minimum_similarity 
                    for score in results.values()
                )
                
                if final_score >= 0.55 and not below_minimum:
                    verification_status = True
            
            results["final_score"] = final_score
            results["verification_status"] = verification_status
            results["valid_models"] = valid_models
            
            print("\n[FACE VERIFICATION SUMMARY]")
            print(f"- Final Score: {final_score:.2f}")
            print(f"- Verification Status: {'✅ Verified' if verification_status else '❌ Not Verified'}")
            print(f"- Models Used: {valid_models}")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Face comparison failed: {e}")
            return {
                "error": str(e),
                "final_score": 0.0,
                "verification_status": False
            }

class BackImageDataExtractor:
    """Enhanced extractor for back image data including address and MRZ information"""
    
    def __init__(self):
        self.address_patterns = {
            "village": [r'VILLAGE[:\s]*([^.]+?)(?:\s+PARISH|$)', r'VIL[:\s]*([^.]+?)(?:\s+PAR|$)'],
            "parish": [r'PARISH[:\s]*([^.]+?)(?:\s+S\.COUNT|SUBCOUNTY|$)', r'PAR[:\s]*([^.]+?)(?:\s+S\.COUNT|SUBCOUNTY|$)'],
            "subcounty": [r'S\.COUNT[Y]*[:\s]*([^.]+?)(?:\s+COUNT|$)', r'SUBCOUNTY[:\s]*([^.]+?)(?:\s+COUNT|$)', r'SUB-COUNTY[:\s]*([^.]+?)(?:\s+COUNT|$)'],
            "county": [r'COUNT[Y]*[:\s]*([^.]+?)(?:\s+DISTRICT|$)'],
            "district": [r'DISTRICT[:\s]*([^.]+?)(?:\s|$)', r'DIST[:\s]*([^.]+?)(?:\s|$)']
        }
        
        self.mrz_patterns = {
            "document_type": r'^([A-Z]{1,2})',
            "country_code": r'^[A-Z]{1,2}([A-Z]{3})',
            "nin_pattern": r'(CM|CF)([A-Z0-9]{12})',
            "mrz_line": r'^[A-Z0-9<\s]{30,}$'
        }
    
    def clean_mrz_line(self, line: str) -> str:
        """Clean and normalize MRZ line by removing spaces and fixing common OCR errors"""
        # Remove all spaces
        cleaned = re.sub(r'\s+', '', line.upper())
        
        # Fix common OCR errors in MRZ
        cleaned = cleaned.replace('O', '0')  # O to 0
        cleaned = cleaned.replace('I', '1')  # I to 1
        cleaned = cleaned.replace('Z', '2')  # Z to 2
        cleaned = cleaned.replace('S', '5')  # S to 5
        cleaned = cleaned.replace('/', '')   # Remove slashes
        cleaned = cleaned.replace(':', '')   # Remove colons
        
        return cleaned
    
    def extract_address_info(self, lines: List[str]) -> Dict[str, str]:
        """Extract address information from back image text"""
        address_data = {
            "village": "",
            "parish": "",
            "subcounty": "",
            "county": "",
            "district": ""
        }
        
        # Combine all lines and clean
        combined_text = ' '.join(lines).upper()
        
        # Look for address patterns
        for field, patterns in self.address_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, combined_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value - remove unwanted characters
                    value = re.sub(r'[^A-Z\s-]', '', value).strip()
                    if value and len(value) > 1:
                        address_data[field] = value
                        break
        
        return address_data
    
    def extract_mrz_data(self, lines: List[str]) -> Dict[str, Any]:
        """Extract Machine Readable Zone (MRZ) data with improved detection"""
        mrz_data = {
            "mrz_lines": [],
            "document_type": "",
            "country_code": "",
            "document_number": "",
            "nin": "",
            "birth_date": "",
            "sex": "",
            "expiry_date": "",
            "personal_number": "",
            "check_digits": {}
        }
        
        # Process each line to find MRZ patterns
        for line in lines:
            line_clean = self.clean_mrz_line(line)
            
            # Check if this looks like an MRZ line (contains ID + country + alphanumeric)
            if (len(line_clean) >= 30 and 
                re.match(r'^[A-Z0-9<]+$', line_clean) and
                ('IDUGA' in line_clean or 'UGA' in line_clean)):
                
                mrz_data["mrz_lines"].append(line_clean)
                self._parse_mrz_line(line_clean, mrz_data)
        
        return mrz_data
    
    def _parse_mrz_line(self, mrz_line: str, mrz_data: Dict[str, Any]):
        """Parse individual MRZ line for specific data"""
        
        # Extract document type and country (IDUGA pattern)
        if mrz_line.startswith('IDUGA'):
            mrz_data["document_type"] = "ID"
            mrz_data["country_code"] = "UGA"
            
            # Extract document number (after IDUGA)
            doc_num_match = re.search(r'IDUGA(\d+)', mrz_line)
            if doc_num_match:
                mrz_data["document_number"] = doc_num_match.group(1)
        
        # Extract NIN (CM/CF pattern)
        nin_match = re.search(r'(CM|CF)([A-Z0-9]{12})', mrz_line)
        if nin_match:
            mrz_data["nin"] = nin_match.group(1) + nin_match.group(2)
        
        # Extract birth date (YYMMDD pattern)
        # Look for 6-digit sequences that could be dates
        date_matches = re.findall(r'(\d{6})', mrz_line)
        for date_str in date_matches:
            year = int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            
            # Validate date components
            if 1 <= month <= 12 and 1 <= day <= 31:
                # Convert 2-digit year to 4-digit
                full_year = 1900 + year if year >= 50 else 2000 + year
                
                # Check if this is a reasonable birth date (not in the future, not too old)
                from datetime import datetime
                current_year = datetime.now().year
                if 1930 <= full_year <= current_year - 10:  # Reasonable birth year range
                    mrz_data["birth_date"] = f"{full_year:04d}-{month:02d}-{day:02d}"
                    break
        
        # Extract sex (M/F after date)
        sex_match = re.search(r'\d{6}([MF])', mrz_line)
        if sex_match:
            mrz_data["sex"] = sex_match.group(1)
    
    def extract_biometric_info(self, lines: List[str]) -> Dict[str, str]:
        """Extract biometric information like fingerprint references"""
        biometric_data = {
            "fingerprint_type": "",
            "thumb_reference": ""
        }
        
        for line in lines:
            line_upper = line.upper().strip()
            if "THUMB" in line_upper:
                biometric_data["fingerprint_type"] = line.strip()
                if "RIGHT" in line_upper:
                    biometric_data["thumb_reference"] = "RIGHT_THUMB"
                elif "LEFT" in line_upper:
                    biometric_data["thumb_reference"] = "LEFT_THUMB"
        
        return biometric_data
    
    def extract_complete_back_data(self, lines: List[str]) -> Dict[str, Any]:
        """Extract all available data from back image"""
        
        address_data = self.extract_address_info(lines)
        mrz_data = self.extract_mrz_data(lines)
        biometric_data = self.extract_biometric_info(lines)
        
        complete_back_data = {
            "raw_text_lines": lines,
            "address_information": address_data,
            "machine_readable_zone": mrz_data,
            "biometric_information": biometric_data,
            "extraction_metadata": {
                "total_lines": len(lines),
                "address_fields_found": sum(1 for v in address_data.values() if v),
                "mrz_lines_found": len(mrz_data["mrz_lines"]),
                "nin_extracted_from_mrz": bool(mrz_data.get("nin")),
                "biometric_info_found": bool(biometric_data.get("fingerprint_type"))
            }
        }
        
        return complete_back_data

class IDCardProcessor:
    def __init__(self, 
                mongo_uri: str = "mongodb://localhost:27017/",
                db_name: str = "id_verification"):
        
        # Initialize MongoDB connection first
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db.id_cards
        
        # Initialize processors
        self.face_verifier = MultiModelFaceVerifier()
        self.back_extractor = BackImageDataExtractor()
        self.ocr_processor = MultiOCRProcessor()
        
        print("\n[ID CARD PROCESSOR INITIALIZED]")
        print(f"- MongoDB Connected: {self.client.server_info()['ok'] == 1.0}")
        
    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text using PaddleOCR and return all results with accuracy metrics"""
        print(f"\n[EXTRACTING TEXT FROM {image_path}]")
        
        ocr_result = self.ocr_processor.extract_consensus_text(image_path)
        
        engine_results = {
            'paddle': {
                "texts": ocr_result['all_results']['paddle'],
                "accuracy_score": 1.0,
                "text_count": len(ocr_result['all_results']['paddle'])
            }
        }
        
        return {
            "consensus_text": ocr_result['consensus_text'],
            "engine_results": engine_results,
            "best_engine": "paddle"
        }
    
    def extract_nin_from_combined_line(self, combined_line: str) -> Optional[str]:
        """Enhanced NIN extraction supporting both CM and CF prefixes"""
        # Clean the line first
        cleaned_line = re.sub(r'\s+', '', combined_line.upper())
        
        # Method 1: Direct CM/CF pattern matching
        nin_patterns = [
            r'(CM|CF)([A-Z0-9]{12})',
            r'(CM|CF)(\d{12})',
        ]
        
        for pattern in nin_patterns:
            matches = re.findall(pattern, cleaned_line)
            for match in matches:
                prefix, suffix = match
                if len(suffix) >= 12:
                    nin = prefix + suffix[:12]
                    # Validate NIN format
                    if len(nin) == 14 and nin[:2] in ['CM', 'CF'] and nin[2:].isalnum():
                        return nin
        
        # Method 2: Look for CM/CF positions and extract following characters
        for match in re.finditer(r'(CM|CF)', cleaned_line):
            prefix = match.group()
            start_pos = match.end()
            remaining_text = cleaned_line[start_pos:]
            
            # Extract exactly 12 alphanumeric characters
            alphanumeric_chars = []
            for char in remaining_text:
                if char.isalnum():
                    alphanumeric_chars.append(char)
                    if len(alphanumeric_chars) == 12:
                        break
            
            if len(alphanumeric_chars) == 12:
                nin = prefix + ''.join(alphanumeric_chars)
                return nin
        
        return None

    def extract_nin_from_back_image(self, back_image_path: str) -> tuple:
        """Extract NIN from the back of ID card using enhanced extractor"""
        try:
            print(f"\n[EXTRACTING NIN FROM BACK IMAGE: {back_image_path}]")
            ocr_result = self.extract_text_from_image(back_image_path)
            
            # Get all OCR results, not just consensus
            all_lines = []
            for engine, texts in ocr_result['engine_results'].items():
                all_lines.extend(texts['texts'])
            
            # Also include consensus
            all_lines.extend(ocr_result['consensus_text'])
            
            # Use enhanced back extractor
            back_data = self.back_extractor.extract_complete_back_data(all_lines)
            
            # First try to get NIN from MRZ data
            mrz_nin = back_data["machine_readable_zone"].get("nin")
            if mrz_nin and len(mrz_nin) == 14:
                print(f"- Found NIN in MRZ: {mrz_nin}")
                return mrz_nin, back_data, ocr_result
            
            # Fallback to line-by-line extraction from all OCR results
            for i, line in enumerate(all_lines):
                nin = self.extract_nin_from_combined_line(line)
                if nin and len(nin) == 14 and nin.startswith(('CM', 'CF')):
                    print(f"- Found NIN in line {i+1}: {nin}")
                    return nin, back_data, ocr_result
            
            # Try combined text
            combined_text = ''.join(all_lines)
            nin = self.extract_nin_from_combined_line(combined_text)
            if nin and len(nin) == 14 and nin.startswith(('CM', 'CF')):
                print(f"- Found NIN in combined text: {nin}")
                return nin, back_data, ocr_result
            
            print("- No valid NIN found in back image")
            return None, back_data, ocr_result
                
        except Exception as e:
            print(f"[ERROR] Failed to process back image: {e}")
            return None, {}, {}

    def extract_structured_data(self, ocr_result: Dict[str, Any], back_image_path: Optional[str] = None) -> tuple:
        """Structured data extraction with improved field parsing"""
        print("\n[EXTRACTING STRUCTURED DATA FOR UGANDA NATIONAL ID]")
        
        data = {
            "COUNTRY": "UGANDA",
            "DOCUMENT_TYPE": "NATIONAL ID CARD",
            "SURNAME": "",
            "GIVEN_NAME": "",
            "NATIONALITY": "",
            "SEX": "",
            "DATE_OF_BIRTH": "",
            "NIN": "",
            "CARD_NUMBER": "",
            "DATE_OF_EXPIRY": "",
            "HOLDER_SIGNATURE": "",
            "ADDRESS": {},
            "BIOMETRICS": {}
        }
        
        all_lines = []
        for engine, engine_data in ocr_result['engine_results'].items():
            all_lines.extend(engine_data['texts'])
        all_lines.extend(ocr_result['consensus_text'])
        all_lines = [line.strip() for line in all_lines if line.strip()]
        
        print("\n[RAW OCR LINES]")
        pprint(all_lines)
        
        # Smart field extraction - search for values in all lines
        self._extract_field_values(all_lines, data)
        
        # Special NIN extraction (since it might appear in multiple places)
        if not data["NIN"]:
            for line in all_lines:
                nin = self.extract_nin_from_combined_line(line)
                if nin and len(nin) == 14 and nin.startswith(('CM', 'CF')):
                    data["NIN"] = nin
                    print(f"- Extracted NIN from alternative pattern: {nin}")
                    break
        
        # Process back image if provided
        if back_image_path:
            print("\n[PROCESSING BACK IMAGE DATA]")
            back_nin, back_image_data, _ = self.extract_nin_from_back_image(back_image_path)
            
            if back_nin and not data["NIN"]:
                data["NIN"] = back_nin
                print(f"- Using NIN from back image: {back_nin}")
            
            if back_image_data:
                data["ADDRESS"] = back_image_data.get("address_information", {})
                data["BIOMETRICS"] = back_image_data.get("biometric_information", {})
        
        # Post-processing
        data = self._post_process_extracted_data(data)
        
        print("\n[FINAL EXTRACTED DATA]")
        pprint(data)
        
        return data, back_image_data if back_image_path else {}

    def _extract_field_values(self, all_lines: List[str], data: Dict) -> None:
        """Extract field values by searching through all lines for the best matches"""
        
        # Find all potential values for each field type
        potential_values = {
            "SURNAME": [],
            "GIVEN_NAME": [],
            "NATIONALITY": [],
            "SEX": [],
            "DATE_OF_BIRTH": [],
            "CARD_NUMBER": [],
            "DATE_OF_EXPIRY": [],
            "NIN": []
        }
        
        # Search through all lines for potential values
        for i, line in enumerate(all_lines):
            line_clean = line.strip()
            line_upper = line_clean.upper()
            
            # Skip field labels themselves
            if any(label in line_upper for label in ["SURNAME", "GIVEN NAME", "NATIONALITY", "SEX", 
                                                  "DATE OF BIRTH", "CARD NO.", "DATE OF EXPIRY", 
                                                  "HOLDERS SIGNATURE", "REPUBLIC", "NATIONAL ID"]):
                continue
            
            # Look for dates (DD.MM.YYYY format)
            if re.match(r'\d{2}\.\d{2}\.\d{4}', line_clean):
                day, month, year = line_clean.split('.')
                year_int = int(year)
                formatted_date = f"{year}-{month}-{day}"
                
                if year_int < 2025:  # Likely birth date
                    potential_values["DATE_OF_BIRTH"].append(formatted_date)
                else:  # Likely expiry date
                    potential_values["DATE_OF_EXPIRY"].append(formatted_date)
            
            # Look for 9-digit card numbers
            digits_only = re.sub(r'[^0-9]', '', line_clean)
            if len(digits_only) == 9:
                potential_values["CARD_NUMBER"].append(digits_only)
            
            # Look for NIN (CM/CF + 12 alphanumeric)
            nin = self.extract_nin_from_combined_line(line_clean)
            if nin and len(nin) == 14 and nin.startswith(('CM', 'CF')):
                potential_values["NIN"].append(nin)
            
            # Look for sex (single M or F)
            if line_clean.upper() in ["M", "F"]:
                potential_values["SEX"].append(line_clean.upper())
            
            # Look for nationality codes
            if line_upper in ["UGA", "UG", "UGANDA", "UGANDAN"]:
                potential_values["NATIONALITY"].append(line_upper)
            
            # Look for names (alphabetic strings with reasonable length)
            if (len(line_clean) >= 2 and 
                re.match(r'^[A-Z\- ]+$', line_upper) and 
                not re.search(r'(CM|CF|UGA|REPUBLIC|NATIONAL|\d)', line_upper)):
                
                # Determine if it's more likely a surname or given name
                # Simple heuristic: single words more likely surname, multiple words given name
                words = line_clean.split()
                if len(words) == 1 and len(line_clean) <= 15:
                    potential_values["SURNAME"].append(line_clean.upper())
                elif len(words) > 1:
                    potential_values["GIVEN_NAME"].append(line_clean.upper())
                else:
                    # Add to both if uncertain
                    potential_values["SURNAME"].append(line_clean.upper())
                    potential_values["GIVEN_NAME"].append(line_clean.upper())
        
        # Select best values for each field
        for field, candidates in potential_values.items():
            if candidates and not data.get(field):
                if field in ["SURNAME", "GIVEN_NAME"]:
                    # For names, prefer the first valid candidate
                    data[field] = candidates[0]
                elif field in ["DATE_OF_BIRTH", "DATE_OF_EXPIRY"]:
                    # For dates, prefer based on year logic
                    data[field] = candidates[0]
                elif field == "CARD_NUMBER":
                    # For card number, take first 9-digit sequence
                    data[field] = candidates[0]
                elif field == "NIN":
                    # For NIN, take first valid one
                    data[field] = candidates[0]
                elif field == "SEX":
                    # For sex, take first M or F
                    data[field] = candidates[0]
                elif field == "NATIONALITY":
                    # For nationality, prefer full forms
                    if "UGANDAN" in candidates:
                        data[field] = "UGANDAN"
                    elif "UGANDA" in candidates:
                        data[field] = "UGANDA"
                    else:
                        data[field] = candidates[0]
                
                if data[field]:
                    print(f"- Extracted {field}: {data[field]}")

    def _validate_field_value(self, field: str, value: str) -> bool:
        """Enhanced field validation with NIN gender detection"""
        if field == "SEX":
            # Validate against NIN prefix if available
            if hasattr(self, 'extracted_data') and self.extracted_data.get("NIN"):
                nin_prefix = self.extracted_data["NIN"][:2]
                if nin_prefix == "CF" and value.upper() != "F":
                    return False
                elif nin_prefix == "CM" and value.upper() != "M":
                    return False
            return value.upper() in ["M", "F"]
        
        elif field == "NIN":
            if not (len(value) == 14 and value[:2] in ("CM", "CF") and value[2:].isalnum()):
                return False
            # Validate gender consistency if SEX field exists
            if hasattr(self, 'extracted_data') and self.extracted_data.get("SEX"):
                expected_gender = "F" if value.startswith("CF") else "M"
                if self.extracted_data["SEX"].upper() != expected_gender:
                    return False
            return True
        
        elif field == "CARD_NUMBER":
            # Must be different from NIN
            if hasattr(self, 'extracted_data') and self.extracted_data.get("NIN"):
                if value == self.extracted_data["NIN"]:
                    return False
            return 6 <= len(value) <= 20 and value.isalnum()
        
        elif field == "NATIONALITY":
            # Convert UGA to UGANDAN
            if value.upper() in ["UGA", "UG", "UGANDA"]:
                return True
            return len(value) >= 3 and value.isalpha()
        
        elif field.endswith("DATE"):
            return self._validate_date(value)
        
        return True

    def _validate_date(self, date_str: str) -> bool:
        """Validate date format and ensure birth date is before 2025"""
        try:
            if '.' in date_str:
                # Handle DD.MM.YYYY format
                day, month, year = map(int, date_str.split('.'))
            elif '-' in date_str:
                # Handle YYYY-MM-DD format
                year, month, day = map(int, date_str.split('-'))
            else:
                return False
                
            # Basic date validation
            if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100):
                return False
            
            # Special validation for DATE_OF_BIRTH - must be before 2025
            # Note: We need to check if this is being called for birth date
            # This is a simplified check - in practice you might want to pass the field name
            if year >= 2025:
                # For birth dates, reject if year is 2025 or later
                return False
                
            return True
        except:
            return False

    def _post_process_extracted_data(self, data: Dict) -> Dict:
        """Clean and standardize extracted data with enhanced validation"""
        # 1. Determine gender from NIN if SEX field is empty
        if not data.get("SEX") and data.get("NIN"):
            if data["NIN"].startswith("CF"):
                data["SEX"] = "F"
            elif data["NIN"].startswith("CM"):
                data["SEX"] = "M"
        
        # 2. Enhanced card number validation - only 9 digits
        if data.get("CARD_NUMBER"):
            card_num = re.sub(r'[^0-9]', '', data["CARD_NUMBER"])  # Keep only digits
            if len(card_num) == 9:
                data["CARD_NUMBER"] = card_num
            else:
                data["CARD_NUMBER"] = ""  # Clear invalid card number
                print(f"- Invalid card number (not 9 digits): {data.get('CARD_NUMBER', 'N/A')}")
        
        # 3. Clean name fields
        for field in ["SURNAME", "GIVEN_NAME"]:
            if data.get(field):
                cleaned = re.sub(r'[^A-Z\- ]', '', data[field].upper())
                cleaned = ' '.join(cleaned.split())
                if len(cleaned) >= 2 and not re.search(r'(CM|CF|UGA|\d{4,})', cleaned):
                    data[field] = cleaned
                else:
                    data[field] = ""
        
        # 4. Standardize nationality (UGA should become UGANDAN)
        if data.get("NATIONALITY"):
            nat = data["NATIONALITY"].upper()
            if nat in ["UGA", "UG", "UGANDA"]:
                data["NATIONALITY"] = "UGANDAN"
            elif len(nat) >= 3 and nat.isalpha():
                data["NATIONALITY"] = nat.capitalize()
            else:
                data["NATIONALITY"] = ""
        
        # 5. Enhanced date validation and formatting
        date_fields = ["DATE_OF_BIRTH", "DATE_OF_EXPIRY"]
        for field in date_fields:
            if data.get(field):
                if isinstance(data[field], str):
                    # Handle DD.MM.YYYY format
                    if re.match(r'\d{2}\.\d{2}\.\d{4}', data[field]):
                        day, month, year = data[field].split('.')
                        year_int = int(year)
                        
                        # Special check for birth date - must be before 2025
                        if field == "DATE_OF_BIRTH" and year_int >= 2025:
                            print(f"- Invalid birth date (year {year_int} >= 2025): {data[field]}")
                            data[field] = ""
                            continue
                            
                        data[field] = f"{year}-{month}-{day}"
                    # Handle YYYY-MM-DD format
                    elif re.match(r'\d{4}-\d{2}-\d{2}', data[field]):
                        year_int = int(data[field].split('-')[0])
                        
                        # Special check for birth date - must be before 2025
                        if field == "DATE_OF_BIRTH" and year_int >= 2025:
                            print(f"- Invalid birth date (year {year_int} >= 2025): {data[field]}")
                            data[field] = ""
                            continue
                    # Handle other cases (like when OCR misreads date as text)
                    else:
                        data[field] = ""
        
        # 6. Remove empty nested dictionaries
        if not any(data.get("ADDRESS", {}).values()):
            data["ADDRESS"] = {}
        if not any(data.get("BIOMETRICS", {}).values()):
            data["BIOMETRICS"] = {}
        
        return data

    def process_extraction_only(self, id_card_path: str, back_image_path: Optional[str] = None) -> Dict[str, Any]:
        """Process just the data extraction without face verification"""
        try:
            print(f"\n[INFO] Starting extraction process for {id_card_path}")
            
            # Extract text from front image
            ocr_result = self.extract_text_from_image(id_card_path)
            
            # Extract structured data
            extracted_data, back_image_data = self.extract_structured_data(ocr_result, back_image_path)
            
            return {
                "status": "success",
                "extracted_data": extracted_data,
                "back_data": back_image_data or {},
                "raw_ocr_results": ocr_result,
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "extracted_data": {},
                "back_data": {},
                "raw_ocr_results": {},
                "processed_at": datetime.utcnow().isoformat()
            }

    def verify_face(self, id_card_path: str, selfie_path: str) -> Dict[str, Any]:
        """Verify face similarity using multiple models"""
        return self.face_verifier.compare_faces_all_models(id_card_path, selfie_path)
    
    def save_to_mongodb(self, extracted_data: Dict[str, Any], 
                    face_verification: Dict[str, Any],
                    id_card_path: str,
                    selfie_path: Optional[str] = None,
                    back_image_path: Optional[str] = None,
                    back_image_data: Optional[Dict[str, Any]] = None) -> str:
        """Save extracted data and verification results to MongoDB"""
        try:
            document = {
                "extracted_data": extracted_data,
                "face_verification": face_verification,
                "back_image_data": back_image_data or {},
                "metadata": {
                    "id_card_path": id_card_path,
                    "selfie_path": selfie_path,
                    "back_image_path": back_image_path,
                    "processed_at": datetime.utcnow(),
                    "processor_version": "4.0_MULTI_MODEL_FACE_VERIFICATION",
                    "has_back_image": bool(back_image_path),
                    "back_data_extracted": bool(back_image_data)
                },
                "verification_status": {
                    "face_match": face_verification.get("verification_status", False),
                    "similarity_score": face_verification.get("final_score", None),
                    "verified": face_verification.get("verification_status", False),
                    "model_details": {
                        k: v for k, v in face_verification.items() 
                        if k not in ["final_score", "verification_status"]
                    }
                }
            }
            
            # Validate document before insertion
            if not isinstance(document, dict):
                raise ValueError("Document must be a dictionary")
                
            # Insert document and return the ID
            print("\n[SAVING TO MONGODB]")
            pprint(document)
            result = self.collection.insert_one(document)
            
            if not result.inserted_id:
                raise ValueError("Failed to insert document - no ID returned")
                
            print(f"- Document saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except pymongo.errors.PyMongoError as e:
            print(f"[ERROR] MongoDB operation failed: {str(e)}")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to save document: {str(e)}")
            raise
    
    def get_verification_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve verification record by MongoDB document ID"""
        try:
            from bson import ObjectId
            document = self.collection.find_one({"_id": ObjectId(document_id)})
            if document:
                document["_id"] = str(document["_id"])  # Convert ObjectId to string
                return document
            return None
        except Exception as e:
            print(f"Error retrieving document: {e}")
            return None

    def search_by_nin(self, nin: str) -> list:
        """Search for records by NIN"""
        try:
            documents = list(self.collection.find({"extracted_data.NIN": nin}))
            for doc in documents:
                doc["_id"] = str(doc["_id"])
            return documents
        except Exception as e:
            print(f"Error searching by NIN: {e}")
            return []

    def process_complete_verification(self, id_card_path: str, 
                                    selfie_path: str,
                                    back_image_path: Optional[str] = None) -> Dict[str, Any]:
        """Complete processing pipeline with detailed console output"""
        
        print(f"\n{'='*50}")
        print(f"[STARTING COMPLETE VERIFICATION PROCESS]")
        print(f"- Front Image: {id_card_path}")
        print(f"- Selfie Image: {selfie_path}")
        if back_image_path:
            print(f"- Back Image: {back_image_path}")
        print(f"{'='*50}\n")
        
        # Step 1: Extract text from ID card front
        print("\n[STEP 1/4] EXTRACTING TEXT FROM ID CARD FRONT...")
        ocr_result = self.extract_text_from_image(id_card_path)
        
        # Step 2: Extract structured data
        print("\n[STEP 2/4] EXTRACTING STRUCTURED DATA...")
        extracted_data, back_image_data = self.extract_structured_data(ocr_result, back_image_path)
        
        print("\n[STRUCTURED DATA EXTRACTION RESULTS]")
        pprint(extracted_data)
        
        if back_image_path and back_image_data:
            print("\n[BACK IMAGE DATA EXTRACTION RESULTS]")
            pprint(back_image_data)
        
        # Step 3: Verify face similarity
        print("\n[STEP 3/4] VERIFYING FACE SIMILARITY...")
        face_verification = self.verify_face(id_card_path, selfie_path)
        
        print("\n[FACE VERIFICATION RESULTS]")
        pprint(face_verification)
        
        # Step 4: Save to MongoDB
        print("\n[STEP 4/4] SAVING TO DATABASE...")
        document_id = self.save_to_mongodb(
            extracted_data, face_verification, 
            id_card_path, selfie_path, back_image_path, back_image_data
        )
        
        print(f"\n[PROCESS COMPLETE] Document ID: {document_id}")
        
        return {
            "status": "success",
            "document_id": document_id,
            "extracted_data": extracted_data,
            "face_verification": face_verification,
            "back_image_data": back_image_data,
            "metadata": {
                "id_card_path": id_card_path,
                "selfie_path": selfie_path,
                "back_image_path": back_image_path,
                "processed_at": datetime.utcnow().isoformat()
            }
        }

    def close(self):
        """Clean up resources"""
        self.client.close()
        print("[INFO] MongoDB connection closed")


# Example usage with test images
if __name__ == "__main__":
    try:
        # Initialize processor
        print("\nInitializing ID Card Processor...")
        processor = IDCardProcessor()
        
        # Example paths (replace with actual paths)
        test_front = "test_id_front.jpg"
        test_back = "test_id_back.jpg"
        test_selfie = "test_selfie.jpg"
        
        # Run complete verification
        print("\nStarting complete verification process...")
        results = processor.process_complete_verification(
            id_card_path=test_front,
            selfie_path=test_selfie,
            back_image_path=test_back
        )
        
        print("\n\n[FINAL PROCESSING RESULTS]")
        pprint(results)
        
    except Exception as e:
        print(f"\n[PROCESSING ERROR] {str(e)}")
    finally:
        processor.close()