import torch
import re
import base64
from io import BytesIO
from PIL import Image
import pymongo
from pymongo import MongoClient
from datetime import datetime
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pprint import pprint
from typing import Optional, Dict, Any, List
import json
import os
import cv2
import numpy as np
import dlib

class FaceVerifier:
    """Integrated face verification using dlib"""
    
    def __init__(self):
        """Initialize dlib models for face detection and recognition"""
        try:
            self.detector = dlib.get_frontal_face_detector()
            
            # Try to load the models - you'll need to download these files
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            face_model_path = "dlib_face_recognition_resnet_model_v1.dat"
            
            if os.path.exists(predictor_path) and os.path.exists(face_model_path):
                self.predictor = dlib.shape_predictor(predictor_path)
                self.face_rec_model = dlib.face_recognition_model_v1(face_model_path)
                self.models_loaded = True
                print("[INFO] Dlib face recognition models loaded successfully")
            else:
                print("[WARNING] Dlib model files not found. Face verification will be simulated.")
                print(f"[INFO] Missing files: {predictor_path}, {face_model_path}")
                print("[INFO] Download from: http://dlib.net/files/")
                self.models_loaded = False
                
        except Exception as e:
            print(f"[WARNING] Failed to initialize dlib models: {e}")
            self.models_loaded = False
    
    def read_image_from_path(self, image_path: str) -> np.ndarray:
        """Read image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            return image
        except Exception as e:
            raise ValueError(f"Error reading image: {e}")
    
    def find_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Find and extract face from image"""
        if not self.models_loaded:
            return image  # Return full image if models not loaded
            
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.detector(image_rgb)
            
            if not faces:
                return None
                
            face = faces[0]
            x1 = max(face.left(), 0)
            y1 = max(face.top(), 0)
            x2 = min(face.right(), image.shape[1])
            y2 = min(face.bottom(), image.shape[0])
            
            return image[y1:y2, x1:x2]
            
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            return None
    
    def get_face_descriptor(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get face descriptor for comparison"""
        if not self.models_loaded:
            # Return a dummy descriptor for simulation
            return np.random.rand(128)
            
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets = self.detector(image_rgb, 1)
            
            if len(dets) == 0:
                return None
                
            shape = self.predictor(image_rgb, dets[0])
            face_descriptor = self.face_rec_model.compute_face_descriptor(image_rgb, shape)
            return np.array(face_descriptor)
            
        except Exception as e:
            print(f"[ERROR] Face descriptor computation failed: {e}")
            return None
    
    def compare_faces(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.6) -> tuple:
        """Compare two face images"""
        try:
            desc1 = self.get_face_descriptor(img1)
            desc2 = self.get_face_descriptor(img2)
            
            if desc1 is None or desc2 is None:
                return None, None
                
            dist = np.linalg.norm(desc1 - desc2)
            match = dist < threshold
            
            return float(dist), bool(match)
            
        except Exception as e:
            print(f"[ERROR] Face comparison failed: {e}")
            return None, None
    
    def verify_faces(self, id_card_path: str, selfie_path: str) -> Dict[str, Any]:
        """Main face verification method"""
        try:
            # Read images
            img1 = self.read_image_from_path(id_card_path)
            img2 = self.read_image_from_path(selfie_path)
            
            # Find faces
            face1 = self.find_face(img1)
            face2 = self.find_face(img2)
            
            if face1 is None or face2 is None:
                return {
                    "error": "One or both faces not detected",
                    "match": False,
                    "distance": None,
                    "threshold": 0.6
                }
            
            # Compare faces
            distance, match = self.compare_faces(face1, face2)
            
            if distance is None:
                return {
                    "error": "Could not compute face descriptors",
                    "match": False,
                    "distance": None,
                    "threshold": 0.6
                }
            
            return {
                "match": match,
                "distance": round(distance, 4),
                "threshold": 0.6,
                "message": "Faces match" if match else "Faces do not match",
                "models_loaded": self.models_loaded
            }
            
        except Exception as e:
            return {
                "error": f"Face verification error: {str(e)}",
                "match": False,
                "distance": None,
                "threshold": 0.6
            }


class BackImageDataExtractor:
    """Enhanced extractor for back image data including address and MRZ information"""
    
    def __init__(self):
        self.address_patterns = {
            "village": [r'VILLAGE:\s*([^.]+)\.?', r'VIL:\s*([^.]+)\.?'],
            "parish": [r'PARISH:\s*([^.]+)\.?', r'PAR:\s*([^.]+)\.?'],
            "subcounty": [r'S\.COUNTY:\s*([^.]+)\.?', r'SUBCOUNTY:\s*([^.]+)\.?', r'SUB-COUNTY:\s*([^.]+)\.?'],
            "county": [r'COUNTY:\s*([^.]+)\.?'],
            "district": [r'DISTRICT:\s*([^.]+)\.?', r'DIST:\s*([^.]+)\.?']
        }
        
        self.mrz_patterns = {
            "document_type": r'^([A-Z]{1,2})',  # ID, P, etc.
            "country_code": r'^[A-Z]{1,2}([A-Z]{3})',  # UGA, etc.
            "nin_pattern": r'(CM|CF)([A-Z0-9]{12})',
            "mrz_line": r'^[A-Z0-9<]{30,44}$'  # Standard MRZ line length
        }
    
    def extract_address_info(self, lines: List[str]) -> Dict[str, str]:
        """Extract address information from back image text"""
        address_data = {
            "village": "",
            "parish": "",
            "subcounty": "",
            "county": "",
            "district": ""
        }
        
        combined_text = ' '.join(lines).upper()
        print(f"[DEBUG] Combined text for address extraction: {combined_text}")
        
        for field, patterns in self.address_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, combined_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    value = re.sub(r'[^\w\s-]', '', value).strip()
                    if value and len(value) > 1:  # Avoid single characters
                        address_data[field] = value
                        print(f"[DEBUG] Found {field}: {value}")
                        break
        
        return address_data
    
    def extract_mrz_data(self, lines: List[str]) -> Dict[str, Any]:
        """Extract Machine Readable Zone (MRZ) data"""
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
        
        # Find MRZ lines (lines with specific patterns)
        for line in lines:
            line_clean = line.strip()
            
            # Check if line looks like MRZ
            if len(line_clean) >= 30 and re.match(r'^[A-Z0-9<]+$', line_clean):
                mrz_data["mrz_lines"].append(line_clean)
                print(f"[DEBUG] Found MRZ line: {line_clean}")
                
                # Try to extract specific data from this line
                self._parse_mrz_line(line_clean, mrz_data)
        
        return mrz_data
    
    def _parse_mrz_line(self, mrz_line: str, mrz_data: Dict[str, Any]):
        """Parse individual MRZ line for specific data"""
        
        # Example MRZ line: IDUGA0235626084CM0303410NPP3J<0301017M3505090UGA250509<<<<<7
        
        # Extract document type and country (first 5 chars: IDUGA)
        if len(mrz_line) >= 5:
            doc_type_country = mrz_line[:5]
            if doc_type_country.startswith('ID'):
                mrz_data["document_type"] = "ID"
                mrz_data["country_code"] = doc_type_country[2:5]  # UGA
        
        # Extract NIN (CM/CF pattern)
        nin_match = re.search(r'(CM|CF)([A-Z0-9]{12})', mrz_line)
        if nin_match:
            mrz_data["nin"] = nin_match.group(1) + nin_match.group(2)
            print(f"[DEBUG] Extracted NIN from MRZ: {mrz_data['nin']}")
        
        # Extract dates and other info (this is complex and depends on MRZ format)
        # For Ugandan ID: IDUGA + document_number + NIN + check + birth_date + sex + expiry + country + filler + check
        
        # Try to extract birth date (6 digits: YYMMDD)
        date_pattern = r'(\d{6})'
        dates = re.findall(date_pattern, mrz_line)
        if dates:
            for date in dates:
                # Check if it could be a birth date (reasonable year range)
                year = int(date[:2])
                month = int(date[2:4])
                day = int(date[4:6])
                
                if 1 <= month <= 12 and 1 <= day <= 31:
                    # Convert 2-digit year to 4-digit (assume 1950-2049)
                    full_year = 1900 + year if year >= 50 else 2000 + year
                    mrz_data["birth_date"] = f"{full_year:04d}-{month:02d}-{day:02d}"
                    break
        
        # Extract sex (M/F)
        sex_match = re.search(r'(\d{6})([MF])', mrz_line)
        if sex_match:
            mrz_data["sex"] = sex_match.group(2)
    
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
        print(f"\n[INFO] Processing {len(lines)} lines from back image")
        
        # Extract different types of data
        address_data = self.extract_address_info(lines)
        mrz_data = self.extract_mrz_data(lines)
        biometric_data = self.extract_biometric_info(lines)
        
        # Combine all data
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
        
        print(f"[INFO] Back data extraction summary:")
        print(f"  - Address fields found: {complete_back_data['extraction_metadata']['address_fields_found']}")
        print(f"  - MRZ lines found: {complete_back_data['extraction_metadata']['mrz_lines_found']}")
        print(f"  - NIN from MRZ: {complete_back_data['extraction_metadata']['nin_extracted_from_mrz']}")
        print(f"  - Biometric info: {complete_back_data['extraction_metadata']['biometric_info_found']}")
        
        return complete_back_data


class IDCardProcessor:
    def __init__(self, 
                 mongo_uri: str = "mongodb://localhost:27017/",
                 db_name: str = "id_verification"):
        
        # MongoDB setup
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db.id_cards
        
        # Initialize integrated face verifier
        self.face_verifier = FaceVerifier()
        
        # Initialize back image data extractor
        self.back_extractor = BackImageDataExtractor()
        
        # Load the OCR model
        self.model = ocr_predictor(pretrained=True)
        
        # Define patterns for different field types - UPDATED FOR CM/CF
        self.patterns = {
            "date": [
                r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b',  # DD/MM/YYYY or DD.MM.YYYY
                r'\b\d{4}[./]\d{1,2}[./]\d{1,2}\b',  # YYYY/MM/DD or YYYY.MM.DD
                r'\b\d{1,2}[-]\d{1,2}[-]\d{4}\b',    # DD-MM-YYYY
            ],
            "nin": [
                r'\b(CM|CF)\d{12}\b',                # CM or CF followed by exactly 12 digits (14 total)
                r'\b(CM|CF)[A-Z0-9]{12}\b',         # CM or CF followed by 12 alphanumeric chars
            ],
            "card_no": [
                r'\bNPP\d+\b',                       # Starts with NPP
                r'\b[A-Z]{3}\d+\b',                  # Three letters followed by numbers
            ],
            "gender": [
                r'\b[MF]\b',                         # Single M or F
                r'\bMALE\b|\bFEMALE\b',             # Full words
            ],
            "nationality": [
                r'\bUGANDAN\b',
                r'\bRWANDAN\b',
                r'\bKENYAN\b',
                r'\bTANZANIAN\b',
            ]
        }
        
        # Common field indicators
        self.field_indicators = {
            "surname": ["surname", "family name", "last name"],
            "given_name": ["given name", "first name", "other names"],
            "nationality": ["nationality", "citizen"],
            "sex": ["sex", "gender"],
            "date_of_birth": ["date of birth", "dob", "born"],
            "nin": ["nin", "national id", "id number"],
            "card_no": ["card no", "card number", "document no"],
            "expiry": ["expiry", "expires", "valid until"],
            "country": ["country", "republic of"]
        }

    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        doc = DocumentFile.from_images(image_path)
        result = self.model(doc)
        extracted_data = result.export()
        
        lines = []
        for page in extracted_data['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    line_text = " ".join(word['value'] for word in line['words'])
                    if line_text.strip():
                        lines.append(line_text.strip())
        
        return lines

    def extract_nin_from_combined_line(self, combined_line):
        """
        Enhanced NIN extraction supporting both CM and CF prefixes
        NIN must be exactly 14 characters total (CM/CF + 12 characters)
        """
        print(f"[DEBUG] Processing combined line: {combined_line}")
        
        # Method 1: Look for CM or CF in MRZ format
        cm_cf_positions = []
        for match in re.finditer(r'(CM|CF)', combined_line):
            cm_cf_positions.append((match.start(), match.group()))
        
        print(f"[DEBUG] Found CM/CF at positions: {cm_cf_positions}")
        
        # Try each CM/CF position
        for pos, prefix in cm_cf_positions:
            remaining_text = combined_line[pos + 2:]  # Skip CM/CF (2 chars)
            
            # Extract exactly 12 alphanumeric characters
            alphanumeric_chars = []
            for char in remaining_text:
                if char.isalnum():  # A-Z, 0-9
                    alphanumeric_chars.append(char)
                    if len(alphanumeric_chars) == 12:
                        break
            
            if len(alphanumeric_chars) == 12:
                nin = prefix + ''.join(alphanumeric_chars)
                print(f"[DEBUG] Found NIN: {nin}")
                return nin
        
        # Method 2: Standard pattern matching
        nin_patterns = [
            r'(CM|CF)([A-Z0-9]{12})',
            r'(CM|CF)(\d{12})',
        ]
        
        for pattern in nin_patterns:
            matches = re.findall(pattern, combined_line)
            for match in matches:
                prefix, suffix = match
                if len(suffix) >= 12:
                    nin = prefix + suffix[:12]
                    print(f"[DEBUG] Pattern match NIN: {nin}")
                    return nin
        
        print(f"[DEBUG] No valid NIN found in line: {combined_line}")
        return None

    def extract_nin_from_back_image(self, back_image_path):
        """Extract NIN from the back of ID card using enhanced extractor"""
        print(f"[INFO] Processing back image: {back_image_path}")
        
        try:
            lines = self.extract_text_from_image(back_image_path)
            
            # Use enhanced back extractor
            back_data = self.back_extractor.extract_complete_back_data(lines)
            
            # First try to get NIN from MRZ data
            mrz_nin = back_data["machine_readable_zone"].get("nin")
            if mrz_nin and len(mrz_nin) == 14:
                print(f"[SUCCESS] Found NIN from MRZ: {mrz_nin}")
                return mrz_nin, back_data
            
            # Fallback to line-by-line extraction
            for line in lines:
                nin = self.extract_nin_from_combined_line(line)
                if nin and len(nin) == 14 and nin.startswith(('CM', 'CF')):
                    print(f"[SUCCESS] Found NIN in back image: {nin}")
                    return nin, back_data
            
            # Try combined text
            combined_text = ''.join(lines)
            nin = self.extract_nin_from_combined_line(combined_text)
            if nin and len(nin) == 14 and nin.startswith(('CM', 'CF')):
                print(f"[SUCCESS] Found NIN in combined text: {nin}")
                return nin, back_data
            
            return None, back_data
                
        except Exception as e:
            print(f"[ERROR] Failed to process back image: {e}")
            return None, {}

    def extract_structured_data(self, lines, back_image_path=None):
        """Extract structured data from OCR lines for Ugandan ID format"""
        data = {
            "COUNTRY": "",
            "SURNAME": "",
            "GIVEN NAME": "",
            "NATIONALITY": "",
            "SEX": "",
            "DATE OF BIRTH": "",
            "NIN": "",
            "CARD NO": "",
            "DATE OF EXPIRY": "",
            "HOLDER'S SIGNATURE": ""
        }
        
        back_image_data = {}
        
        # Extract from front image
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            if "REPUBLIC OF UGANDA" in line_upper:
                data["COUNTRY"] = "UGANDA"
            elif line_upper == "SURNAME" and i + 1 < len(lines):
                data["SURNAME"] = lines[i + 1].strip()
            elif line_upper == "GIVEN NAME" and i + 1 < len(lines):
                data["GIVEN NAME"] = lines[i + 1].strip()
            elif line_upper == "NATIONALITY":
                if i + 3 < len(lines):
                    data["NATIONALITY"] = lines[i + 3].strip()
                if i + 4 < len(lines):
                    data["SEX"] = lines[i + 4].strip()
                if i + 5 < len(lines):
                    data["DATE OF BIRTH"] = lines[i + 5].strip()
            elif line_upper == "NIN":
                if i + 2 < len(lines):
                    combined_line = lines[i + 2].strip()
                    extracted_nin = self.extract_nin_from_combined_line(combined_line)
                    if extracted_nin:
                        data["NIN"] = extracted_nin
                    
                    if i + 3 < len(lines):
                        data["CARD NO"] = lines[i + 3].strip()
            elif line_upper == "DATE OF EXPIRY" and i + 1 < len(lines):
                data["DATE OF EXPIRY"] = lines[i + 1].strip()
            elif line_upper in ["HOLDERS SIGNATURE", "HOLDER'S SIGNATURE"]:
                data["HOLDER'S SIGNATURE"] = "Present"
            
            # Also check if any line contains NIN pattern
            if not data["NIN"]:
                extracted_nin = self.extract_nin_from_combined_line(line)
                if extracted_nin:
                    data["NIN"] = extracted_nin
        
        # Process back image if provided
        if not data["NIN"] and back_image_path:
            print("\n[INFO] NIN not found in front image, processing back image...")
            back_nin, back_image_data = self.extract_nin_from_back_image(back_image_path)
            if back_nin:
                data["NIN"] = back_nin
        elif back_image_path:
            print("\n[INFO] Processing back image for additional data...")
            _, back_image_data = self.extract_nin_from_back_image(back_image_path)
        
        # Clean up nationality
        if data["NATIONALITY"] == "UGA":
            data["NATIONALITY"] = "UGANDAN"
            
        # Merge back image data with front data if MRZ contains additional info
        if back_image_data and "machine_readable_zone" in back_image_data:
            mrz = back_image_data["machine_readable_zone"]
            
            # Use MRZ data to fill missing fields
            if not data["SEX"] and mrz.get("sex"):
                data["SEX"] = mrz["sex"]
            
            if not data["DATE OF BIRTH"] and mrz.get("birth_date"):
                data["DATE OF BIRTH"] = mrz["birth_date"]
            
            if data["NATIONALITY"] == "UGANDAN" and mrz.get("country_code") == "UGA":
                # Confirm nationality from MRZ
                pass
        
        return data, back_image_data

    def verify_face(self, id_card_path: str, selfie_path: str) -> Dict[str, Any]:
        """Verify face similarity using integrated face verification"""
        return self.face_verifier.verify_faces(id_card_path, selfie_path)

    def save_to_mongodb(self, extracted_data: Dict[str, Any], 
                       face_verification: Dict[str, Any],
                       id_card_path: str,
                       selfie_path: Optional[str] = None,
                       back_image_path: Optional[str] = None,
                       back_image_data: Optional[Dict[str, Any]] = None) -> str:
        """Save extracted data and verification results to MongoDB"""
        
        document = {
            "extracted_data": extracted_data,
            "face_verification": face_verification,
            "back_image_data": back_image_data or {},
            "metadata": {
                "id_card_path": id_card_path,
                "selfie_path": selfie_path,
                "back_image_path": back_image_path,
                "processed_at": datetime.utcnow(),
                "processor_version": "3.0_ENHANCED_BACK_EXTRACTION",
                "has_back_image": bool(back_image_path),
                "back_data_extracted": bool(back_image_data)
            },
            "verification_status": {
                "face_match": face_verification.get("match", False) if "error" not in face_verification else False,
                "similarity_score": face_verification.get("distance", None) if "error" not in face_verification else None,
                "verified": face_verification.get("match", False) and "error" not in face_verification
            }
        }
        
        # Insert document and return the ID
        result = self.collection.insert_one(document)
        return str(result.inserted_id)

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
        """Complete processing pipeline: OCR + Face Verification + MongoDB Storage"""
        
        print(f"[INFO] Starting enhanced complete verification process...")
        print(f"[INFO] ID Card Front: {id_card_path}")
        print(f"[INFO] Selfie: {selfie_path}")
        if back_image_path:
            print(f"[INFO] ID Card Back: {back_image_path}")
        
        # Step 1: Extract text from ID card front
        print("\n[STEP 1] Extracting text from ID card front...")
        front_lines = self.extract_text_from_image(id_card_path)
        
        print("\n[INFO] Extracted Text from Front:")
        for i, line in enumerate(front_lines):
            print(f"{i+1:2d}: {line}")
        
        # Step 2: Extract structured data (including enhanced back image processing)
        print("\n[STEP 2] Extracting structured data with enhanced back processing...")
        extracted_data, back_image_data = self.extract_structured_data(front_lines, back_image_path)
        
        print("\n🧾 Extracted ID Card Data:")
        pprint(extracted_data)
        
        if back_image_data:
            print("\n🔙 Enhanced Back Image Data:")
            pprint(back_image_data)
        
        # Step 3: Verify face similarity
        print("\n[STEP 3] Verifying face similarity...")
        face_verification = self.verify_face(id_card_path, selfie_path)
        
        print("\n👤 Face Verification Results:")
        pprint(face_verification)
        
        # Step 4: Save to MongoDB with enhanced back data
        print("\n[STEP 4] Saving to MongoDB...")
        document_id = self.save_to_mongodb(
            extracted_data, face_verification, 
            id_card_path, selfie_path, back_image_path, back_image_data
        )
        
        print(f"✅ Saved to MongoDB with ID: {document_id}")
        
        # Return complete results
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


# Example usage
if __name__ == "__main__":
    try:
        # Initialize processor
        processor = IDCardProcessor()
        
        # Example paths (replace with actual paths)
        id_front = "/id_front.jpg"
        id_back = "path/to/id_back.jpg"
        selfie = "path/to/selfie.jpg"
        
        results = processor.process_complete_verification(
            id_card_path=id_front,
            selfie_path=selfie,
            back_image_path=id_back
        )
        
        print("\n🎉 Final Results:")
        pprint(results)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        processor.close()