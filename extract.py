import torch
import re
import requests
import base64
from io import BytesIO
from PIL import Image
import pymongo
from pymongo import MongoClient
from datetime import datetime
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pprint import pprint
from typing import Optional, Dict, Any
import json
import os

class IDCardProcessor:
    def __init__(self, 
                 mongo_uri: str = "mongodb://localhost:27017/",
                 db_name: str = "id_verification",
                 face_api_url: str = "http://localhost:8000/verify-face/"):
        
        # MongoDB setup
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db.id_cards
        
        # Face verification API URL
        self.face_api_url = face_api_url
        
        # Load the OCR model
        self.model = ocr_predictor(pretrained=True)
        
        # Define patterns for different field types
        self.patterns = {
            "date": [
                r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b',  # DD/MM/YYYY or DD.MM.YYYY
                r'\b\d{4}[./]\d{1,2}[./]\d{1,2}\b',  # YYYY/MM/DD or YYYY.MM.DD
                r'\b\d{1,2}[-]\d{1,2}[-]\d{4}\b',    # DD-MM-YYYY
            ],
            "nin": [
                r'\bCM\d{14}\b',                     # CM followed by exactly 14 digits (16 total)
                r'\bCM[A-Z0-9]{14}\b',              # CM followed by 14 alphanumeric chars
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
        """Extract NIN from combined line like 'CM0303410NPP3'"""
        nin_match = re.search(r'CM[A-Z0-9]+', combined_line)
        if nin_match:
            potential_nin = nin_match.group()
            if len(potential_nin) == 16:
                return potential_nin
            elif len(potential_nin) > 16:
                return potential_nin[:16]
            else:
                if 'NPP' in potential_nin:
                    base_nin = potential_nin.replace('NPP', '')
                    if len(base_nin) < 16 and base_nin.startswith('CM'):
                        after_cm = combined_line[2:]
                        digits_after_cm = re.findall(r'\d', after_cm)
                        
                        if len(digits_after_cm) >= 14:
                            full_nin = 'CM' + ''.join(digits_after_cm[:14])
                            return full_nin
                
                return potential_nin
        
        return None

    def extract_structured_data(self, lines):
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
        
        # Clean up nationality
        if data["NATIONALITY"] == "UGA":
            data["NATIONALITY"] = "UGANDAN"
            
        return data

    def verify_face(self, id_card_path: str, selfie_path: str) -> Dict[str, Any]:
        """Verify face similarity using the face verification API"""
        try:
            with open(id_card_path, 'rb') as id_file, open(selfie_path, 'rb') as selfie_file:
                files = {
                    'id_card': ('id_card.jpg', id_file, 'image/jpeg'),
                    'selfie': ('selfie.jpg', selfie_file, 'image/jpeg')
                }
                
                response = requests.post(self.face_api_url, files=files)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "error": f"Face verification failed with status {response.status_code}",
                        "details": response.text
                    }
        except Exception as e:
            return {
                "error": f"Face verification error: {str(e)}"
            }

    def save_to_mongodb(self, extracted_data: Dict[str, Any], 
                       face_verification: Dict[str, Any],
                       id_card_path: str,
                       selfie_path: Optional[str] = None) -> str:
        """Save extracted data and verification results to MongoDB"""
        
        document = {
            "extracted_data": extracted_data,
            "face_verification": face_verification,
            "metadata": {
                "id_card_path": id_card_path,
                "selfie_path": selfie_path,
                "processed_at": datetime.utcnow(),
                "processor_version": "1.0"
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
                                    selfie_path: str) -> Dict[str, Any]:
        """Complete processing pipeline: OCR + Face Verification + MongoDB Storage"""
        
        print(f"[INFO] Starting complete verification process...")
        print(f"[INFO] ID Card: {id_card_path}")
        print(f"[INFO] Selfie: {selfie_path}")
        
        # Step 1: Extract text from ID card
        print("\n[STEP 1] Extracting text from ID card...")
        lines = self.extract_text_from_image(id_card_path)
        
        print("\n[INFO] Extracted Text:")
        for i, line in enumerate(lines):
            print(f"{i+1:2d}: {line}")
        
        # Step 2: Extract structured data
        print("\n[STEP 2] Extracting structured data...")
        extracted_data = self.extract_structured_data(lines)
        
        print("\n🧾 Extracted ID Card Data:")
        pprint(extracted_data)
        
        # Step 3: Verify face similarity
        print("\n[STEP 3] Verifying face similarity...")
        face_verification = self.verify_face(id_card_path, selfie_path)
        
        print("\n👤 Face Verification Results:")
        pprint(face_verification)
        
        # Step 4: Save to MongoDB
        print("\n[STEP 4] Saving to MongoDB...")
        document_id = self.save_to_mongodb(extracted_data, face_verification, 
                                         id_card_path, selfie_path)
        
        print(f"✅ Saved to MongoDB with ID: {document_id}")
        
        # Step 5: Prepare final response
        final_result = {
            "document_id": document_id,
            "extracted_data": extracted_data,
            "face_verification": face_verification,
            "verification_summary": {
                "face_match": face_verification.get("match", False) if "error" not in face_verification else False,
                "similarity_distance": face_verification.get("distance", None) if "error" not in face_verification else None,
                "similarity_percentage": None,
                "threshold": face_verification.get("threshold", 0.6) if "error" not in face_verification else None,
                "overall_verified": False
            },
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Calculate similarity percentage (lower distance = higher similarity)
        if "distance" in face_verification and face_verification["distance"] is not None:
            # Convert distance to similarity percentage (0-100%)
            # Lower distance means higher similarity
            max_distance = 1.2  # Assume max meaningful distance
            distance = face_verification["distance"]
            similarity_percentage = max(0, (max_distance - distance) / max_distance * 100)
            final_result["verification_summary"]["similarity_percentage"] = round(similarity_percentage, 2)
            
            # Overall verification: face match AND reasonable similarity
            final_result["verification_summary"]["overall_verified"] = (
                face_verification.get("match", False) and 
                similarity_percentage > 50  # At least 50% similarity
            )
        
        print("\n🎯 Final Verification Summary:")
        pprint(final_result["verification_summary"])
        
        return final_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics from MongoDB"""
        try:
            total_records = self.collection.count_documents({})
            verified_records = self.collection.count_documents({"verification_status.verified": True})
            face_matches = self.collection.count_documents({"verification_status.face_match": True})
            
            # Get average similarity score
            pipeline = [
                {"$match": {"verification_status.similarity_score": {"$ne": None}}},
                {"$group": {"_id": None, "avg_similarity": {"$avg": "$verification_status.similarity_score"}}}
            ]
            avg_result = list(self.collection.aggregate(pipeline))
            avg_similarity = avg_result[0]["avg_similarity"] if avg_result else None
            
            return {
                "total_records": total_records,
                "verified_records": verified_records,
                "face_matches": face_matches,
                "verification_rate": round((verified_records / total_records * 100), 2) if total_records > 0 else 0,
                "face_match_rate": round((face_matches / total_records * 100), 2) if total_records > 0 else 0,
                "average_similarity_distance": round(avg_similarity, 4) if avg_similarity else None
            }
        except Exception as e:
            return {"error": f"Error getting statistics: {str(e)}"}


# Usage example and API wrapper
class IDVerificationAPI:
    def __init__(self):
        self.processor = IDCardProcessor()
    
    def verify_identity(self, id_card_path: str, selfie_path: str) -> Dict[str, Any]:
        """Main API method for identity verification"""
        return self.processor.process_complete_verification(id_card_path, selfie_path)
    
    def get_record(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get verification record by ID"""
        return self.processor.get_verification_by_id(document_id)
    
    def search_by_nin(self, nin: str) -> list:
        """Search records by NIN"""
        return self.processor.search_by_nin(nin)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.processor.get_statistics()


# Main execution
if __name__ == "__main__":
    # Initialize the processor
    processor = IDCardProcessor()
    
    # Example usage
    id_card_path = 'test/National ID Specimen_0.jpg'
    selfie_path = 'test/selfie.jpg'  # Replace with actual selfie path
    
    try:
        # Process complete verification
        result = processor.process_complete_verification(id_card_path, selfie_path)
        
        print("\n" + "="*60)
        print("🎉 COMPLETE VERIFICATION RESULTS")
        print("="*60)
        
        print(f"\n📄 Document ID: {result['document_id']}")
        print(f"👤 Name: {result['extracted_data'].get('GIVEN NAME', '')} {result['extracted_data'].get('SURNAME', '')}")
        print(f"🆔 NIN: {result['extracted_data'].get('NIN', 'N/A')}")
        print(f"🏳️ Nationality: {result['extracted_data'].get('NATIONALITY', 'N/A')}")
        print(f"👥 Sex: {result['extracted_data'].get('SEX', 'N/A')}")
        print(f"🎂 Date of Birth: {result['extracted_data'].get('DATE OF BIRTH', 'N/A')}")
        
        verification = result['verification_summary']
        print(f"\n✅ Face Match: {'YES' if verification['face_match'] else 'NO'}")
        if verification['similarity_percentage']:
            print(f"📊 Similarity: {verification['similarity_percentage']}%")
        if verification['similarity_distance']:
            print(f"📏 Distance: {verification['similarity_distance']}")
        print(f"🎯 Overall Verified: {'YES' if verification['overall_verified'] else 'NO'}")
        
        # Show statistics
        stats = processor.get_statistics()
        print(f"\n📈 System Statistics:")
        print(f"   Total Records: {stats.get('total_records', 0)}")
        print(f"   Verification Rate: {stats.get('verification_rate', 0)}%")
        print(f"   Face Match Rate: {stats.get('face_match_rate', 0)}%")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        
    # Example of searching by NIN
    if 'result' in locals() and 'NIN' in result['extracted_data']:
        nin = result['extracted_data']['NIN']
        print(f"\n🔍 Searching for records with NIN: {nin}")
        records = processor.search_by_nin(nin)
        print(f"Found {len(records)} record(s)")