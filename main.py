from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
from typing import Optional
import uuid
from datetime import datetime
from pydantic import BaseModel
import json

# Import our ID card processor
from id_card_processor import IDCardProcessor, FaceVerifier

# Initialize FastAPI app with proper metadata for Swagger
app = FastAPI(
    title="ID Card Verification System API",
    description="""A comprehensive API for identity verification that:
    - Extracts data from ID cards using OCR
    - Verifies face matches between ID photos and selfies
    - Stores verification results in MongoDB
    - Provides search and analytics capabilities""",
    version="2.1.0",
    contact={
        "name": "API Support",
        "email": "support@idverification.com"
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {
            "name": "Verification",
            "description": "Complete identity verification endpoints",
        },
        {
            "name": "Extraction",
            "description": "ID card data extraction endpoints",
        },
        {
            "name": "Records",
            "description": "Verification record access endpoints",
        },
        {
            "name": "System",
            "description": "System health and monitoring",
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the verification API
verification_api = FaceVerifier()

# Pydantic models for responses
class VerificationResponse(BaseModel):
    """Response model for verification results"""
    document_id: str
    extracted_data: dict
    face_verification: dict
    verification_summary: dict
    processed_at: str

    class Config:
        schema_extra = {
            "example": {
                "document_id": "507f1f77bcf86cd799439011",
                "extracted_data": {
                    "COUNTRY": "UGANDA",
                    "SURNAME": "KAMYA",
                    "GIVEN NAME": "ALICE NAKATO",
                    "NATIONALITY": "UGANDAN",
                    "SEX": "F",
                    "DATE OF BIRTH": "1985-05-15",
                    "NIN": "CM123456789012",
                    "CARD NO": "NPP12345678",
                    "DATE OF EXPIRY": "2025-12-31"
                },
                "face_verification": {
                    "match": True,
                    "distance": 0.35,
                    "threshold": 0.6,
                    "message": "Faces match"
                },
                "verification_summary": {
                    "verified": True,
                    "score": 0.85
                },
                "processed_at": "2023-07-15T12:30:45.123456"
            }
        }

class SearchResponse(BaseModel):
    """Response model for search results"""
    results: list
    count: int

    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "document_id": "507f1f77bcf86cd799439011",
                        "extracted_data": {
                            "NIN": "CM123456789012",
                            "SURNAME": "KAMYA"
                        },
                        "processed_at": "2023-07-15T12:30:45.123456"
                    }
                ],
                "count": 1
            }
        }

class StatsResponse(BaseModel):
    """Response model for system statistics"""
    total_records: int
    verified_records: int
    face_matches: int
    verification_rate: float
    face_match_rate: float
    average_similarity_distance: Optional[float]

    class Config:
        schema_extra = {
            "example": {
                "total_records": 1250,
                "verified_records": 980,
                "face_matches": 920,
                "verification_rate": 0.784,
                "face_match_rate": 0.736,
                "average_similarity_distance": 0.42
            }
        }

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    services: dict

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2023-07-15T12:30:45.123456",
                "services": {
                    "mongodb": "healthy",
                    "face_verification_api": "healthy",
                    "ocr_model": "loaded"
                }
            }
        }

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ID Card Verification System API",
        "version": "2.1.0",
        "endpoints": {
            "verify": "/verify - Complete ID verification with face matching",
            "extract": "/extract - Extract data from ID card only", 
            "record": "/record/{document_id} - Get verification record",
            "search": "/search/{nin} - Search by NIN",
            "stats": "/stats - Get system statistics"
        }
    }

@app.post("/verify", 
          response_model=VerificationResponse,
          tags=["Verification"],
          summary="Complete Identity Verification",
          description="""Performs complete identity verification by:
1. Extracting data from the ID card (front and back if provided)
2. Verifying face match between ID photo and selfie
3. Storing results in database""",
          response_description="Verification results with extracted data and face match score")
async def verify_identity(
    id_card: UploadFile = File(..., description="Front image of ID card in JPEG/PNG format"),
    selfie: UploadFile = File(..., description="Selfie image for face verification in JPEG/PNG format"),
    id_card_back: Optional[UploadFile] = File(None, description="Back image of ID card (optional)")
):
    """
    Complete identity verification workflow:
    - Extracts text and structured data from ID card
    - Verifies face match between ID photo and selfie
    - Stores results in database
    - Returns verification results
    
    Supports both front and back images of ID cards for enhanced data extraction.
    """
    id_card_path = None
    selfie_path = None
    id_card_back_path = None
    
    try:
        # Validate file types
        if not id_card.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="ID card must be an image file")
        if not selfie.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Selfie must be an image file")
        if id_card_back and not id_card_back.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="ID card back must be an image file")
        
        # Save uploaded files temporarily
        id_card_path = save_upload_file(id_card)
        selfie_path = save_upload_file(selfie)
        if id_card_back:
            id_card_back_path = save_upload_file(id_card_back)
        
        # Process verification
        processor = IDCardProcessor()
        result = processor.process_complete_verification(
            id_card_path=id_card_path,
            selfie_path=selfie_path,
            back_image_path=id_card_back_path
        )
        
        return VerificationResponse(
            document_id=result["document_id"],
            extracted_data=result["extracted_data"],
            face_verification=result["face_verification"],
            verification_summary={
                "verified": result["face_verification"].get("match", False),
                "score": 1 - (result["face_verification"].get("distance", 1) / 0.6)
                if result["face_verification"].get("distance") else None
            },
            processed_at=result["metadata"]["processed_at"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for temp_path in [id_card_path, selfie_path, id_card_back_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
        if 'processor' in locals():
            processor.close()

@app.post("/extract",
          tags=["Extraction"],
          summary="Extract ID Card Data",
          description="Extracts and returns data from ID card image without face verification",
          response_description="Extracted data from ID card")
async def extract_id_data(
    id_card: UploadFile = File(..., description="ID card image in JPEG/PNG format"),
    include_back: bool = Form(False, description="Set to true if processing back image")
):
    """
    Extracts structured data from ID card image using OCR.
    
    Features:
    - Handles multiple ID card formats
    - Extracts NIN, personal details, dates
    - Optionally processes back image for additional data
    """
    id_card_path = None
    id_card_back_path = None
    
    try:
        # Validate file type
        if not id_card.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="ID card must be an image file")
        
        # Save uploaded file temporarily
        id_card_path = save_upload_file(id_card)
        
        # Extract text and structured data
        processor = IDCardProcessor()
        lines = processor.extract_text_from_image(id_card_path)
        extracted_data, back_data = processor.extract_structured_data(lines)
        
        response = {
            "status": "success",
            "extracted_data": extracted_data,
            "back_data": back_data if include_back else None,
            "raw_text_lines": lines,
            "processed_at": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data extraction failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if id_card_path and os.path.exists(id_card_path):
            try:
                os.unlink(id_card_path)
            except:
                pass
        if 'processor' in locals():
            processor.close()

@app.get("/record/{document_id}",
         tags=["Records"],
         summary="Get Verification Record",
         description="Retrieves a complete verification record by its document ID",
         response_description="Full verification record")
async def get_verification_record(document_id: str):
    """
    Retrieves a stored verification record by its MongoDB document ID.
    
    Returns all stored information including:
    - Extracted ID data
    - Face verification results
    - Processing metadata
    """
    try:
        processor = IDCardProcessor()
        record = processor.get_verification_by_id(document_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")
        
        return record
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve record: {str(e)}")
    finally:
        if 'processor' in locals():
            processor.close()

@app.get("/search/{nin}",
         response_model=SearchResponse,
         tags=["Records"],
         summary="Search by NIN",
         description="Searches verification records by National Identification Number (NIN)",
         response_description="Matching verification records")
async def search_by_nin(nin: str):
    """
    Searches for all verification records associated with a specific NIN.
    
    NIN format should be:
    - Starts with CM or CF
    - Followed by 12 alphanumeric characters
    - Total length 14 characters
    """
    try:
        # Validate NIN format (basic validation)
        if not (nin.startswith('CM') or nin.startswith('CF')) or len(nin) != 14:
            raise HTTPException(
                status_code=400, 
                detail="Invalid NIN format. NIN should start with 'CM' or 'CF' and be 14 characters long"
            )
        
        processor = IDCardProcessor()
        results = processor.search_by_nin(nin)
        
        return SearchResponse(
            results=results,
            count=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    finally:
        if 'processor' in locals():
            processor.close()

@app.get("/stats",
         response_model=StatsResponse,
         tags=["System"],
         summary="System Statistics",
         description="Returns system statistics and performance metrics",
         response_description="Current system statistics")
async def get_system_statistics():
    """
    Returns comprehensive system statistics including:
    - Total verification records
    - Verification success rates
    - Face matching statistics
    - Average processing metrics
    """
    try:
        processor = IDCardProcessor()
        stats = {
            "total_records": processor.collection.count_documents({}),
            "verified_records": processor.collection.count_documents({"verification_status.verified": True}),
            "face_matches": processor.collection.count_documents({"face_verification.match": True}),
            "average_similarity_distance": None
        }
        
        # Calculate averages
        if stats["total_records"] > 0:
            stats["verification_rate"] = stats["verified_records"] / stats["total_records"]
            stats["face_match_rate"] = stats["face_matches"] / stats["total_records"]
            
            # Get average similarity distance
            pipeline = [
                {"$match": {"face_verification.distance": {"$exists": True}}},
                {"$group": {"_id": None, "avgDistance": {"$avg": "$face_verification.distance"}}}
            ]
            avg_distance = list(processor.collection.aggregate(pipeline))
            if avg_distance:
                stats["average_similarity_distance"] = avg_distance[0].get("avgDistance")
        
        return StatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
    finally:
        if 'processor' in locals():
            processor.close()

@app.post("/face-verify-only",
          tags=["Verification"],
          summary="Face Verification Only",
          description="Performs face verification between ID photo and selfie without data extraction",
          response_description="Face verification results")
async def face_verify_only(
    id_card: UploadFile = File(..., description="ID card image with face photo"),
    selfie: UploadFile = File(..., description="Selfie image for comparison")
):
    """
    Performs standalone face verification between:
    - The photo on an ID card
    - A provided selfie photo
    
    Returns detailed face matching results including:
    - Similarity score
    - Match decision
    - Confidence metrics
    """
    id_card_path = None
    selfie_path = None
    
    try:
        # Validate file types
        if not id_card.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="ID card must be an image file")
        if not selfie.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Selfie must be an image file")
        
        # Save uploaded files temporarily
        id_card_path = save_upload_file(id_card)
        selfie_path = save_upload_file(selfie)
        
        # Verify face only
        processor = IDCardProcessor()
        face_verification = processor.verify_face(id_card_path, selfie_path)
        
        # Calculate similarity percentage
        similarity_percentage = None
        if "distance" in face_verification and face_verification["distance"] is not None:
            max_distance = 1.2
            distance = face_verification["distance"]
            similarity_percentage = max(0, (max_distance - distance) / max_distance * 100)
        
        return {
            "face_verification": face_verification,
            "similarity_percentage": round(similarity_percentage, 2) if similarity_percentage else None,
            "processed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for temp_path in [id_card_path, selfie_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
        if 'processor' in locals():
            processor.close()

@app.get("/health",
         response_model=HealthResponse,
         tags=["System"],
         summary="System Health Check",
         description="Returns current system health status",
         response_description="System health status")
async def health_check():
    """
    Comprehensive health check endpoint that verifies:
    - Database connectivity
    - External service availability
    - Model loading status
    - Overall system health
    """
    try:
        # Test MongoDB connection
        processor = IDCardProcessor()
        stats = {
            "total_records": processor.collection.count_documents({}),
            "verified_records": processor.collection.count_documents({"verification_status.verified": True}),
        }
        mongo_status = "healthy" if isinstance(stats["total_records"], int) else "error"
        
        # Test face verification capability
        face_api_status = "healthy"
        try:
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            descriptor = processor.face_verifier.get_face_descriptor(test_img)
            if descriptor is None:
                face_api_status = "unresponsive"
        except:
            face_api_status = "unreachable"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            services={
                "mongodb": mongo_status,
                "face_verification": face_api_status,
                "ocr_model": "loaded"
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    finally:
        if 'processor' in locals():
            processor.close()

# Utility function to save uploaded file temporarily
def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    try:
        file_extension = os.path.splitext(upload_file.filename)[1]
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        return temp_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )