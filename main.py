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
import numpy as np

# Import our ID card processor
from id_card_processor import IDCardProcessor

# Initialize FastAPI app
app = FastAPI(
    title="ID Card Verification System API",
    description="API for identity verification with OCR and face matching",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for responses
class VerificationResponse(BaseModel):
    document_id: str
    extracted_data: dict
    face_verification: dict
    verification_summary: dict
    processed_at: str

class SearchResponse(BaseModel):
    results: list
    count: int

class StatsResponse(BaseModel):
    total_records: int
    verified_records: int
    face_matches: int
    verification_rate: float
    face_match_rate: float
    average_similarity_score: Optional[float]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: dict

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "ID Card Verification System API",
        "version": "2.1.0",
        "endpoints": {
            "verify": "/verify - Complete ID verification",
            "extract": "/extract - Extract ID card data", 
            "record": "/record/{document_id} - Get verification record",
            "search": "/search/{nin} - Search by NIN",
            "stats": "/stats - Get system statistics"
        }
    }

@app.post("/verify", response_model=VerificationResponse)
async def verify_identity(
    id_card: UploadFile = File(...),
    selfie: UploadFile = File(...),
    id_card_back: Optional[UploadFile] = File(None)
):
    """Complete identity verification workflow"""
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
                "verified": result["face_verification"].get("verification_status", False),
                "score": result["face_verification"].get("final_score", 0)
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

@app.post("/extract")
async def extract_id_data(
    id_card: UploadFile = File(...),
    include_back: bool = Form(False)
):
    """Extracts structured data from ID card image"""
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

@app.get("/record/{document_id}")
async def get_verification_record(document_id: str):
    """Retrieves a stored verification record"""
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

@app.get("/search/{nin}", response_model=SearchResponse)
async def search_by_nin(nin: str):
    """Searches verification records by NIN"""
    try:
        # Validate NIN format
        if not (nin.startswith('CM') or nin.startswith('CF')) or len(nin) != 14:
            raise HTTPException(
                status_code=400, 
                detail="Invalid NIN format. Should start with 'CM' or 'CF' and be 14 characters"
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

@app.get("/stats", response_model=StatsResponse)
async def get_system_statistics():
    """Returns system statistics"""
    try:
        processor = IDCardProcessor()
        stats = {
            "total_records": processor.collection.count_documents({}),
            "verified_records": processor.collection.count_documents({"face_verification.verification_status": True}),
            "face_matches": processor.collection.count_documents({"face_verification.verification_status": True}),
            "average_similarity_score": None
        }
        
        # Calculate averages
        if stats["total_records"] > 0:
            stats["verification_rate"] = stats["verified_records"] / stats["total_records"]
            stats["face_match_rate"] = stats["face_matches"] / stats["total_records"]
            
            # Get average similarity score
            pipeline = [
                {"$match": {"face_verification.final_score": {"$exists": True}}},
                {"$group": {"_id": None, "avgScore": {"$avg": "$face_verification.final_score"}}}
            ]
            avg_score = list(processor.collection.aggregate(pipeline))
            if avg_score:
                stats["average_similarity_score"] = avg_score[0].get("avgScore")
        
        return StatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
    finally:
        if 'processor' in locals():
            processor.close()

@app.post("/face-verify-only")
async def face_verify_only(
    id_card: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    """Performs standalone face verification"""
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
        face_verification = processor.face_verifier.compare_faces_all_models(id_card_path, selfie_path)
        
        return {
            "face_verification": face_verification,
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    try:
        # Test MongoDB connection
        processor = IDCardProcessor()
        mongo_status = "healthy"
        try:
            stats = processor.collection.count_documents({})
        except:
            mongo_status = "unreachable"
        
        # Test face verification capability
        face_api_status = "healthy"
        try:
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            if not processor.face_verifier.models["dlib"]["enabled"] and not processor.face_verifier.models["face_recognition"]["enabled"]:
                face_api_status = "partial (some models unavailable)"
        except:
            face_api_status = "unhealthy"
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)