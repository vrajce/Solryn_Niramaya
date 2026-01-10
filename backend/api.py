from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import base64
import io
from datetime import datetime
from collections import deque
import uuid

# --- FASTAPI APP ---
app = FastAPI(
    title="Dayflow Multi-Specialist CDSS API",
    description="Clinical Decision Support System for Bone Fracture, TB, and Pneumonia Detection",
    version="3.0"
)

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL MODELS ---
router_model = None
bone_model = None
lung_model = None

# --- IN-MEMORY HISTORY (max 100 entries, FIFO) ---
scan_history = deque(maxlen=100)

# --- LOAD ALL MODELS ---
def load_all_models():
    global router_model, bone_model, lung_model
    models_loaded = {"router": False, "bone": False, "lung": False}
    
    try:
        router_model = tf.keras.models.load_model('router_model.h5')
        models_loaded["router"] = True
        print("✅ Router model loaded")
    except Exception as e:
        print(f"⚠️ Router model failed: {e}")
    
    try:
        bone_model = YOLO('bone_model.pt')
        models_loaded["bone"] = True
        print("✅ Bone model loaded")
    except Exception as e:
        print(f"⚠️ Bone model failed: {e}")
    
    try:
        lung_model = YOLO('lung_model.pt')
        models_loaded["lung"] = True
        print("✅ Lung model loaded")
    except Exception as e:
        print(f"⚠️ Lung model failed: {e}")
    
    return models_loaded

@app.on_event("startup")
async def startup_event():
    models_status = load_all_models()
    print(f"📊 Models Status: {models_status}")

# ==================== REQUEST/RESPONSE MODELS ====================

class SymptomsInput(BaseModel):
    # Common
    pain: bool = False
    fever: bool = False
    # Bone-specific
    deformity: bool = False
    immobility: bool = False
    trauma: bool = False
    swelling: bool = False
    # Lung/TB-specific
    cough: bool = False
    blood_sputum: bool = False
    night_sweats: bool = False
    weight_loss: bool = False

class AnalysisResponse(BaseModel):
    success: bool
    scan_type: str  # "Bone", "Chest", "Invalid"
    scan_type_confidence: float
    ai_confidence: float
    ai_location: str
    ai_method: str
    detected_condition: Optional[str] = None
    clinical_boost: float
    clinical_reasons: List[str]
    final_confidence: float
    risk_class: str
    status_message: str
    visual_analysis_text: str
    clinical_context_text: str
    annotated_image_base64: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    version: str


# ==================== CORE LOGIC FUNCTIONS ====================

def route_image(image: Image.Image) -> tuple:
    """
    Route the image to appropriate specialist model.
    Returns: (scan_type, confidence)
    """
    if router_model is None:
        return "Unknown", 0.0
    
    # Preprocess for router (224x224)
    img_array = np.array(image.resize((224, 224))) / 255.0
    predictions = router_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
    
    route_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))
    
    scan_map = {0: "Bone", 1: "Chest", 2: "Invalid"}
    scan_type = scan_map.get(route_idx, "Invalid")
    
    return scan_type, confidence


def get_spatial_description(box, w, h, scan_type: str) -> str:
    """
    Generate human-readable location description.
    """
    cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
    
    if scan_type == "Bone":
        loc_x = "Right" if cx > w/2 else "Left"
        loc_y = "Distal/Bottom" if cy > h/2 else "Proximal/Top"
        return f"{loc_y}-{loc_x} Quadrant"
    else:
        # Lung Zone Logic (Medical Standard)
        if cy < h/3:
            zone = "Upper Lung Field"
        elif cy < 2*h/3:
            zone = "Middle Lung Field"
        else:
            zone = "Lower Lung Field"
        side = "Right" if cx > w/2 else "Left"
        return f"{side} {zone}"


def analyze_bone(image: Image.Image) -> tuple:
    """
    Bone fracture detection with 2-phase analysis.
    Returns: (annotated_image, confidence, location, method, label)
    """
    if bone_model is None:
        raise HTTPException(status_code=500, detail="Bone model not loaded")
    
    w, h = image.size
    
    # PHASE 1: Standard Full Scan
    results = bone_model.predict(image, conf=0.15, augment=True, verbose=False)
    
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        conf = float(box.conf[0])
        loc = get_spatial_description(box, w, h, "Bone")
        label = bone_model.names[int(box.cls[0])] if hasattr(bone_model, 'names') else "Fracture"
        plot = results[0].plot(line_width=3)
        return Image.fromarray(plot[..., ::-1]), conf, loc, "Standard Scan", label

    # PHASE 2: Deep Zoom (4 overlapping quadrants)
    crops = [
        ("Top-Left",  (0, 0, w//2 + 50, h//2 + 50)),
        ("Top-Right", (w//2 - 50, 0, w, h//2 + 50)),
        ("Bot-Left",  (0, h//2 - 50, w//2 + 50, h)),
        ("Bot-Right", (w//2 - 50, h//2 - 50, w, h))
    ]
    
    for loc_name, region in crops:
        crop = image.crop(region)
        res = bone_model.predict(crop, conf=0.15, verbose=False)
        
        if len(res[0].boxes) > 0:
            conf = float(res[0].boxes.conf[0])
            label = bone_model.names[int(res[0].boxes.cls[0])] if hasattr(bone_model, 'names') else "Fracture"
            plot = res[0].plot(line_width=3)
            return Image.fromarray(plot[..., ::-1]), conf, f"{loc_name} Quadrant", "Deep Zoom", label
    
    # PHASE 3: Clean
    return image, 0.0, "None", "Clean", "Normal"


def analyze_lung(image: Image.Image) -> tuple:
    """
    Lung/TB/Pneumonia detection.
    Returns: (annotated_image, confidence, location, method, label)
    """
    if lung_model is None:
        raise HTTPException(status_code=500, detail="Lung model not loaded")
    
    w, h = image.size
    
    # Run prediction
    results = lung_model.predict(image, conf=0.15, augment=True, verbose=False)
    
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        conf = float(box.conf[0])
        loc = get_spatial_description(box, w, h, "Chest")
        
        # Get class label (TB, Pneumonia, etc.)
        class_idx = int(box.cls[0])
        label = lung_model.names[class_idx] if hasattr(lung_model, 'names') else "Abnormality"
        
        plot = results[0].plot(line_width=3)
        return Image.fromarray(plot[..., ::-1]), conf, loc, "Standard Scan", label
    
    return image, 0.0, "None", "Clean", "Normal"


def calculate_clinical_boost(symptoms: dict, scan_type: str) -> tuple:
    """
    Calculate clinical boost based on symptoms and scan type.
    Returns: (boost_score, reasons_list, context_text)
    """
    score = 0.0
    reasons = []
    
    if scan_type == "Bone":
        # Orthopedic symptoms
        if symptoms.get('deformity'):
            score += 0.40
            reasons.append("Visible Bone Deformity (+40%)")
        if symptoms.get('immobility'):
            score += 0.30
            reasons.append("Functional Loss/Immobility (+30%)")
        if symptoms.get('trauma'):
            score += 0.15
            reasons.append("Recent Trauma History (+15%)")
        if symptoms.get('swelling'):
            score += 0.10
            reasons.append("Soft Tissue Swelling (+10%)")
        if symptoms.get('pain'):
            score += 0.05
            reasons.append("Localized Pain (+5%)")
        
        context = "Orthopedic risk elevated due to: " + ", ".join([r.split(" (+")[0] for r in reasons]) + "." if reasons else "No significant orthopedic risk factors reported."
    
    else:  # Chest/Lung
        # Pulmonary symptoms
        if symptoms.get('blood_sputum'):
            score += 0.45
            reasons.append("Hemoptysis/Blood in Sputum (+45%)")
        if symptoms.get('weight_loss'):
            score += 0.20
            reasons.append("Unexplained Weight Loss (+20%)")
        if symptoms.get('cough'):
            score += 0.15
            reasons.append("Chronic Cough >2 weeks (+15%)")
        if symptoms.get('night_sweats'):
            score += 0.10
            reasons.append("Night Sweats (+10%)")
        if symptoms.get('fever'):
            score += 0.10
            reasons.append("High Fever (+10%)")
        
        context = "Pulmonary risk elevated due to: " + ", ".join([r.split(" (+")[0] for r in reasons]) + "." if reasons else "No significant respiratory symptoms reported."
    
    # Cap at 50%
    final_boost = min(score, 0.50)
    return final_boost, reasons, context


def calculate_risk_assessment(ai_conf: float, boost_score: float, scan_type: str) -> tuple:
    """
    Fusion logic combining AI confidence with clinical symptoms.
    Uses weighted formula: Final = (AI_Confidence * 0.6) + (Clinical_Boost * 0.4)
    Returns: (final_confidence, risk_class, status_message)
    """
    # Weighted combination: 60% AI model + 40% clinical symptoms
    # Normalize boost_score to 0-1 range (it's already capped at 0.5, so multiply by 2)
    normalized_boost = min(boost_score * 2, 1.0)  # Convert 0-0.5 range to 0-1
    final_conf = (ai_conf * 0.6) + (normalized_boost * 0.4)
    
    # Risk classification
    if final_conf > 0.60:
        risk_class = "HIGH RISK"
        if scan_type == "Bone":
            status_msg = "Fracture Confirmed (AI + Clinical)"
        else:
            status_msg = "Pulmonary Abnormality Confirmed (AI + Clinical)"
    elif final_conf > 0.30:
        risk_class = "MODERATE RISK"
        if scan_type == "Bone":
            status_msg = "Suspected Fracture (Review Required)"
        else:
            status_msg = "Suspected Lung Pathology (Review Required)"
    elif boost_score > 0.30:
        risk_class = "CLINICAL WARNING"
        if scan_type == "Bone":
            status_msg = "Scan Negative, but Symptoms Critical (Occult Injury?)"
        else:
            status_msg = "Scan Negative, but Symptoms Critical (Consider CT)"
    else:
        risk_class = "LOW RISK"
        status_msg = "No Significant Findings"
    
    return final_conf, risk_class, status_msg


def generate_visual_reasoning(ai_conf: float, ai_loc: str, ai_method: str, label: str, scan_type: str) -> str:
    """
    Generate human-readable visual analysis text.
    """
    if ai_conf > 0:
        if scan_type == "Bone":
            return f"AI localized a {label} pattern in the {ai_loc} using {ai_method} logic."
        else:
            return f"AI detected {label} in the {ai_loc} using {ai_method} analysis."
    else:
        if scan_type == "Bone":
            return "Radiological analysis is negative. No fracture patterns detected in Standard or Deep Zoom scans."
        else:
            return "Lung fields appear clear. No significant opacities or lesions detected."


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==================== API ENDPOINTS ====================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "router": router_model is not None,
            "bone": bone_model is not None,
            "lung": lung_model is not None
        },
        version="3.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return await health_check()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_scan(
    file: UploadFile = File(...),
    # Common symptoms
    pain: bool = Form(False),
    fever: bool = Form(False),
    # Bone symptoms
    deformity: bool = Form(False),
    immobility: bool = Form(False),
    trauma: bool = Form(False),
    swelling: bool = Form(False),
    # Lung symptoms
    cough: bool = Form(False),
    blood_sputum: bool = Form(False),
    night_sweats: bool = Form(False),
    weight_loss: bool = Form(False),
    # Options
    include_image: bool = Form(True)
):
    """
    Universal analysis endpoint - automatically routes to appropriate specialist.
    
    **Workflow:**
    1. Router classifies image as Bone/Chest/Invalid
    2. Appropriate specialist model analyzes the scan
    3. Clinical symptoms boost the risk score
    4. Returns comprehensive diagnosis
    """
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an image (JPG, PNG, JPEG).")
    
    try:
        # Load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Step 1: ROUTE THE IMAGE
        scan_type, route_confidence = route_image(image)
        
        # Handle Invalid Images
        if scan_type == "Invalid":
            return AnalysisResponse(
                success=False,
                scan_type="Invalid",
                scan_type_confidence=route_confidence,
                ai_confidence=0.0,
                ai_location="N/A",
                ai_method="N/A",
                detected_condition=None,
                clinical_boost=0.0,
                clinical_reasons=[],
                final_confidence=0.0,
                risk_class="INVALID",
                status_message="Invalid input detected. Please upload a valid medical X-ray scan.",
                visual_analysis_text="The uploaded image does not appear to be a valid medical scan.",
                clinical_context_text="Analysis cannot proceed without a valid X-ray image.",
                annotated_image_base64=None
            )
        
        # Build symptoms dict
        symptoms = {
            'pain': pain,
            'fever': fever,
            'deformity': deformity,
            'immobility': immobility,
            'trauma': trauma,
            'swelling': swelling,
            'cough': cough,
            'blood_sputum': blood_sputum,
            'night_sweats': night_sweats,
            'weight_loss': weight_loss
        }
        
        # Step 2: RUN SPECIALIST ANALYSIS
        if scan_type == "Bone":
            ai_img, ai_conf, ai_loc, ai_method, label = analyze_bone(image)
        else:  # Chest
            ai_img, ai_conf, ai_loc, ai_method, label = analyze_lung(image)
        
        # Step 3: CLINICAL BOOST
        boost_score, clinical_reasons, clinical_context = calculate_clinical_boost(symptoms, scan_type)
        
        # Step 4: FUSION LOGIC
        final_conf, risk_class, status_msg = calculate_risk_assessment(ai_conf, boost_score, scan_type)
        
        # Step 5: GENERATE REASONING
        visual_text = generate_visual_reasoning(ai_conf, ai_loc, ai_method, label, scan_type)
        
        # Build response dict for history
        result_dict = {
            "scan_type": scan_type,
            "scan_type_confidence": round(route_confidence, 4),
            "ai_confidence": round(ai_conf, 4),
            "ai_location": ai_loc,
            "ai_method": ai_method,
            "detected_condition": label if ai_conf > 0 else None,
            "clinical_boost": round(boost_score, 4),
            "clinical_reasons": clinical_reasons,
            "final_confidence": round(final_conf, 4),
            "risk_class": risk_class,
            "status_message": status_msg,
            "visual_analysis_text": visual_text,
            "clinical_context_text": clinical_context
        }
        
        # Add to history (non-blocking)
        add_to_history(result_dict, file.filename or "Scan")
        
        # Build response
        return AnalysisResponse(
            success=True,
            scan_type=scan_type,
            scan_type_confidence=round(route_confidence, 4),
            ai_confidence=round(ai_conf, 4),
            ai_location=ai_loc,
            ai_method=ai_method,
            detected_condition=label if ai_conf > 0 else None,
            clinical_boost=round(boost_score, 4),
            clinical_reasons=clinical_reasons,
            final_confidence=round(final_conf, 4),
            risk_class=risk_class,
            status_message=status_msg,
            visual_analysis_text=visual_text,
            clinical_context_text=clinical_context,
            annotated_image_base64=image_to_base64(ai_img) if include_image else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/bone", response_model=AnalysisResponse)
async def analyze_bone_only(
    file: UploadFile = File(...),
    deformity: bool = Form(False),
    immobility: bool = Form(False),
    trauma: bool = Form(False),
    swelling: bool = Form(False),
    pain: bool = Form(False),
    include_image: bool = Form(True)
):
    """
    Direct bone fracture analysis endpoint (bypasses router).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        symptoms = {
            'deformity': deformity,
            'immobility': immobility,
            'trauma': trauma,
            'swelling': swelling,
            'pain': pain
        }
        
        ai_img, ai_conf, ai_loc, ai_method, label = analyze_bone(image)
        boost_score, clinical_reasons, clinical_context = calculate_clinical_boost(symptoms, "Bone")
        final_conf, risk_class, status_msg = calculate_risk_assessment(ai_conf, boost_score, "Bone")
        visual_text = generate_visual_reasoning(ai_conf, ai_loc, ai_method, label, "Bone")
        
        return AnalysisResponse(
            success=True,
            scan_type="Bone",
            scan_type_confidence=1.0,
            ai_confidence=round(ai_conf, 4),
            ai_location=ai_loc,
            ai_method=ai_method,
            detected_condition=label if ai_conf > 0 else None,
            clinical_boost=round(boost_score, 4),
            clinical_reasons=clinical_reasons,
            final_confidence=round(final_conf, 4),
            risk_class=risk_class,
            status_message=status_msg,
            visual_analysis_text=visual_text,
            clinical_context_text=clinical_context,
            annotated_image_base64=image_to_base64(ai_img) if include_image else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bone analysis failed: {str(e)}")


@app.post("/analyze/lung", response_model=AnalysisResponse)
async def analyze_lung_only(
    file: UploadFile = File(...),
    cough: bool = Form(False),
    blood_sputum: bool = Form(False),
    night_sweats: bool = Form(False),
    weight_loss: bool = Form(False),
    fever: bool = Form(False),
    include_image: bool = Form(True)
):
    """
    Direct lung/TB/Pneumonia analysis endpoint (bypasses router).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        symptoms = {
            'cough': cough,
            'blood_sputum': blood_sputum,
            'night_sweats': night_sweats,
            'weight_loss': weight_loss,
            'fever': fever
        }
        
        ai_img, ai_conf, ai_loc, ai_method, label = analyze_lung(image)
        boost_score, clinical_reasons, clinical_context = calculate_clinical_boost(symptoms, "Chest")
        final_conf, risk_class, status_msg = calculate_risk_assessment(ai_conf, boost_score, "Chest")
        visual_text = generate_visual_reasoning(ai_conf, ai_loc, ai_method, label, "Chest")
        
        return AnalysisResponse(
            success=True,
            scan_type="Chest",
            scan_type_confidence=1.0,
            ai_confidence=round(ai_conf, 4),
            ai_location=ai_loc,
            ai_method=ai_method,
            detected_condition=label if ai_conf > 0 else None,
            clinical_boost=round(boost_score, 4),
            clinical_reasons=clinical_reasons,
            final_confidence=round(final_conf, 4),
            risk_class=risk_class,
            status_message=status_msg,
            visual_analysis_text=visual_text,
            clinical_context_text=clinical_context,
            annotated_image_base64=image_to_base64(ai_img) if include_image else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lung analysis failed: {str(e)}")


@app.post("/symptoms/calculate")
async def calculate_symptoms_endpoint(symptoms: SymptomsInput, scan_type: str = "Bone"):
    """
    Calculate clinical boost from symptoms only.
    """
    symptoms_dict = symptoms.dict()
    boost_score, reasons, context = calculate_clinical_boost(symptoms_dict, scan_type)
    
    return {
        "scan_type": scan_type,
        "boost_score": round(boost_score, 4),
        "boost_percentage": f"+{boost_score*100:.0f}%",
        "factors_count": len(reasons),
        "reasons": reasons,
        "clinical_context": context
    }


@app.post("/route")
async def route_only(file: UploadFile = File(...)):
    """
    Route an image to determine scan type without full analysis.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        scan_type, confidence = route_image(image)
        
        return {
            "scan_type": scan_type,
            "confidence": round(confidence, 4),
            "is_valid": scan_type != "Invalid"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")


# ==================== HISTORY ENDPOINTS ====================

@app.get("/history")
async def get_history(limit: int = 50):
    """
    Get scan history. Returns most recent scans first.
    """
    return list(scan_history)[:limit]


@app.get("/history/{scan_id}")
async def get_history_item(scan_id: str):
    """
    Get a specific scan from history.
    """
    for item in scan_history:
        if item["id"] == scan_id:
            return item
    raise HTTPException(status_code=404, detail="Scan not found")


@app.delete("/history/{scan_id}")
async def delete_history_item(scan_id: str):
    """
    Delete a scan from history.
    """
    global scan_history
    scan_history = deque([h for h in scan_history if h["id"] != scan_id], maxlen=100)
    return {"success": True, "message": "Scan deleted"}


@app.delete("/history")
async def clear_history():
    """
    Clear all scan history.
    """
    scan_history.clear()
    return {"success": True, "message": "History cleared"}


def add_to_history(result: dict, scan_name: str = "Unknown"):
    """
    Add a scan result to history (internal function).
    """
    history_entry = {
        "id": str(uuid.uuid4())[:8],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M"),
        "patientId": f"PX-{uuid.uuid4().hex[:4].upper()}",
        "scanType": result.get("scan_type", "Unknown"),
        "result": result.get("risk_class", "Unknown"),
        "confidence": result.get("final_confidence", 0),
        "severity": result.get("risk_class", "Unknown"),
        "detectedCondition": result.get("detected_condition"),
        "aiConfidence": result.get("ai_confidence", 0),
        "clinicalBoost": result.get("clinical_boost", 0),
        "location": result.get("ai_location", "N/A"),
        "method": result.get("ai_method", "N/A"),
        "statusMessage": result.get("status_message", ""),
        "visualAnalysis": result.get("visual_analysis_text", ""),
        "clinicalContext": result.get("clinical_context_text", ""),
        "clinicalReasons": result.get("clinical_reasons", []),
        "findings": [{"id": 1, "label": result.get("detected_condition", "N/A"), "region": result.get("ai_location", "N/A")}] if result.get("ai_confidence", 0) > 0 else [],
        "status": "Reviewed" if result.get("final_confidence", 0) > 0.6 else "Pending Review"
    }
    scan_history.appendleft(history_entry)
    return history_entry["id"]


# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
