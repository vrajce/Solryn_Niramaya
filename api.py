from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import cv2
import base64
import io
from datetime import datetime
from collections import deque
import uuid

# --- FASTAPI APP ---
app = FastAPI(
    title="Niramaya Multi-Specialist CDSS API",
    description="Clinical Decision Support System for Bone Fracture, TB, and Pneumonia Detection",
    version="4.0"
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
lung_model = None  # TB detection model
pneumonia_classifier = None  # Pneumonia classification model (best_model_auc.keras)
pneumonia_segmentation = None  # Pneumonia segmentation model (pneumonia_unet_v2.h5)

# --- IN-MEMORY HISTORY (max 100 entries, FIFO) ---
scan_history = deque(maxlen=100)

# --- LOAD ALL MODELS ---
def load_all_models():
    global router_model, bone_model, lung_model, pneumonia_classifier, pneumonia_segmentation
    models_loaded = {"router": False, "bone": False, "lung": False, "pneumonia_classifier": False, "pneumonia_segmentation": False}
    
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
        print("✅ Lung/TB model loaded")
    except Exception as e:
        print(f"⚠️ Lung/TB model failed: {e}")
    
    try:
        pneumonia_classifier = tf.keras.models.load_model('best_model_auc.keras')
        models_loaded["pneumonia_classifier"] = True
        print("✅ Pneumonia classifier model loaded")
    except Exception as e:
        print(f"⚠️ Pneumonia classifier model failed: {e}")
    
    try:
        # Load U-Net model with compile=False to avoid custom loss/metric issues
        pneumonia_segmentation = tf.keras.models.load_model('pneumonia_unet_v2.h5', compile=False)
        models_loaded["pneumonia_segmentation"] = True
        print("✅ Pneumonia segmentation model loaded")
    except Exception as e:
        print(f"⚠️ Pneumonia segmentation model failed: {e}")
    
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
    chest_condition: Optional[str] = None  # "TB", "Pneumonia", or None for Bone/Invalid
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


class ChestRouteResponse(BaseModel):
    """Response when chest X-ray is detected, prompting user to select TB or Pneumonia"""
    success: bool
    scan_type: str
    scan_type_confidence: float
    requires_selection: bool
    options: List[str]
    message: str


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
    Lung/TB detection using YOLO model.
    Returns: (annotated_image, confidence, location, method, label)
    """
    if lung_model is None:
        raise HTTPException(status_code=500, detail="Lung/TB model not loaded")
    
    w, h = image.size
    
    # Run prediction
    results = lung_model.predict(image, conf=0.15, augment=True, verbose=False)
    
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        conf = float(box.conf[0])
        loc = get_spatial_description(box, w, h, "Chest")
        
        # Get class label (TB, etc.)
        class_idx = int(box.cls[0])
        label = lung_model.names[class_idx] if hasattr(lung_model, 'names') else "TB"
        
        plot = results[0].plot(line_width=3)
        return Image.fromarray(plot[..., ::-1]), conf, loc, "Standard Scan", label
    
    return image, 0.0, "None", "Clean", "Normal"


def analyze_pneumonia(image: Image.Image) -> tuple:
    """
    Pneumonia detection using classification + U-Net segmentation.
    Step 1: Classify if pneumonia exists (best_model_auc.keras)
    Step 2: If positive, generate contour map (pneumonia_unet_v2.h5)
    Returns: (annotated_image, confidence, location, method, label)
    """
    if pneumonia_classifier is None:
        raise HTTPException(status_code=500, detail="Pneumonia classifier model not loaded")
    
    w, h = image.size
    
    # Step 1: Classification - Check if pneumonia exists
    # Model expects 512x512 RGB images
    img_array = np.array(image.resize((512, 512))) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    
    # Predict pneumonia presence
    classification_pred = pneumonia_classifier.predict(np.expand_dims(img_array, axis=0), verbose=0)
    
    # Handle different output formats (sigmoid vs softmax)
    if classification_pred.shape[-1] == 1:
        # Binary sigmoid output
        pneumonia_conf = float(classification_pred[0][0])
    else:
        # Softmax output (assume index 1 is pneumonia)
        pneumonia_conf = float(classification_pred[0][1]) if classification_pred.shape[-1] > 1 else float(classification_pred[0][0])
    
    # If pneumonia detected with reasonable confidence, generate segmentation
    if pneumonia_conf > 0.3 and pneumonia_segmentation is not None:
        try:
            # Step 2: Segmentation - Generate contour map
            # Preprocess for U-Net (typically 256x256 or model-specific)
            seg_size = 256  # Common U-Net input size
            img_for_seg = np.array(image.resize((seg_size, seg_size))) / 255.0
            
            if len(img_for_seg.shape) == 2:
                img_for_seg = np.expand_dims(img_for_seg, axis=-1)
            elif img_for_seg.shape[-1] == 3:
                # Convert to grayscale for U-Net if needed
                img_for_seg = np.mean(img_for_seg, axis=-1, keepdims=True)
            
            # Get segmentation mask
            seg_pred = pneumonia_segmentation.predict(np.expand_dims(img_for_seg, axis=0), verbose=0)
            seg_mask = (seg_pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
            
            # Resize mask back to original size
            seg_mask_resized = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Find contours
            contours, _ = cv2.findContours(seg_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create annotated image with contours
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            if len(contours) > 0:
                # Draw contours in red
                cv2.drawContours(img_cv, contours, -1, (0, 0, 255), 2)
                
                # Add filled overlay for affected regions
                overlay = img_cv.copy()
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)
                img_cv = cv2.addWeighted(img_cv, 0.7, overlay, 0.3, 0)
                
                # Calculate location based on contour centroid
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Determine lung zone
                    if cy < h/3:
                        zone = "Upper Lung Field"
                    elif cy < 2*h/3:
                        zone = "Middle Lung Field"
                    else:
                        zone = "Lower Lung Field"
                    side = "Right" if cx > w/2 else "Left"
                    location = f"{side} {zone}"
                else:
                    location = "Lung Fields"
                
                # Add label on image
                cv2.putText(img_cv, f"Pneumonia: {pneumonia_conf*100:.1f}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                annotated_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                return annotated_img, pneumonia_conf, location, "Classification + U-Net Segmentation", "Pneumonia"
            else:
                # No contours found but classification positive
                location = "Diffuse/Bilateral"
                cv2.putText(img_cv, f"Pneumonia: {pneumonia_conf*100:.1f}%", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                annotated_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                return annotated_img, pneumonia_conf, location, "Classification (Segmentation Negative)", "Pneumonia"
                
        except Exception as seg_error:
            print(f"⚠️ Segmentation failed: {seg_error}")
            # Fall back to classification only
            return image, pneumonia_conf, "Lung Fields", "Classification Only", "Pneumonia"
    
    elif pneumonia_conf > 0.3:
        # Pneumonia detected but no segmentation model
        return image, pneumonia_conf, "Lung Fields", "Classification Only", "Pneumonia"
    
    # No pneumonia detected
    return image, pneumonia_conf, "None", "Clean", "Normal"


def calculate_clinical_boost(symptoms: dict, scan_type: str, chest_condition: str = None) -> tuple:
    """
    Calculate clinical boost based on symptoms and scan type.
    For Chest scans, chest_condition specifies 'TB' or 'Pneumonia'.
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
    
    elif chest_condition == "TB":
        # TB-specific symptoms
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
            reasons.append("Persistent Low-Grade Fever (+10%)")
        
        context = "TB risk elevated due to: " + ", ".join([r.split(" (+")[0] for r in reasons]) + "." if reasons else "No significant TB symptoms reported."
    
    elif chest_condition == "Pneumonia":
        # Pneumonia-specific symptoms
        if symptoms.get('high_fever'):
            score += 0.35
            reasons.append("High Fever >38.5°C (+35%)")
        if symptoms.get('productive_cough'):
            score += 0.25
            reasons.append("Productive Cough with Sputum (+25%)")
        if symptoms.get('shortness_breath'):
            score += 0.20
            reasons.append("Shortness of Breath/Dyspnea (+20%)")
        if symptoms.get('chest_pain'):
            score += 0.10
            reasons.append("Pleuritic Chest Pain (+10%)")
        if symptoms.get('rapid_breathing'):
            score += 0.10
            reasons.append("Rapid/Shallow Breathing (+10%)")
        # Also consider general symptoms
        if symptoms.get('fever') and not symptoms.get('high_fever'):
            score += 0.15
            reasons.append("Fever (+15%)")
        if symptoms.get('cough') and not symptoms.get('productive_cough'):
            score += 0.10
            reasons.append("Cough (+10%)")
        
        context = "Pneumonia risk elevated due to: " + ", ".join([r.split(" (+")[0] for r in reasons]) + "." if reasons else "No significant pneumonia symptoms reported."
    
    else:
        # Generic Chest/Lung symptoms (fallback)
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


def calculate_risk_assessment(ai_conf: float, boost_score: float, scan_type: str, chest_condition: str = None) -> tuple:
    """
    Fusion logic combining AI confidence with clinical symptoms.
    Uses weighted formula:
    - Bone Fracture: Final = (AI_Confidence * 0.9) + (Clinical_Boost * 0.1)
    - TB/Pneumonia: Final = (AI_Confidence * 0.6) + (Clinical_Boost * 0.4)
    Returns: (final_confidence, risk_class, status_message)
    """
    # Normalize boost_score to 0-1 range (it's already capped at 0.5, so multiply by 2)
    normalized_boost = min(boost_score * 2, 1.0)  # Convert 0-0.5 range to 0-1
    
    # Apply different weights based on scan type
    # For bone fractures, AI confidence is much more reliable than symptoms
    if scan_type == "Bone":
        # 90% AI model + 10% clinical symptoms (fractures are visually definitive)
        final_conf = (ai_conf * 0.9) + (normalized_boost * 0.1)
        print(f"🦴 BONE CALCULATION: ai_conf={ai_conf:.4f} * 0.9 + boost={normalized_boost:.4f} * 0.1 = {final_conf:.4f}")
    else:
        # 60% AI model + 40% clinical symptoms (TB/Pneumonia benefit more from clinical context)
        final_conf = (ai_conf * 0.6) + (normalized_boost * 0.4)
        print(f"🫁 CHEST CALCULATION: ai_conf={ai_conf:.4f} * 0.6 + boost={normalized_boost:.4f} * 0.4 = {final_conf:.4f}")
    
    # Determine condition label for messages
    if scan_type == "Bone":
        condition_name = "Fracture"
    elif chest_condition == "TB":
        condition_name = "Tuberculosis"
    elif chest_condition == "Pneumonia":
        condition_name = "Pneumonia"
    else:
        condition_name = "Pulmonary Abnormality"
    
    # Risk classification
    if final_conf > 0.60:
        risk_class = "HIGH RISK"
        if scan_type == "Bone":
            status_msg = "Fracture Confirmed (AI + Clinical)"
        else:
            status_msg = f"{condition_name} Confirmed (AI + Clinical)"
    elif final_conf > 0.30:
        risk_class = "MODERATE RISK"
        if scan_type == "Bone":
            status_msg = "Suspected Fracture (Review Required)"
        else:
            status_msg = f"Suspected {condition_name} (Review Required)"
    elif boost_score > 0.30:
        risk_class = "CLINICAL WARNING"
        if scan_type == "Bone":
            status_msg = "Scan Negative, but Symptoms Critical (Occult Injury?)"
        elif chest_condition == "TB":
            status_msg = "Scan Negative, but TB Symptoms Critical (Consider Sputum Test)"
        elif chest_condition == "Pneumonia":
            status_msg = "Scan Negative, but Symptoms Critical (Consider CT/Lab Tests)"
        else:
            status_msg = "Scan Negative, but Symptoms Critical (Consider CT)"
    else:
        risk_class = "LOW RISK"
        status_msg = "No Significant Findings"
    
    return final_conf, risk_class, status_msg


def analyze_bone_location(location_str: str) -> tuple:
    """
    Map YOLO detection to anatomical regions and provide medical context.
    Returns: (anatomical_region, fracture_type, clinical_notes)
    """
    location_lower = location_str.lower()
    
    # Anatomical mapping
    if "arm" in location_lower or "humer" in location_lower:
        region = "Upper Extremity (Humeral)"
        fracture_type = "Humeral Shaft Fracture"
        notes = "Risk of radial nerve injury. Requires neurovascular assessment."
    elif "forearm" in location_lower or "radius" in location_lower or "ulna" in location_lower:
        region = "Forearm (Radius/Ulna)"
        fracture_type = "Distal Forearm Fracture"
        notes = "Common in fall-on-outstretched-hand (FOOSH) mechanism. Check for wrist stability."
    elif "leg" in location_lower or "tibia" in location_lower or "fibula" in location_lower:
        region = "Lower Extremity (Tibial/Fibular)"
        fracture_type = "Tibial/Fibular Fracture"
        notes = "High risk of compartment syndrome. Monitor for neurovascular compromise."
    elif "femur" in location_lower or "thigh" in location_lower:
        region = "Proximal Lower Extremity (Femoral)"
        fracture_type = "Femoral Shaft Fracture"
        notes = "High-energy trauma. Risk of significant blood loss. Urgent orthopedic consultation required."
    elif "rib" in location_lower or "chest" in location_lower:
        region = "Thoracic Cage (Ribs)"
        fracture_type = "Rib Fracture"
        notes = "Monitor for pneumothorax or hemothorax. Assess respiratory function."
    elif "spine" in location_lower or "vertebra" in location_lower:
        region = "Axial Skeleton (Vertebral)"
        fracture_type = "Vertebral Compression Fracture"
        notes = "Spinal precautions mandatory. Assess for neurological deficits."
    elif "pelvis" in location_lower or "hip" in location_lower:
        region = "Pelvic Girdle"
        fracture_type = "Pelvic/Hip Fracture"
        notes = "High-risk injury. Assess for internal hemorrhage and pelvic stability."
    elif "clavicle" in location_lower or "collar" in location_lower:
        region = "Shoulder Girdle (Clavicular)"
        fracture_type = "Clavicle Fracture"
        notes = "Common in trauma. Check for associated rib fractures or pneumothorax."
    elif "ankle" in location_lower:
        region = "Distal Lower Extremity (Ankle)"
        fracture_type = "Ankle Fracture"
        notes = "Assess joint stability using Ottawa Ankle Rules. May require surgical fixation."
    elif "wrist" in location_lower:
        region = "Distal Upper Extremity (Wrist)"
        fracture_type = "Distal Radius Fracture (Colles'/Smith's)"
        notes = "Common osteoporotic fracture. Assess for median nerve compression."
    else:
        region = location_str
        fracture_type = "Fracture"
        notes = "Detailed anatomical localization recommended with additional imaging."
    
    return region, fracture_type, notes


def generate_bone_medical_report(ai_conf: float, location: str, method: str, symptoms: dict) -> str:
    """
    Generate comprehensive medical report for bone fractures.
    """
    region, fracture_type, clinical_notes = analyze_bone_location(location)
    
    # Build narrative report
    if ai_conf > 0.7:
        severity = "high-probability"
        recommendation = "Immediate orthopedic consultation is strongly recommended for definitive management"
    elif ai_conf > 0.4:
        severity = "moderate-probability"
        recommendation = "Orthopedic review advised within 24-48 hours. Consider additional imaging if clinically indicated"
    else:
        severity = "low-probability"
        recommendation = "Clinical correlation advised. Consider follow-up imaging if symptoms persist"
    
    report = f"Radiographic analysis reveals a {severity} {fracture_type} localized to the {region}. "
    report += f"The detection algorithm utilized {method}, achieving {ai_conf*100:.1f}% confidence in fracture identification. "
    report += f"{clinical_notes} "
    
    # Add symptom correlation
    present_symptoms = []
    if symptoms.get('deformity'): present_symptoms.append("visible deformity")
    if symptoms.get('immobility'): present_symptoms.append("functional impairment")
    if symptoms.get('trauma'): present_symptoms.append("recent traumatic mechanism")
    if symptoms.get('swelling'): present_symptoms.append("periarticular swelling")
    if symptoms.get('pain'): present_symptoms.append("localized pain")
    
    if present_symptoms:
        symptom_str = ", ".join(present_symptoms)
        report += f"Clinical presentation includes {symptom_str}, which correlates with the radiographic findings and supports the diagnosis. "
    else:
        report += "Patient is currently reporting minimal symptoms, suggesting possible occult or non-displaced fracture requiring close clinical follow-up. "
    
    report += recommendation + "."
    
    return report


def analyze_tb_location(location_str: str) -> tuple:
    """
    Map TB detection to anatomical lung zones.
    Returns: (lung_zone, pattern_type, clinical_significance)
    """
    location_lower = location_str.lower()
    
    if "upper" in location_lower:
        zone = "Upper Lobe"
        pattern = "Apical/Posterior Segment Involvement"
        significance = "Classic reactivation TB pattern. High bacillary load expected."
    elif "middle" in location_lower:
        zone = "Middle Lobe/Lingula"
        pattern = "Middle Lobe Syndrome"
        significance = "May indicate endobronchial obstruction or primary TB."
    elif "lower" in location_lower:
        zone = "Lower Lobe"
        pattern = "Basilar Infiltration"
        significance = "Atypical for post-primary TB. Consider primary infection or atypical mycobacteria."
    else:
        zone = location_str
        pattern = "Diffuse Parenchymal Pattern"
        significance = "Extensive disease suggests active transmission risk."
    
    if "right" in location_lower:
        laterality = "Right"
    elif "left" in location_lower:
        laterality = "Left"
    else:
        laterality = "Bilateral"
    
    return f"{laterality} {zone}", pattern, significance


def generate_tb_medical_report(ai_conf: float, location: str, method: str, symptoms: dict) -> str:
    """
    Generate comprehensive medical report for TB detection.
    """
    lung_zone, pattern, significance = analyze_tb_location(location)
    
    # Severity assessment
    if ai_conf > 0.75:
        suspicion_level = "high clinical suspicion"
        action = "Immediate isolation precautions and sputum AFB collection (3 samples) are mandated"
    elif ai_conf > 0.5:
        suspicion_level = "moderate clinical suspicion"
        action = "Sputum AFB microscopy, GeneXpert MTB/RIF, and tuberculin skin test are recommended"
    else:
        suspicion_level = "low clinical suspicion"
        action = "Clinical correlation advised. Consider alternative diagnoses (fungal, atypical pneumonia)"
    
    report = f"Chest radiograph demonstrates {suspicion_level} for active pulmonary tuberculosis, with {lung_zone} {pattern.lower()}. "
    report += f"AI analysis using {method} achieved {ai_conf*100:.1f}% confidence, identifying TB-characteristic lesions. "
    report += f"{significance} "
    
    # Symptom correlation
    tb_symptoms = []
    if symptoms.get('cough'): tb_symptoms.append("chronic productive cough")
    if symptoms.get('blood_sputum'): tb_symptoms.append("hemoptysis")
    if symptoms.get('night_sweats'): tb_symptoms.append("nocturnal diaphoresis")
    if symptoms.get('weight_loss'): tb_symptoms.append("constitutional weight loss")
    if symptoms.get('fever'): tb_symptoms.append("persistent fever")
    
    if tb_symptoms:
        symptom_str = ", ".join(tb_symptoms)
        report += f"Clinical presentation includes {symptom_str}, which aligns with classic pulmonary TB symptomatology and increases pre-test probability. "
    else:
        report += "Patient is asymptomatic, suggesting subclinical or latent disease. Consider LTBI screening in high-risk populations. "
    
    report += f"{action}. Contact tracing and directly observed therapy (DOTS) protocols should be initiated pending confirmatory testing."
    
    return report


def analyze_pneumonia_severity(mask_area_ratio: float) -> tuple:
    """
    Classify pneumonia extent based on U-Net segmentation.
    Returns: (severity, extent_description, clinical_impact)
    """
    if mask_area_ratio < 0.01:
        return "Minimal", "trace opacity (<1% lung involvement)", "likely self-limiting, outpatient management appropriate"
    elif mask_area_ratio < 0.05:
        return "Focal", "localized consolidation (1-5% lung involvement)", "community management with oral antibiotics, reassess in 48-72 hours"
    elif mask_area_ratio < 0.15:
        return "Moderate", "multilobar involvement (5-15% lung involvement)", "consider hospitalization, IV antibiotics, and oxygen supplementation if hypoxic"
    elif mask_area_ratio < 0.30:
        return "Severe", "extensive consolidation (15-30% lung involvement)", "ICU monitoring indicated, risk of respiratory failure, consider mechanical ventilation"
    else:
        return "Critical", "bilateral diffuse opacification (>30% lung involvement)", "ARDS risk, immediate ICU admission, high-flow oxygen or ECMO may be required"


def generate_pneumonia_medical_report(classifier_conf: float, location: str, mask_area_ratio: float, method: str, symptoms: dict) -> str:
    """
    Generate comprehensive medical report for pneumonia with U-Net analysis.
    """
    severity, extent_desc, clinical_impact = analyze_pneumonia_severity(mask_area_ratio)
    
    # Parse location for anatomical detail
    if "right" in location.lower():
        laterality = "right"
    elif "left" in location.lower():
        laterality = "left"
    else:
        laterality = "bilateral"
    
    if "upper" in location.lower():
        zone = "upper lobe"
    elif "middle" in location.lower():
        zone = "middle lobe"
    elif "lower" in location.lower():
        zone = "lower lobe"
    else:
        zone = "lung fields"
    
    report = f"Chest radiograph demonstrates {severity.lower()} community-acquired pneumonia with {extent_desc} affecting the {laterality} {zone}. "
    report += f"AI analysis combining ResNet classification ({classifier_conf*100:.1f}% confidence) and U-Net segmentation mapping reveals consolidation patterns consistent with bacterial pneumonia. "
    report += f"The affected region occupies approximately {mask_area_ratio*100:.1f}% of total lung parenchyma, indicating {clinical_impact}. "
    
    # Add pathophysiology
    if mask_area_ratio > 0.15:
        report += "The extensive consolidation suggests alveolar filling with inflammatory exudate, impairing gas exchange. "
    
    # Symptom correlation
    pneumonia_symptoms = []
    if symptoms.get('fever'): pneumonia_symptoms.append("fever >38°C")
    if symptoms.get('cough'): pneumonia_symptoms.append("productive cough")
    if symptoms.get('chest_pain') or symptoms.get('pain'): pneumonia_symptoms.append("pleuritic chest pain")
    
    # Get additional pneumonia-specific symptoms if available
    if symptoms.get('high_fever'): pneumonia_symptoms.append("high-grade fever")
    if symptoms.get('productive_cough'): pneumonia_symptoms.append("purulent sputum production")
    if symptoms.get('shortness_breath'): pneumonia_symptoms.append("dyspnea")
    if symptoms.get('rapid_breathing'): pneumonia_symptoms.append("tachypnea")
    
    if pneumonia_symptoms:
        symptom_str = ", ".join(pneumonia_symptoms)
        report += f"Clinical presentation includes {symptom_str}, supporting the radiographic diagnosis and indicating active infection. "
    else:
        report += "Patient is minimally symptomatic, suggesting early-stage infection or atypical presentation. "
    
    # Treatment recommendation
    if severity == "Critical" or severity == "Severe":
        report += "Urgent hospitalization with empiric broad-spectrum antibiotics (e.g., ceftriaxone + azithromycin) is mandated. Blood cultures, sputum culture, and urinary antigens should be obtained prior to antibiotic initiation."
    elif severity == "Moderate":
        report += "Hospital admission recommended for IV antibiotics and monitoring. CURB-65 or PSI scoring should guide disposition and antibiotic selection."
    else:
        report += "Outpatient management with oral antibiotics (e.g., amoxicillin-clavulanate or macrolide) is appropriate. Clinical reassessment within 48-72 hours is advised."
    
    return report


def generate_visual_reasoning(ai_conf: float, ai_loc: str, ai_method: str, label: str, scan_type: str, chest_condition: str = None, symptoms: dict = None, mask_area: float = 0.0) -> str:
    """
    Generate comprehensive medical-grade visual analysis text.
    """
    if symptoms is None:
        symptoms = {}
    
    if ai_conf > 0:
        if scan_type == "Bone":
            return generate_bone_medical_report(ai_conf, ai_loc, ai_method, symptoms)
        elif chest_condition == "TB":
            return generate_tb_medical_report(ai_conf, ai_loc, ai_method, symptoms)
        elif chest_condition == "Pneumonia":
            return generate_pneumonia_medical_report(ai_conf, ai_loc, mask_area, ai_method, symptoms)
        else:
            return f"AI detected {label} in the {ai_loc} using {ai_method} analysis with {ai_conf*100:.1f}% confidence."
    else:
        if scan_type == "Bone":
            return "Radiographic analysis demonstrates no evidence of acute fracture, dislocation, or bony destruction. Cortical margins are intact. Soft tissue planes appear preserved. Clinical correlation is advised, as occult fractures may not be visible on initial radiographs. Consider MRI or bone scan if clinical suspicion remains high."
        elif chest_condition == "TB":
            return "Chest radiograph shows clear lung fields bilaterally with no evidence of cavitation, infiltrates, or hilar lymphadenopathy characteristic of pulmonary tuberculosis. Normal cardiomediastinal silhouette. Costophrenic angles are sharp. TB is radiographically excluded, though latent infection cannot be ruled out without additional testing (TST/IGRA)."
        elif chest_condition == "Pneumonia":
            return "Chest radiograph demonstrates clear lung parenchyma without consolidation, infiltrates, or pleural effusion. No air-space opacification suggestive of pneumonia. Cardiac silhouette and mediastinal contours are within normal limits. Pneumonia is radiographically excluded."
        else:
            return "Radiographic analysis is within normal limits. No significant pathology detected."


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
            "lung": lung_model is not None,
            "pneumonia_classifier": pneumonia_classifier is not None,
            "pneumonia_segmentation": pneumonia_segmentation is not None
        },
        version="4.0"
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
    For Chest X-rays, this endpoint returns the image type. Use /analyze/tb or /analyze/pneumonia for specific analysis.
    
    **Workflow:**
    1. Router classifies image as Bone/Chest/Invalid
    2. For Bone: Runs fracture analysis
    3. For Chest: Returns that user needs to select TB or Pneumonia
    4. Clinical symptoms boost the risk score
    5. Returns comprehensive diagnosis
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
        print(f"🔍 ROUTER OUTPUT: scan_type='{scan_type}', confidence={route_confidence:.4f}")
        
        # Handle Invalid Images
        if scan_type == "Invalid":
            return AnalysisResponse(
                success=False,
                scan_type="Invalid",
                chest_condition=None,
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
            chest_condition = None
            
            # Step 3: CLINICAL BOOST
            boost_score, clinical_reasons, clinical_context = calculate_clinical_boost(symptoms, scan_type)
            
            # Step 4: FUSION LOGIC
            final_conf, risk_class, status_msg = calculate_risk_assessment(ai_conf, boost_score, scan_type)
            
            # Step 5: GENERATE REASONING
            visual_text = generate_visual_reasoning(ai_conf, ai_loc, ai_method, label, scan_type, None, symptoms, 0.0)
            
            # Build response dict for history
            result_dict = {
                "scan_type": scan_type,
                "chest_condition": None,
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
                chest_condition=None,
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
        
        else:  # Chest - Return that selection is required
            # For the universal endpoint, we return a response indicating chest detected
            # but no specific analysis done - user must choose TB or Pneumonia
            return AnalysisResponse(
                success=True,
                scan_type="Chest",
                chest_condition="SELECTION_REQUIRED",
                scan_type_confidence=round(route_confidence, 4),
                ai_confidence=0.0,
                ai_location="N/A",
                ai_method="Pending Selection",
                detected_condition=None,
                clinical_boost=0.0,
                clinical_reasons=[],
                final_confidence=0.0,
                risk_class="PENDING",
                status_message="Chest X-ray detected. Please select analysis type: TB or Pneumonia",
                visual_analysis_text="Chest X-ray identified. Select between TB Analysis or Pneumonia Analysis to proceed.",
                clinical_context_text="Different symptom profiles apply for TB vs Pneumonia. Please select the appropriate condition to analyze.",
                annotated_image_base64=image_to_base64(image) if include_image else None
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
        visual_text = generate_visual_reasoning(ai_conf, ai_loc, ai_method, label, "Bone", None, symptoms, 0.0)
        
        return AnalysisResponse(
            success=True,
            scan_type="Bone",
            chest_condition=None,
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


@app.post("/analyze/tb", response_model=AnalysisResponse)
async def analyze_tb(
    file: UploadFile = File(...),
    # TB-specific symptoms
    cough: bool = Form(False),
    blood_sputum: bool = Form(False),
    night_sweats: bool = Form(False),
    weight_loss: bool = Form(False),
    fever: bool = Form(False),
    include_image: bool = Form(True)
):
    """
    TB-specific analysis endpoint using the lung/TB YOLO model.
    Use this when user selects TB analysis for chest X-ray.
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
        
        # Use TB-specific lung model
        ai_img, ai_conf, ai_loc, ai_method, label = analyze_lung(image)
        boost_score, clinical_reasons, clinical_context = calculate_clinical_boost(symptoms, "Chest", "TB")
        final_conf, risk_class, status_msg = calculate_risk_assessment(ai_conf, boost_score, "Chest", "TB")
        visual_text = generate_visual_reasoning(ai_conf, ai_loc, ai_method, label, "Chest", "TB", symptoms, 0.0)
        
        # Build response for history
        result_dict = {
            "scan_type": "Chest",
            "chest_condition": "TB",
            "scan_type_confidence": 1.0,
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
        
        add_to_history(result_dict, file.filename or "TB Scan")
        
        return AnalysisResponse(
            success=True,
            scan_type="Chest",
            chest_condition="TB",
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
        raise HTTPException(status_code=500, detail=f"TB analysis failed: {str(e)}")


@app.post("/analyze/pneumonia", response_model=AnalysisResponse)
async def analyze_pneumonia_endpoint(
    file: UploadFile = File(...),
    # Pneumonia-specific symptoms
    high_fever: bool = Form(False),
    productive_cough: bool = Form(False),
    shortness_breath: bool = Form(False),
    chest_pain: bool = Form(False),
    rapid_breathing: bool = Form(False),
    fever: bool = Form(False),
    cough: bool = Form(False),
    include_image: bool = Form(True)
):
    """
    Pneumonia-specific analysis endpoint.
    Uses classification model (best_model_auc.keras) + U-Net segmentation (pneumonia_unet_v2.h5).
    Use this when user selects Pneumonia analysis for chest X-ray.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        symptoms = {
            'high_fever': high_fever,
            'productive_cough': productive_cough,
            'shortness_breath': shortness_breath,
            'chest_pain': chest_pain,
            'rapid_breathing': rapid_breathing,
            'fever': fever,
            'cough': cough
        }
        
        # Use Pneumonia-specific models (classification + segmentation)
        ai_img, ai_conf, ai_loc, ai_method, label = analyze_pneumonia(image)
        boost_score, clinical_reasons, clinical_context = calculate_clinical_boost(symptoms, "Chest", "Pneumonia")
        final_conf, risk_class, status_msg = calculate_risk_assessment(ai_conf, boost_score, "Chest", "Pneumonia")
        
        # Extract mask_area from ai_loc for pneumonia
        mask_area = ai_loc.get('mask_area_ratio', 0.0) if isinstance(ai_loc, dict) else 0.0
        visual_text = generate_visual_reasoning(ai_conf, ai_loc, ai_method, label, "Chest", "Pneumonia", symptoms, mask_area)
        
        # Build response for history
        result_dict = {
            "scan_type": "Chest",
            "chest_condition": "Pneumonia",
            "scan_type_confidence": 1.0,
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
        
        add_to_history(result_dict, file.filename or "Pneumonia Scan")
        
        return AnalysisResponse(
            success=True,
            scan_type="Chest",
            chest_condition="Pneumonia",
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
        raise HTTPException(status_code=500, detail=f"Pneumonia analysis failed: {str(e)}")


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
    Legacy lung analysis endpoint (uses TB model). 
    Prefer /analyze/tb or /analyze/pneumonia for specific analysis.
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
        boost_score, clinical_reasons, clinical_context = calculate_clinical_boost(symptoms, "Chest", "TB")
        final_conf, risk_class, status_msg = calculate_risk_assessment(ai_conf, boost_score, "Chest", "TB")
        visual_text = generate_visual_reasoning(ai_conf, ai_loc, ai_method, label, "Chest", "TB", symptoms, 0.0)
        
        return AnalysisResponse(
            success=True,
            scan_type="Chest",
            chest_condition="TB",
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
        "chestCondition": result.get("chest_condition"),
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
