import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import symptom_solver  # Importing your logic file

# --- CONFIGURATION ---
st.set_page_config(page_title="Dayflow Ortho CDSS", page_icon="🦴", layout="wide")

# --- CUSTOM CSS (FIXED BLACK TEXT) ---
st.markdown("""
    <style>
    .report-box { 
        border: 1px solid #ddd; 
        padding: 25px; 
        border-radius: 12px; 
        background-color: #f8f9fa; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: black !important; /* <--- FORCES TEXT TO BE BLACK */
    }
    .report-box h4, .report-box li, .report-box p {
        color: black !important; /* <--- ENSURES HEADERS/LISTS ARE BLACK */
    }
    .metric-card { background-color: white; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .high-risk { color: #d32f2f !important; font-weight: 800; font-size: 24px; }
    .mod-risk { color: #fbc02d !important; font-weight: 800; font-size: 24px; }
    .warn-risk { color: #ff9800 !important; font-weight: 800; font-size: 24px; }
    .low-risk { color: #388e3c !important; font-weight: 800; font-size: 24px; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD INTELLIGENCE ---
@st.cache_resource
def load_brain():
    try:
        model = YOLO('bone_model.pt')
        return model
    except Exception as e:
        return None

model = load_brain()

# --- VISUAL LOGIC ENGINE (SMART ZOOM) ---
def analyze_image(image):
    w, h = image.size
    
    # PHASE 1: Standard Full Scan (with TTA for extra sensitivity)
    results = model.predict(image, conf=0.15, augment=True, verbose=False)
    
    if len(results[0].boxes) > 0:
        # Found in Standard Scan
        box = results[0].boxes[0]
        conf = float(box.conf[0])
        
        # Calculate Quadrant
        cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
        loc_x = "Right" if cx > w/2 else "Left"
        loc_y = "Bottom" if cy > h/2 else "Top"
        
        plot = results[0].plot(line_width=3)
        return Image.fromarray(plot[..., ::-1]), conf, f"{loc_y}-{loc_x} Quadrant", "Standard Scan"

    # PHASE 2: Deep Zoom (The Rescue Mechanism)
    # Split image into 4 overlapping quadrants
    crops = [
        ("Top-Left",  (0, 0, w//2 + 50, h//2 + 50)),
        ("Top-Right", (w//2 - 50, 0, w, h//2 + 50)),
        ("Bot-Left",  (0, h//2 - 50, w//2 + 50, h)),
        ("Bot-Right", (w//2 - 50, h//2 - 50, w, h))
    ]
    
    for loc, region in crops:
        crop = image.crop(region)
        res = model.predict(crop, conf=0.15, verbose=False)
        
        if len(res[0].boxes) > 0:
            # Found in Zoom
            conf = float(res[0].boxes.conf[0])
            plot = res[0].plot(line_width=3)
            return Image.fromarray(plot[..., ::-1]), conf, f"{loc} Quadrant", "Deep Zoom"
            
    # PHASE 3: Clean
    return image, 0.0, "None", "Clean"

# --- SIDEBAR UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80)
    st.title("Dayflow Intake")
    st.write("### Patient Symptoms")
    
    # Symptom Checklist
    symptoms = {
        'deformity': st.checkbox("Visible Bone Deformity"),
        'immobility': st.checkbox("Inability to Move Limb"),
        'trauma': st.checkbox("Recent Trauma History"),
        'swelling': st.checkbox("Swelling / Bruising"),
        'pain': st.checkbox("Localized Pain")
    }
    
    st.write("---")
    st.caption("Dayflow Clinical Decision Support System v2.1")

# --- MAIN PAGE UI ---
st.title("Dayflow: Clinical Decision Support System")
st.markdown("### 🧠 AI + Clinical Logic Fusion Engine")

if not model:
    st.error("⚠️ CRITICAL ERROR: 'bone_model.pt' not found. Please place the model file in the project folder.")
    st.stop()

uploaded_file = st.file_uploader("Upload X-Ray Scan", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Load Image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        # FIXED: Replaced use_column_width with use_container_width
        st.image(image, caption="Source Scan", use_container_width=True)
        
    with col2:
        with st.spinner("Running Multi-Modal Analysis..."):
            # 1. RUN AI ANALYSIS
            ai_img, ai_conf, ai_loc, ai_method = analyze_image(image)
            
            # 2. RUN CLINICAL ANALYSIS
            boost_score, clinical_reasons = symptom_solver.calculate_clinical_boost(symptoms)
            
            # 3. FUSION LOGIC
            # Formula: Final Risk = AI Confidence + Clinical Boost
            if ai_conf > 0:
                final_conf = min(ai_conf + boost_score, 1.0)
            else:
                # If AI sees nothing, symptoms alone do not confirm a fracture,
                # but they can trigger a "Clinical Warning".
                final_conf = 0.0
            
            # 4. DETERMINE STATUS
            if final_conf > 0.60:
                risk_class = "HIGH RISK"
                css = "high-risk"
                status_msg = "Fracture Confirmed (AI + Clinical)"
            elif final_conf > 0.30: 
                risk_class = "MODERATE RISK"
                css = "mod-risk"
                status_msg = "Suspected Fracture (Review Required)"
            elif boost_score > 0.30: # High symptoms, Clean Scan
                risk_class = "CLINICAL WARNING"
                css = "warn-risk"
                status_msg = "Scan Negative, but Symptoms Critical (Occult Injury?)"
            else:
                risk_class = "LOW RISK"
                css = "low-risk"
                status_msg = "No Significant Findings"

        # --- DISPLAY RESULTS ---
        # FIXED: Replaced use_column_width with use_container_width
        st.image(ai_img, caption=f"AI Analysis Layer ({ai_method})", use_container_width=True)
        
        # Metrics Dashboard
        m1, m2, m3 = st.columns(3)
        m1.metric("AI Confidence", f"{ai_conf*100:.1f}%", ai_loc)
        m2.metric("Clinical Boost", f"+{boost_score*100:.0f}%", f"{len(clinical_reasons)} Factors")
        m3.metric("Final Probability", f"{final_conf*100:.1f}%")

    # --- FINAL REPORT CARD ---
    st.markdown("---")
    
    # Dynamic Reasoning Text
    if ai_conf > 0:
        visual_text = f"AI localized a fracture pattern in the <b>{ai_loc}</b> using <b>{ai_method}</b> logic."
    else:
        visual_text = "Radiological analysis is negative. No fracture patterns detected in Standard or Deep Zoom scans."
        
    if clinical_reasons:
        clinical_text = f"Risk elevated due to: {', '.join(clinical_reasons)}."
    else:
        clinical_text = "Patient is asymptomatic with no significant risk factors reported."

    st.markdown(f"""
    <div class="report-box">
        <h4>📄 Diagnostic Reasoning</h4>
        <ul>
            <li><strong>Visual Analysis:</strong> {visual_text}</li>
            <li><strong>Clinical Context:</strong> {clinical_text}</li>
        </ul>
        <hr>
        <p style="margin-bottom:5px; font-size:14px; color:#333;">FINAL DIAGNOSIS:</p>
        <p class="{css}">{risk_class}</p>
        <p style="font-style:italic; color:#333;">{status_msg}</p>
    </div>
    """, unsafe_allow_html=True)