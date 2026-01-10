# # # # # # # # import streamlit as st
# # # # # # # # from PIL import Image
# # # # # # # # import numpy as np
# # # # # # # # import tensorflow as tf
# # # # # # # # from ultralytics import YOLO
# # # # # # # # import cv2
# # # # # # # # import pre_symptoms_solver

# # # # # # # # # --- CONFIG ---
# # # # # # # # st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# # # # # # # # # --- LOAD MODELS ---
# # # # # # # # @st.cache_resource
# # # # # # # # def load_all_brains():
# # # # # # # #     try:
# # # # # # # #         # Replace these with your actual model filenames
# # # # # # # #         router = tf.keras.models.load_model('router_model.h5')
# # # # # # # #         bone_model = YOLO('bone_model.pt')
# # # # # # # #         lung_model = YOLO('lung_model.pt')
# # # # # # # #         return router, bone_model, lung_model
# # # # # # # #     except:
# # # # # # # #         return None, None, None

# # # # # # # # router, bone_model, lung_model = load_all_brains()

# # # # # # # # # --- SPATIAL REASONING ENGINE ---
# # # # # # # # def get_spatial_description(box, w, h, scan_type):
# # # # # # # #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
    
# # # # # # # #     if scan_type == "Bone":
# # # # # # # #         loc_x = "Right" if cx > w/2 else "Left"
# # # # # # # #         loc_y = "Bottom" if cy > h/2 else "Top"
# # # # # # # #         return f"{loc_y}-{loc_x} Quadrant"
# # # # # # # #     else:
# # # # # # # #         # Lung Zone Logic (Upper, Middle, Lower)
# # # # # # # #         if cy < h/3: zone = "Upper Lobe"
# # # # # # # #         elif cy < 2*h/3: zone = "Middle Zone"
# # # # # # # #         else: zone = "Lower Lobe"
# # # # # # # #         side = "Right" if cx > w/2 else "Left"
# # # # # # # #         return f"{side} {zone}"

# # # # # # # # # --- ANALYSIS ENGINE ---
# # # # # # # # def run_unified_analysis(image, scan_type):
# # # # # # # #     w, h = image.size
# # # # # # # #     model = bone_model if scan_type == "Bone" else lung_model
    
# # # # # # # #     # Run Prediction
# # # # # # # #     results = model.predict(image, conf=0.15, augment=True, verbose=False)
    
# # # # # # # #     if len(results[0].boxes) > 0:
# # # # # # # #         box = results[0].boxes[0]
# # # # # # # #         conf = float(box.conf[0])
# # # # # # # #         loc = get_spatial_description(box, w, h, scan_type)
# # # # # # # #         label = model.names[int(box.cls[0])]
# # # # # # # #         plot = results[0].plot()
# # # # # # # #         return Image.fromarray(plot[..., ::-1]), conf, loc, label
    
# # # # # # # #     return image, 0.0, "None", "Clean"

# # # # # # # # # --- UI SIDEBAR ---
# # # # # # # # with st.sidebar:
# # # # # # # #     st.title("🏥 Dayflow Intake")
# # # # # # # #     st.info(f"Connected to: {scan_type if 'scan_type' in locals() else 'Waiting for Upload'}")
    
# # # # # # # #     # Symptom Logic
# # # # # # # #     st.write("### Patient Symptoms")
# # # # # # # #     syms = {}
# # # # # # # #     # Common
# # # # # # # #     syms['fever'] = st.checkbox("Fever")
# # # # # # # #     syms['pain'] = st.checkbox("Localized Pain")
# # # # # # # #     # Bone Specific
# # # # # # # #     with st.expander("Bone/Trauma Symptoms"):
# # # # # # # #         syms['deformity'] = st.checkbox("Visible Deformity")
# # # # # # # #         syms['immobility'] = st.checkbox("Inability to Move")
# # # # # # # #         syms['trauma'] = st.checkbox("Recent Trauma")
# # # # # # # #     # Lung Specific
# # # # # # # #     with st.expander("Respiratory Symptoms"):
# # # # # # # #         syms['cough'] = st.checkbox("Persistent Cough")
# # # # # # # #         syms['blood_sputum'] = st.checkbox("Blood in Sputum")
# # # # # # # #         syms['night_sweats'] = st.checkbox("Night Sweats")
# # # # # # # #         syms['weight_loss'] = st.checkbox("Weight Loss")

# # # # # # # # # --- MAIN PAGE ---
# # # # # # # # st.title("Dayflow: Multi-Specialist AI Suite")

# # # # # # # # uploaded_file = st.file_uploader("Upload Medical Scan", type=['jpg', 'png', 'jpeg'])

# # # # # # # # if uploaded_file and router:
# # # # # # # #     image = Image.open(uploaded_file).convert('RGB')
    
# # # # # # # #     # 1. ROUTER PHASE
# # # # # # # #     img_array = np.array(image.resize((224, 224))) / 255.0
# # # # # # # #     route_idx = np.argmax(router.predict(np.expand_dims(img_array, axis=0), verbose=0))
# # # # # # # #     scan_type = ["Bone", "Chest", "Invalid"][route_idx]
    
# # # # # # # #     col1, col2 = st.columns([1, 1.5])
# # # # # # # #     with col1:
# # # # # # # #         st.image(image, caption=f"Input: {scan_type} X-ray", use_container_width=True)

# # # # # # # #     with col2:
# # # # # # # #         with st.spinner(f"Analyzing {scan_type} patterns..."):
# # # # # # # #             # 2. SPECIALIST & SPATIAL PHASE
# # # # # # # #             ai_img, ai_conf, ai_loc, ai_label = run_unified_analysis(image, scan_type)
            
# # # # # # # #             # 3. CLINICAL FUSION
# # # # # # # #             boost, reasons = pre_symptoms_solver.calculate_clinical_boost(syms, scan_type)
# # # # # # # #             final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
            
# # # # # # # #         st.image(ai_img, caption="AI Reasoning Overlay", use_container_width=True)
        
# # # # # # # #         # Metrics
# # # # # # # #         m1, m2, m3 = st.columns(3)
# # # # # # # #         m1.metric("AI Confidence", f"{ai_conf*100:.1f}%", ai_loc if ai_conf > 0 else "N/A")
# # # # # # # #         m2.metric("Symptom Boost", f"+{boost*100:.0f}%", f"{len(reasons)} Factors")
# # # # # # # #         m3.metric("Final Risk", f"{final_prob*100:.1f}%")

# # # # # # # #     # 4. REPORTING
# # # # # # # #     st.markdown("---")
# # # # # # # #     risk_color = "#d32f2f" if final_prob > 0.6 else "#388e3c"
    
# # # # # # # #     st.markdown(f"""
# # # # # # # #     <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; color:black;">
# # # # # # # #         <h4 style="color:black;">📋 Automated Diagnostic Reasoning</h4>
# # # # # # # #         <p><b>Visual Findings:</b> {ai_label} detected in the <b>{ai_loc}</b>.</p>
# # # # # # # #         <p><b>Clinical Correlation:</b> Risk elevated based on {', '.join(reasons) if reasons else 'no symptoms'}.</p>
# # # # # # # #         <hr>
# # # # # # # #         <h2 style="color:{risk_color};">PROBABILITY: {final_prob*100:.1f}%</h2>
# # # # # # # #     </div>
# # # # # # # #     """, unsafe_allow_html=True)











# # # # # # # import streamlit as st
# # # # # # # from PIL import Image
# # # # # # # import numpy as np
# # # # # # # import tensorflow as tf
# # # # # # # from ultralytics import YOLO
# # # # # # # import cv2
# # # # # # # import pre_symptoms_solver

# # # # # # # # --- CONFIGURATION ---
# # # # # # # st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# # # # # # # # --- CUSTOM CSS ---
# # # # # # # st.markdown("""
# # # # # # #     <style>
# # # # # # #     .report-box { 
# # # # # # #         border: 1px solid #e0e0e0; 
# # # # # # #         padding: 20px; 
# # # # # # #         border-radius: 12px; 
# # # # # # #         background-color: #ffffff; 
# # # # # # #         color: #333;
# # # # # # #         box-shadow: 0 2px 5px rgba(0,0,0,0.05);
# # # # # # #     }
# # # # # # #     .report-header { color: #2c3e50; font-weight: bold; margin-bottom: 10px; }
# # # # # # #     .risk-high { color: #d32f2f; font-weight: 800; font-size: 24px; }
# # # # # # #     .risk-safe { color: #2e7d32; font-weight: 800; font-size: 24px; }
# # # # # # #     </style>
# # # # # # # """, unsafe_allow_html=True)

# # # # # # # # --- LOAD MODELS ---
# # # # # # # @st.cache_resource
# # # # # # # def load_all_brains():
# # # # # # #     try:
# # # # # # #         router = tf.keras.models.load_model('router_model.h5')
# # # # # # #         bone_model = YOLO('bone_model.pt')
# # # # # # #         lung_model = YOLO('lung_model.pt')
# # # # # # #         return router, bone_model, lung_model
# # # # # # #     except Exception as e:
# # # # # # #         st.error(f"Model Loading Error: {e}")
# # # # # # #         return None, None, None

# # # # # # # router, bone_model, lung_model = load_all_brains()

# # # # # # # # --- SPATIAL REASONING ENGINE ---
# # # # # # # def get_spatial_description(box, w, h, scan_type):
# # # # # # #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
    
# # # # # # #     if scan_type == "Bone":
# # # # # # #         loc_x = "Right" if cx > w/2 else "Left"
# # # # # # #         loc_y = "Distal/Bottom" if cy > h/2 else "Proximal/Top"
# # # # # # #         return f"{loc_y}-{loc_x} Quadrant"
# # # # # # #     else:
# # # # # # #         # Lung Zone Logic (Medical Standard: Upper, Middle, Lower Fields)
# # # # # # #         if cy < h/3: zone = "Upper Lung Field"
# # # # # # #         elif cy < 2*h/3: zone = "Middle Lung Field"
# # # # # # #         else: zone = "Lower Lung Field"
# # # # # # #         side = "Right" if cx > w/2 else "Left"
# # # # # # #         return f"{side} {zone}"

# # # # # # # # --- ANALYSIS ENGINE ---
# # # # # # # def run_unified_analysis(image, scan_type):
# # # # # # #     w, h = image.size
# # # # # # #     model = bone_model if scan_type == "Bone" else lung_model
    
# # # # # # #     # Run Prediction (TTA enabled for stability)
# # # # # # #     results = model.predict(image, conf=0.15, augment=True, verbose=False)
    
# # # # # # #     if len(results[0].boxes) > 0:
# # # # # # #         box = results[0].boxes[0]
# # # # # # #         conf = float(box.conf[0])
# # # # # # #         loc = get_spatial_description(box, w, h, scan_type)
# # # # # # #         label = model.names[int(box.cls[0])]
        
# # # # # # #         # Generate Plot
# # # # # # #         plot = results[0].plot(line_width=2, font_size=1)
# # # # # # #         return Image.fromarray(plot[..., ::-1]), conf, loc, label
    
# # # # # # #     return image, 0.0, "None", "Clean"

# # # # # # # # --- UI SIDEBAR ---
# # # # # # # with st.sidebar:
# # # # # # #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# # # # # # #     st.title("Dayflow Triage")
# # # # # # #     st.markdown("---")
    
# # # # # # #     st.subheader("Patient Clinical Profile")
    
# # # # # # #     syms = {}
    
# # # # # # #     with st.expander("🦴 Orthopedic Signs", expanded=True):
# # # # # # #         syms['deformity'] = st.checkbox("Gross Deformity")
# # # # # # #         syms['immobility'] = st.checkbox("Functional Loss (Immobility)")
# # # # # # #         syms['trauma'] = st.checkbox("Recent High-Impact Trauma")
# # # # # # #         syms['swelling'] = st.checkbox("Edema / Swelling")
# # # # # # #         syms['pain'] = st.checkbox("Acute Localized Pain")

# # # # # # #     with st.expander("🫁 Pulmonary Signs", expanded=False):
# # # # # # #         syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
# # # # # # #         syms['cough'] = st.checkbox("Chronic Cough (>2w)")
# # # # # # #         syms['weight_loss'] = st.checkbox("Unexplained Weight Loss")
# # # # # # #         syms['night_sweats'] = st.checkbox("Night Sweats")
# # # # # # #         syms['fever'] = st.checkbox("High Fever (>38°C)")

# # # # # # # # --- MAIN PAGE ---
# # # # # # # st.title("🏥 Dayflow: Multi-Specialist AI Suite")

# # # # # # # if not router:
# # # # # # #     st.error("⚠️ System Offline: Models could not be loaded. Please check file paths.")
# # # # # # #     st.stop()

# # # # # # # uploaded_file = st.file_uploader("Upload Radiography Scan (X-Ray)", type=['jpg', 'png', 'jpeg'])

# # # # # # # if uploaded_file:
# # # # # # #     image = Image.open(uploaded_file).convert('RGB')
    
# # # # # # #     # 1. ROUTER PHASE (Gatekeeper)
# # # # # # #     img_array = np.array(image.resize((224, 224))) / 255.0
# # # # # # #     route_preds = router.predict(np.expand_dims(img_array, axis=0), verbose=0)
# # # # # # #     route_idx = np.argmax(route_preds)
# # # # # # #     confidence_score = np.max(route_preds)
    
# # # # # # #     scan_map = {0: "Bone", 1: "Chest", 2: "Invalid"}
# # # # # # #     scan_type = scan_map.get(route_idx, "Invalid")
    
# # # # # # #     # --- INVALID IMAGE GUARDRAIL ---
# # # # # # #     if scan_type == "Invalid":
# # # # # # #         st.error(f"🚫 **Invalid Input Detected:** The system detected a non-medical image or low-quality scan. Analysis Halted.")
# # # # # # #         st.image(image, caption="Rejected Input", width=300)
# # # # # # #         st.stop()  # STOPS HERE. Does not proceed.

# # # # # # #     # If Valid, Proceed...
# # # # # # #     col1, col2 = st.columns([1, 1.4])
# # # # # # #     with col1:
# # # # # # #         st.success(f"✅ **Valid Scan:** Identified as {scan_type} X-Ray")
# # # # # # #         st.image(image, caption="Source Scan", use_container_width=True)

# # # # # # #     with col2:
# # # # # # #         with st.spinner(f"Initializing {scan_type} Diagnostic Protocol..."):
# # # # # # #             # 2. SPECIALIST & SPATIAL PHASE
# # # # # # #             ai_img, ai_conf, ai_loc, ai_label = run_unified_analysis(image, scan_type)
            
# # # # # # #             # 3. CLINICAL FUSION
# # # # # # #             # Unpack the 3 return values properly
# # # # # # #             boost, reason_list, reason_text = pre_symptoms_solver.calculate_clinical_boost(syms, scan_type)
            
# # # # # # #             # 4. FINAL PROBABILITY CALC
# # # # # # #             if ai_conf > 0:
# # # # # # #                 final_prob = min(ai_conf + boost, 1.0)
# # # # # # #             else:
# # # # # # #                 final_prob = 0.0 
            
# # # # # # #         # Display Result
# # # # # # #         st.image(ai_img, caption="AI Reasoning Overlay", use_container_width=True)
        
# # # # # # #         # Metrics Row
# # # # # # #         m1, m2, m3 = st.columns(3)
# # # # # # #         m1.metric("AI Confidence", f"{ai_conf*100:.1f}%", ai_loc if ai_conf > 0 else "Clean")
# # # # # # #         m2.metric("Symptom Boost", f"+{boost*100:.0f}%", f"{len(reason_list)} Factors")
# # # # # # #         m3.metric("Final Risk", f"{final_prob*100:.1f}%")

# # # # # # #     # 4. DIAGNOSTIC REPORT CARD
# # # # # # #     st.markdown("---")
    
# # # # # # #     # Dynamic Visual Text
# # # # # # #     if ai_conf > 0:
# # # # # # #         visual_finding = f"AI localized a <b>{ai_label}</b> pattern in the <b>{ai_loc}</b> with {ai_conf*100:.1f}% confidence."
# # # # # # #         final_css = "risk-high"
# # # # # # #         status = "HIGH RISK"
# # # # # # #     else:
# # # # # # #         visual_finding = "Radiological analysis is negative. No structural anomalies or lesions detected."
# # # # # # #         final_css = "risk-safe"
# # # # # # #         status = "LOW RISK"
        
# # # # # # #     # Override logic if symptoms are critical but scan is clean
# # # # # # #     if final_prob == 0.0 and boost > 0.3:
# # # # # # #         status = "CLINICAL WARNING"
# # # # # # #         final_css = "risk-high" # Make it red to warn doctor
# # # # # # #         visual_finding += " <br>⚠️ <b>Note:</b> Scan is clean, but severe symptoms suggest occult pathology."

# # # # # # #     st.markdown(f"""
# # # # # # #     <div class="report-box">
# # # # # # #         <div class="report-header">📋 Dayflow Diagnostic Reasoning</div>
# # # # # # #         <p><b>👁️ Visual Analysis:</b> {visual_finding}</p>
# # # # # # #         <p><b>🩺 Clinical Correlation:</b> {reason_text}</p>
# # # # # # #         <hr style="margin: 10px 0;">
# # # # # # #         <p style="margin-bottom:0px; font-size:14px; color:#555;">AGGREGATED DIAGNOSIS:</p>
# # # # # # #         <span class="{final_css}">{status} ({final_prob*100:.1f}%)</span>
# # # # # # #     </div>
# # # # # # #     """, unsafe_allow_html=True)








# # # # # # import streamlit as st
# # # # # # from PIL import Image, ImageDraw
# # # # # # import numpy as np
# # # # # # import tensorflow as tf
# # # # # # from ultralytics import YOLO
# # # # # # import pre_symptoms_solver  # <--- The new logic file above

# # # # # # # --- CONFIG ---
# # # # # # st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# # # # # # # --- CSS FOR "EXPLAINABILITY" ---
# # # # # # st.markdown("""
# # # # # #     <style>
# # # # # #     .reasoning-card {
# # # # # #         background-color: #f0f8ff;
# # # # # #         border-left: 5px solid #007bff;
# # # # # #         padding: 20px;
# # # # # #         border-radius: 5px;
# # # # # #         margin-top: 20px;
# # # # # #     }
# # # # # #     .highlight-text { font-weight: bold; color: #0056b3; }
# # # # # #     </style>
# # # # # # """, unsafe_allow_html=True)

# # # # # # # --- LOAD BRAINS ---
# # # # # # @st.cache_resource
# # # # # # def load_models():
# # # # # #     try:
# # # # # #         router = tf.keras.models.load_model('router_model.h5')
# # # # # #         bone_model = YOLO('bone_model.pt')
# # # # # #         lung_model = YOLO('lung_model.pt')
# # # # # #         return router, bone_model, lung_model
# # # # # #     except:
# # # # # #         return None, None, None

# # # # # # router, bone_model, lung_model = load_models()

# # # # # # # --- 1. CONTOUR/HIGHLIGHT VISUALIZER ---
# # # # # # def draw_smart_overlay(image, box, label):
# # # # # #     """
# # # # # #     Draws a semi-transparent 'Highlighter' effect instead of just a box.
# # # # # #     This helps non-experts see the AREA of concern.
# # # # # #     """
# # # # # #     overlay = image.copy().convert("RGBA")
# # # # # #     draw = ImageDraw.Draw(overlay)
    
# # # # # #     # Get coordinates
# # # # # #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# # # # # #     # Define Color (Red for TB/Fracture)
# # # # # #     fill_color = (255, 0, 0, 80)  # Red with 30% transparency
# # # # # #     outline_color = (255, 0, 0, 255) # Solid Red
    
# # # # # #     # Draw the "Contour" Box
# # # # # #     draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=4)
    
# # # # # #     # Blend back
# # # # # #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # # # # # # --- 2. SPATIAL LOCALIZER ---
# # # # # # def get_location_text(box, w, h, scan_type):
# # # # # #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
# # # # # #     if scan_type == "Chest":
# # # # # #         # Medical Zones
# # # # # #         if cy < h/3: zone = "Apical/Upper Zone"
# # # # # #         elif cy < 2*h/3: zone = "Middle Zone"
# # # # # #         else: zone = "Basal/Lower Zone"
# # # # # #         side = "Right" if cx < w/2 else "Left" # Radiology is flipped! (Patient's Right is Image Left)
# # # # # #         return f"{side} {zone}"
# # # # # #     else:
# # # # # #         # Bone Zones
# # # # # #         loc_y = "Distal" if cy > h/2 else "Proximal"
# # # # # #         return f"{loc_y} Shaft Region"

# # # # # # # --- MAIN APP ---
# # # # # # st.title("Dayflow: Explainable Medical AI")
# # # # # # st.markdown("**Mission:** Assisting healthcare workers with automated triage and transparent clinical reasoning.")

# # # # # # if not router:
# # # # # #     st.error("⚠️ Models offline. Check file paths.")
# # # # # #     st.stop()

# # # # # # # SIDEBAR
# # # # # # with st.sidebar:
# # # # # #     st.header("Patient Intake Form")
# # # # # #     scan_mode = st.radio("Expected Scan Type", ["Bone (Trauma)", "Chest (Respiratory)"])
    
# # # # # #     st.subheader("Clinical Signs")
# # # # # #     syms = {}
# # # # # #     if scan_mode == "Bone (Trauma)":
# # # # # #         syms['deformity'] = st.checkbox("Visible Deformity")
# # # # # #         syms['immobility'] = st.checkbox("Function Loss")
# # # # # #         syms['trauma'] = st.checkbox("Trauma History")
# # # # # #         syms['pain'] = st.checkbox("Severe Pain")
# # # # # #     else:
# # # # # #         syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
# # # # # #         syms['cough'] = st.checkbox("Chronic Cough")
# # # # # #         syms['weight_loss'] = st.checkbox("Weight Loss")
# # # # # #         syms['night_sweats'] = st.checkbox("Night Sweats")

# # # # # # # UPLOAD
# # # # # # uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png'])

# # # # # # if uploaded_file:
# # # # # #     image = Image.open(uploaded_file).convert('RGB')
# # # # # #     w, h = image.size
    
# # # # # #     # A. ROUTER CHECK
# # # # # #     img_arr = np.array(image.resize((224,224)))/255.0
# # # # # #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# # # # # #     detected_type = ["Bone", "Chest", "Invalid"][np.argmax(pred)]
    
# # # # # #     # Guardrail
# # # # # #     if detected_type == "Invalid":
# # # # # #         st.error("🚫 Image Rejected: Not a recognized medical scan.")
# # # # # #         st.stop()
        
# # # # # #     # B. AI DETECTION
# # # # # #     col1, col2 = st.columns(2)
# # # # # #     with col1:
# # # # # #         st.image(image, caption="Original Input", use_container_width=True)
        
# # # # # #     with col2:
# # # # # #         model = bone_model if detected_type == "Bone" else lung_model
# # # # # #         results = model.predict(image, conf=0.15, augment=True, verbose=False)
        
# # # # # #         ai_conf = 0.0
# # # # # #         ai_loc = "Unknown"
# # # # # #         final_img = image
        
# # # # # #         if len(results[0].boxes) > 0:
# # # # # #             box = results[0].boxes[0]
# # # # # #             ai_conf = float(box.conf[0])
# # # # # #             ai_loc = get_location_text(box, w, h, detected_type)
# # # # # #             # Use Smart Overlay instead of simple plot
# # # # # #             final_img = draw_smart_overlay(image, box, "Detection")
            
# # # # # #         st.image(final_img, caption="AI Enhanced View (Area of Interest)", use_container_width=True)

# # # # # #     # C. GENERATE REASONING
# # # # # #     boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
# # # # # #         syms, detected_type, ai_conf, ai_loc
# # # # # #     )
    
# # # # # #     final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
    
# # # # # #     # D. THE EXPLAINABILITY DASHBOARD
# # # # # #     st.markdown("---")
    
# # # # # #     # Dynamic Color for Risk
# # # # # #     risk_color = "#d32f2f" if final_prob > 0.6 else "#28a745"
# # # # # #     risk_label = "HIGH PRIORITY" if final_prob > 0.6 else "ROUTINE REVIEW"
    
# # # # # #     st.markdown(f"""
# # # # # #     <div class="reasoning-card">
# # # # # #         <h3 style="margin-top:0;">🤖 System Reasoning Report</h3>
# # # # # #         <p><b>1. Visual Analysis:</b> AI highlighted a region in the <span class="highlight-text">{ai_loc}</span>.</p>
# # # # # #         <p><b>2. Clinical Context:</b> {narrative}</p>
# # # # # #         <p><b>3. Why this decision?</b> The combination of visual opacity ({ai_conf*100:.0f}%) and {len(findings)} clinical risk factors ({', '.join(findings)}) increases the confidence of pathology.</p>
# # # # # #         <hr>
# # # # # #         <h2 style="color:{risk_color}; text-align:center;">{risk_label} ({final_prob*100:.1f}%)</h2>
# # # # # #         <p style="text-align:center; font-style:italic;">Recommended Action: {("Isolate Patient & Order Sputum Test" if "Chest" in detected_type and final_prob>0.6 else "Splint Limb & Order Ortho Consult")}</p>
# # # # # #     </div>
# # # # # #     """, unsafe_allow_html=True)















# # # # # import streamlit as st
# # # # # from PIL import Image, ImageDraw, ImageFont
# # # # # import numpy as np
# # # # # import tensorflow as tf
# # # # # from ultralytics import YOLO
# # # # # import pre_symptoms_solver  # Your logic file

# # # # # # --- CONFIG ---
# # # # # st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# # # # # # --- CSS FOR "EXPLAINABILITY" (FIXED BLACK TEXT) ---
# # # # # st.markdown("""
# # # # #     <style>
# # # # #     .reasoning-card {
# # # # #         background-color: #f8f9fa; /* Light Grey/White Background */
# # # # #         border-left: 6px solid #007bff;
# # # # #         padding: 25px;
# # # # #         border-radius: 8px;
# # # # #         margin-top: 20px;
# # # # #         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
# # # # #     }
# # # # #     /* FORCE BLACK TEXT */
# # # # #     .reasoning-card h3 { color: #000000 !important; margin-top: 0; }
# # # # #     .reasoning-card p { color: #333333 !important; font-size: 16px; line-height: 1.5; }
# # # # #     .reasoning-card li { color: #333333 !important; }
# # # # #     .highlight-text { font-weight: bold; color: #0056b3 !important; }
    
# # # # #     /* Metrics Fix */
# # # # #     div[data-testid="stMetricValue"] { color: #000 !important; }
# # # # #     </style>
# # # # # """, unsafe_allow_html=True)

# # # # # # --- LOAD BRAINS ---
# # # # # @st.cache_resource
# # # # # def load_models():
# # # # #     try:
# # # # #         router = tf.keras.models.load_model('router_model.h5')
# # # # #         bone_model = YOLO('bone_model.pt')
# # # # #         lung_model = YOLO('lung_model.pt')
# # # # #         return router, bone_model, lung_model
# # # # #     except:
# # # # #         return None, None, None

# # # # # router, bone_model, lung_model = load_models()

# # # # # # --- 1. CONTOUR/HIGHLIGHT VISUALIZER (WITH LABELS) ---
# # # # # def draw_smart_overlay(image, box, label, conf):
# # # # #     """
# # # # #     Draws a semi-transparent 'Highlighter' effect + Text Label.
# # # # #     """
# # # # #     # 1. Setup Overlay
# # # # #     overlay = image.copy().convert("RGBA")
# # # # #     draw = ImageDraw.Draw(overlay)
    
# # # # #     # 2. Get Coordinates
# # # # #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# # # # #     # 3. Define Colors (Red for Danger)
# # # # #     fill_color = (255, 0, 0, 70)   # Red with ~25% transparency
# # # # #     outline_color = (255, 0, 0, 200) # Solid Red
# # # # #     text_bg_color = (255, 0, 0, 255) # Solid Red Background for text
    
# # # # #     # 4. Draw the Highlight Box
# # # # #     draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=4)
    
# # # # #     # 5. Draw the Label Tag
# # # # #     text_str = f"{label}: {conf*100:.1f}%"
    
# # # # #     # Simple default font handling
# # # # #     try:
# # # # #         font = ImageFont.truetype("arial.ttf", 20)
# # # # #     except:
# # # # #         font = ImageFont.load_default()
    
# # # # #     # Calculate text background size
# # # # #     text_bbox = draw.textbbox((x1, y1), text_str, font=font)
# # # # #     text_w = text_bbox[2] - text_bbox[0] + 10
# # # # #     text_h = text_bbox[3] - text_bbox[1] + 10
    
# # # # #     # Draw text background rectangle (above the box)
# # # # #     draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=text_bg_color)
    
# # # # #     # Draw White Text
# # # # #     draw.text((x1 + 5, y1 - text_h + 5), text_str, fill="white", font=font)
    
# # # # #     # 6. Blend back
# # # # #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # # # # # --- 2. SPATIAL LOCALIZER ---
# # # # # def get_location_text(box, w, h, scan_type):
# # # # #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
# # # # #     if scan_type == "Chest":
# # # # #         if cy < h/3: zone = "Apical/Upper Zone"
# # # # #         elif cy < 2*h/3: zone = "Middle Zone"
# # # # #         else: zone = "Basal/Lower Zone"
# # # # #         side = "Right" if cx < w/2 else "Left" 
# # # # #         return f"{side} {zone}"
# # # # #     else:
# # # # #         loc_y = "Distal" if cy > h/2 else "Proximal"
# # # # #         side = "Right" if cx > w/2 else "Left"
# # # # #         return f"{loc_y} {side} Region"

# # # # # # --- MAIN APP ---
# # # # # st.title("Dayflow: Explainable Medical AI")
# # # # # st.markdown("**Mission:** Assisting healthcare workers with automated triage and transparent clinical reasoning.")

# # # # # if not router:
# # # # #     st.error("⚠️ Models offline. Please check file paths.")
# # # # #     st.stop()

# # # # # # --- SIDEBAR (DYNAMIC SYMPTOMS) ---
# # # # # with st.sidebar:
# # # # #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# # # # #     st.header("Patient Intake")
    
# # # # #     # Placeholder for dynamic symptoms (will update after upload)
# # # # #     sym_placeholder = st.empty()
# # # # #     st.info("Upload a scan to activate the specific symptom checklist.")

# # # # # # --- UPLOAD & ROUTING ---
# # # # # uploaded_file = st.file_uploader("Upload X-Ray (Chest or Bone)", type=['jpg', 'png', 'jpeg'])

# # # # # if uploaded_file:
# # # # #     image = Image.open(uploaded_file).convert('RGB')
# # # # #     w, h = image.size
    
# # # # #     # A. ROUTER PHASE (Automatic)
# # # # #     img_arr = np.array(image.resize((224,224)))/255.0
# # # # #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# # # # #     route_idx = np.argmax(pred)
# # # # #     detected_type = ["Bone", "Chest", "Invalid"][route_idx]
    
# # # # #     # Guardrail
# # # # #     if detected_type == "Invalid":
# # # # #         st.error("🚫 Image Rejected: Not a recognized medical scan.")
# # # # #         st.stop()
        
# # # # #     st.success(f"✅ **Auto-Detected:** {detected_type} Scan")

# # # # #     # --- B. DYNAMIC SIDEBAR (Now that we know the type) ---
# # # # #     syms = {}
# # # # #     with sym_placeholder.container():
# # # # #         st.subheader(f"{detected_type} Clinical Signs")
# # # # #         if detected_type == "Bone":
# # # # #             syms['deformity'] = st.checkbox("Visible Deformity")
# # # # #             syms['immobility'] = st.checkbox("Function Loss")
# # # # #             syms['trauma'] = st.checkbox("Trauma History")
# # # # #             syms['pain'] = st.checkbox("Severe Pain")
# # # # #         else:
# # # # #             syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
# # # # #             syms['cough'] = st.checkbox("Chronic Cough")
# # # # #             syms['weight_loss'] = st.checkbox("Weight Loss")
# # # # #             syms['night_sweats'] = st.checkbox("Night Sweats")

# # # # #     # --- C. SPECIALIST ANALYSIS ---
# # # # #     col1, col2 = st.columns(2)
# # # # #     with col1:
# # # # #         st.image(image, caption="Original Input", use_container_width=True)
        
# # # # #     with col2:
# # # # #         with st.spinner(f"Scanning for {detected_type} anomalies..."):
# # # # #             model = bone_model if detected_type == "Bone" else lung_model
# # # # #             results = model.predict(image, conf=0.15, augment=True, verbose=False)
            
# # # # #             ai_conf = 0.0
# # # # #             ai_loc = "Unknown"
# # # # #             ai_label = "Anomaly"
# # # # #             final_img = image
            
# # # # #             if len(results[0].boxes) > 0:
# # # # #                 box = results[0].boxes[0]
# # # # #                 ai_conf = float(box.conf[0])
# # # # #                 ai_loc = get_location_text(box, w, h, detected_type)
# # # # #                 ai_label = model.names[int(box.cls[0])]
                
# # # # #                 # DRAW THE SMART OVERLAY (Contour + Text)
# # # # #                 final_img = draw_smart_overlay(image, box, ai_label, ai_conf)
                
# # # # #             st.image(final_img, caption="AI Enhanced View (Contour Map)", use_container_width=True)

# # # # #     # --- D. GENERATE REASONING ---
# # # # #     boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
# # # # #         syms, detected_type, ai_conf, ai_loc
# # # # #     )
    
# # # # #     final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
    
# # # # #     # --- E. EXPLAINABILITY DASHBOARD (BLACK TEXT) ---
# # # # #     st.markdown("---")
    
# # # # #     # Dynamic Color for Risk Header
# # # # #     risk_color = "#d32f2f" if final_prob > 0.6 else "#28a745"
# # # # #     risk_label = "HIGH PRIORITY" if final_prob > 0.6 else "ROUTINE REVIEW"
    
# # # # #     st.markdown(f"""
# # # # #     <div class="reasoning-card">
# # # # #         <h3>🤖 System Reasoning Report</h3>
# # # # #         <p><b>1. Visual Analysis:</b> AI highlighted a specific region in the <span class="highlight-text">{ai_loc}</span>.</p>
# # # # #         <p><b>2. Clinical Context:</b> {narrative}</p>
# # # # #         <p><b>3. Why this decision?</b> The system correlated the visual opacity ({ai_conf*100:.0f}%) with {len(findings)} clinical risk factor(s) ({', '.join(findings) if findings else 'None'}). This multi-modal data increases the diagnostic confidence.</p>
# # # # #         <hr style="border-top: 1px solid #ccc;">
# # # # #         <h2 style="color:{risk_color} !important; text-align:center; margin-bottom:5px;">{risk_label} ({final_prob*100:.1f}%)</h2>
# # # # #         <p style="text-align:center; font-style:italic; font-weight:bold;">Recommended Action: {("Isolate Patient & Order Sputum Test" if "Chest" in detected_type and final_prob>0.6 else "Splint Limb & Order Ortho Consult")}</p>
# # # # #     </div>
# # # # #     """, unsafe_allow_html=True)








# # # # import streamlit as st
# # # # from PIL import Image, ImageDraw, ImageFont
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from ultralytics import YOLO
# # # # import pre_symptoms_solver  # Your logic file

# # # # # --- CONFIG ---
# # # # st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# # # # # --- CSS (BLACK TEXT FIX) ---
# # # # st.markdown("""
# # # #     <style>
# # # #     .reasoning-card {
# # # #         background-color: #f8f9fa;
# # # #         border-left: 6px solid #007bff;
# # # #         padding: 25px;
# # # #         border-radius: 8px;
# # # #         margin-top: 20px;
# # # #         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
# # # #     }
# # # #     .reasoning-card h3 { color: #000000 !important; margin-top: 0; }
# # # #     .reasoning-card p { color: #333333 !important; font-size: 16px; line-height: 1.5; }
# # # #     .reasoning-card li { color: #333333 !important; }
# # # #     .highlight-text { font-weight: bold; color: #0056b3 !important; }
# # # #     div[data-testid="stMetricValue"] { color: #000 !important; }
# # # #     </style>
# # # # """, unsafe_allow_html=True)

# # # # # --- LOAD BRAINS ---
# # # # @st.cache_resource
# # # # def load_models():
# # # #     try:
# # # #         router = tf.keras.models.load_model('router_model.h5')
# # # #         bone_model = YOLO('bone_model.pt')
# # # #         lung_model = YOLO('lung_model.pt')
# # # #         return router, bone_model, lung_model
# # # #     except:
# # # #         return None, None, None

# # # # router, bone_model, lung_model = load_models()

# # # # # --- 1. SMART OVERLAY (BLUE FOR LATENT, RED FOR ACTIVE) ---
# # # # def draw_smart_overlay(image, box, label, conf, risk_status):
# # # #     """
# # # #     Draws the contour. 
# # # #     Color changes based on Risk Status (Red = Active, Blue = Latent).
# # # #     """
# # # #     overlay = image.copy().convert("RGBA")
# # # #     draw = ImageDraw.Draw(overlay)
# # # #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# # # #     # --- COLOR LOGIC ---
# # # #     if risk_status == "HIGH PRIORITY":
# # # #         # RED (Active TB / Fracture)
# # # #         fill_color = (255, 0, 0, 70)
# # # #         outline_color = (255, 0, 0, 200)
# # # #         text_bg = (255, 0, 0, 255)
# # # #         tag_text = f"ACTIVE: {conf*100:.1f}%"
# # # #     else:
# # # #         # BLUE (Latent TB / Hairline)
# # # #         fill_color = (0, 120, 255, 70)   # Blue with transparency
# # # #         outline_color = (0, 120, 255, 200)
# # # #         text_bg = (0, 120, 255, 255)
# # # #         tag_text = f"LATENT/EARLY: {conf*100:.1f}%"
    
# # # #     # Draw Box
# # # #     draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=4)
    
# # # #     # Draw Label
# # # #     try: font = ImageFont.truetype("arial.ttf", 20)
# # # #     except: font = ImageFont.load_default()
    
# # # #     text_bbox = draw.textbbox((x1, y1), tag_text, font=font)
# # # #     text_w = text_bbox[2] - text_bbox[0] + 10
# # # #     text_h = text_bbox[3] - text_bbox[1] + 10
    
# # # #     draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=text_bg)
# # # #     draw.text((x1 + 5, y1 - text_h + 5), tag_text, fill="white", font=font)
    
# # # #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # # # # --- 2. SPATIAL LOCALIZER ---
# # # # def get_location_text(box, w, h, scan_type):
# # # #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
# # # #     if scan_type == "Chest":
# # # #         if cy < h/3: zone = "Apical/Upper Zone"
# # # #         elif cy < 2*h/3: zone = "Middle Zone"
# # # #         else: zone = "Basal/Lower Zone"
# # # #         side = "Right" if cx < w/2 else "Left" 
# # # #         return f"{side} {zone}"
# # # #     else:
# # # #         loc_y = "Distal" if cy > h/2 else "Proximal"
# # # #         side = "Right" if cx > w/2 else "Left"
# # # #         return f"{loc_y} {side} Region"

# # # # # --- MAIN APP ---
# # # # st.title("Dayflow: Explainable Medical AI")
# # # # st.markdown("**Mission:** Assisting healthcare workers with automated triage and transparent clinical reasoning.")

# # # # if not router:
# # # #     st.error("⚠️ Models offline. Please check file paths.")
# # # #     st.stop()

# # # # # --- SIDEBAR ---
# # # # with st.sidebar:
# # # #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# # # #     st.header("Patient Intake")
# # # #     sym_placeholder = st.empty()
# # # #     st.info("Upload a scan to activate the specific symptom checklist.")

# # # # # --- UPLOAD ---
# # # # uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png', 'jpeg'])

# # # # if uploaded_file:
# # # #     image = Image.open(uploaded_file).convert('RGB')
# # # #     w, h = image.size
    
# # # #     # A. ROUTER
# # # #     img_arr = np.array(image.resize((224,224)))/255.0
# # # #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# # # #     detected_type = ["Bone", "Chest", "Invalid"][np.argmax(pred)]
    
# # # #     if detected_type == "Invalid":
# # # #         st.error("🚫 Image Rejected: Not a recognized medical scan.")
# # # #         st.stop()
        
# # # #     st.success(f"✅ **Auto-Detected:** {detected_type} Scan")

# # # #     # B. DYNAMIC SYMPTOMS
# # # #     syms = {}
# # # #     with sym_placeholder.container():
# # # #         st.subheader(f"{detected_type} Clinical Signs")
# # # #         if detected_type == "Bone":
# # # #             syms['deformity'] = st.checkbox("Visible Deformity")
# # # #             syms['immobility'] = st.checkbox("Function Loss")
# # # #             syms['trauma'] = st.checkbox("Trauma History")
# # # #             syms['pain'] = st.checkbox("Severe Pain")
# # # #         else:
# # # #             syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
# # # #             syms['cough'] = st.checkbox("Chronic Cough")
# # # #             syms['weight_loss'] = st.checkbox("Weight Loss")
# # # #             syms['night_sweats'] = st.checkbox("Night Sweats")

# # # #     # C. PRE-CALCULATE RISK (To determine Color)
# # # #     # We need to run inference first, then calculate risk, THEN draw the image.
    
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         st.image(image, caption="Original Input", use_container_width=True)
        
# # # #     with col2:
# # # #         with st.spinner(f"Scanning for {detected_type} anomalies..."):
# # # #             model = bone_model if detected_type == "Bone" else lung_model
# # # #             results = model.predict(image, conf=0.15, augment=True, verbose=False)
            
# # # #             ai_conf = 0.0
# # # #             ai_loc = "Unknown"
# # # #             ai_label = "Anomaly"
            
# # # #             # 1. Get AI Confidence First
# # # #             if len(results[0].boxes) > 0:
# # # #                 box = results[0].boxes[0]
# # # #                 ai_conf = float(box.conf[0])
# # # #                 ai_loc = get_location_text(box, w, h, detected_type)
# # # #                 ai_label = model.names[int(box.cls[0])]

# # # #             # 2. Calculate Final Probability (Fusion)
# # # #             boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
# # # #                 syms, detected_type, ai_conf, ai_loc
# # # #             )
# # # #             final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
            
# # # #             # 3. Determine Status/Color
# # # #             risk_label = "HIGH PRIORITY" if final_prob > 0.60 else "ROUTINE REVIEW"
# # # #             risk_color = "#d32f2f" if final_prob > 0.60 else "#0078ff" # Red vs Blue
            
# # # #             # 4. Draw Overlay (Now utilizing the correct color)
# # # #             final_img = image
# # # #             if ai_conf > 0:
# # # #                 final_img = draw_smart_overlay(image, box, ai_label, ai_conf, risk_label)

# # # #             st.image(final_img, caption="AI Enhanced View (Contour Map)", use_container_width=True)

# # # #     # E. EXPLAINABILITY DASHBOARD
# # # #     st.markdown("---")
    
# # # #     st.markdown(f"""
# # # #     <div class="reasoning-card">
# # # #         <h3>🤖 System Reasoning Report</h3>
# # # #         <p><b>1. Visual Analysis:</b> AI highlighted a specific region in the <span class="highlight-text">{ai_loc}</span>.</p>
# # # #         <p><b>2. Clinical Context:</b> {narrative}</p>
# # # #         <p><b>3. Why this decision?</b> The system correlated the visual opacity ({ai_conf*100:.0f}%) with {len(findings)} clinical risk factor(s). {'The low probability suggests Latent/Early stage.' if final_prob <= 0.6 else 'The high probability indicates Active pathology.'}</p>
# # # #         <hr style="border-top: 1px solid #ccc;">
# # # #         <h2 style="color:{risk_color} !important; text-align:center; margin-bottom:5px;">{risk_label} ({final_prob*100:.1f}%)</h2>
# # # #         <p style="text-align:center; font-style:italic; font-weight:bold;">Recommended Action: {("Isolate Patient" if "Chest" in detected_type and final_prob>0.6 else "Observation / Follow-up X-ray in 2 weeks")}</p>
# # # #     </div>
# # # #     """, unsafe_allow_html=True)







# # # # import streamlit as st
# # # # from PIL import Image, ImageDraw, ImageFont
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from ultralytics import YOLO
# # # # import pre_symptoms_solver  # Ensure this file is in the same folder

# # # # # --- CONFIGURATION ---
# # # # st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# # # # # --- CUSTOM CSS (High Contrast & Black Text) ---
# # # # st.markdown("""
# # # #     <style>
# # # #     .reasoning-card {
# # # #         background-color: #f8f9fa; /* Light Grey Background */
# # # #         border-left: 6px solid #007bff;
# # # #         padding: 25px;
# # # #         border-radius: 8px;
# # # #         margin-top: 20px;
# # # #         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
# # # #     }
# # # #     /* FORCE BLACK TEXT */
# # # #     .reasoning-card h3 { color: #000000 !important; margin-top: 0; }
# # # #     .reasoning-card p { color: #333333 !important; font-size: 16px; line-height: 1.5; }
# # # #     .reasoning-card li { color: #333333 !important; }
# # # #     .highlight-text { font-weight: bold; color: #0056b3 !important; }
    
# # # #     /* Metrics Text Fix */
# # # #     div[data-testid="stMetricValue"] { color: #000 !important; }
# # # #     div[data-testid="stMetricLabel"] { color: #333 !important; }
# # # #     </style>
# # # # """, unsafe_allow_html=True)

# # # # # --- LOAD MODELS ---
# # # # @st.cache_resource
# # # # def load_models():
# # # #     try:
# # # #         router = tf.keras.models.load_model('router_model.h5')
# # # #         bone_model = YOLO('bone_model.pt')
# # # #         lung_model = YOLO('lung_model.pt')
# # # #         return router, bone_model, lung_model
# # # #     except Exception as e:
# # # #         return None, None, None

# # # # router, bone_model, lung_model = load_models()

# # # # # --- 1. SMART OVERLAY (Blue/Red Logic) ---
# # # # def draw_smart_overlay(image, box, label, conf, risk_status):
# # # #     """
# # # #     Draws a contour. Color changes based on Risk Status.
# # # #     RED = Active/High Priority
# # # #     BLUE = Latent/Routine Review
# # # #     """
# # # #     overlay = image.copy().convert("RGBA")
# # # #     draw = ImageDraw.Draw(overlay)
# # # #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# # # #     # --- COLOR LOGIC ---
# # # #     if risk_status == "HIGH PRIORITY":
# # # #         # RED (Active TB / Fracture)
# # # #         fill_color = (255, 0, 0, 70)     # Red with transparency
# # # #         outline_color = (255, 0, 0, 200) # Solid Red
# # # #         text_bg = (255, 0, 0, 255)       # Solid Red for text
# # # #         tag_text = f"ACTIVE: {conf*100:.1f}%"
# # # #     else:
# # # #         # BLUE (Latent TB / Hairline)
# # # #         fill_color = (0, 120, 255, 70)   # Blue with transparency
# # # #         outline_color = (0, 120, 255, 200)
# # # #         text_bg = (0, 120, 255, 255)
# # # #         tag_text = f"LATENT/EARLY: {conf*100:.1f}%"
    
# # # #     # Draw The Box
# # # #     draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=4)
    
# # # #     # Draw The Text Label
# # # #     try: font = ImageFont.truetype("arial.ttf", 20)
# # # #     except: font = ImageFont.load_default()
    
# # # #     text_bbox = draw.textbbox((x1, y1), tag_text, font=font)
# # # #     text_w = text_bbox[2] - text_bbox[0] + 10
# # # #     text_h = text_bbox[3] - text_bbox[1] + 10
    
# # # #     # Draw text background rectangle (above the box)
# # # #     draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=text_bg)
# # # #     # Draw White Text
# # # #     draw.text((x1 + 5, y1 - text_h + 5), tag_text, fill="white", font=font)
    
# # # #     # Blend back
# # # #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # # # # --- 2. SPATIAL LOCALIZER ---
# # # # def get_location_text(box, w, h, scan_type):
# # # #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
# # # #     if scan_type == "Chest":
# # # #         if cy < h/3: zone = "Apical/Upper Zone"
# # # #         elif cy < 2*h/3: zone = "Middle Zone"
# # # #         else: zone = "Basal/Lower Zone"
# # # #         side = "Right" if cx < w/2 else "Left" 
# # # #         return f"{side} {zone}"
# # # #     else:
# # # #         loc_y = "Distal" if cy > h/2 else "Proximal"
# # # #         side = "Right" if cx > w/2 else "Left"
# # # #         return f"{loc_y} {side} Region"

# # # # # --- MAIN APP UI ---
# # # # st.title("Dayflow: Explainable Medical AI")
# # # # st.markdown("**Mission:** Assisting healthcare workers with automated triage and transparent clinical reasoning.")

# # # # if not router:
# # # #     st.error("⚠️ Models offline. Please check file paths.")
# # # #     st.stop()

# # # # # --- SIDEBAR (Dynamic Symptoms) ---
# # # # with st.sidebar:
# # # #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# # # #     st.header("Patient Intake")
# # # #     sym_placeholder = st.empty()
# # # #     st.info("Upload a scan to activate the specific symptom checklist.")

# # # # # --- UPLOAD & ROUTING ---
# # # # uploaded_file = st.file_uploader("Upload X-Ray (Chest or Bone)", type=['jpg', 'png', 'jpeg'])

# # # # if uploaded_file:
# # # #     image = Image.open(uploaded_file).convert('RGB')
# # # #     w, h = image.size
    
# # # #     # A. ROUTER PHASE (Automatic)
# # # #     img_arr = np.array(image.resize((224,224)))/255.0
# # # #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# # # #     route_idx = np.argmax(pred)
# # # #     detected_type = ["Bone", "Chest", "Invalid"][route_idx]
    
# # # #     if detected_type == "Invalid":
# # # #         st.error("🚫 Image Rejected: Not a recognized medical scan.")
# # # #         st.stop()
        
# # # #     st.success(f"✅ **Auto-Detected:** {detected_type} Scan")

# # # #     # B. DYNAMIC SYMPTOMS (Based on Type)
# # # #     syms = {}
# # # #     with sym_placeholder.container():
# # # #         st.subheader(f"{detected_type} Clinical Signs")
# # # #         if detected_type == "Bone":
# # # #             syms['deformity'] = st.checkbox("Visible Deformity")
# # # #             syms['immobility'] = st.checkbox("Function Loss")
# # # #             syms['trauma'] = st.checkbox("Trauma History")
# # # #             syms['pain'] = st.checkbox("Severe Pain")
# # # #         else:
# # # #             syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
# # # #             syms['cough'] = st.checkbox("Chronic Cough")
# # # #             syms['weight_loss'] = st.checkbox("Weight Loss")
# # # #             syms['night_sweats'] = st.checkbox("Night Sweats")

# # # #     # --- C. PRE-CALCULATE RISK (To determine Color first) ---
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         st.image(image, caption="Original Input", use_container_width=True)
        
# # # #     with col2:
# # # #         with st.spinner(f"Scanning for {detected_type} anomalies..."):
# # # #             model = bone_model if detected_type == "Bone" else lung_model
# # # #             results = model.predict(image, conf=0.15, augment=True, verbose=False)
            
# # # #             ai_conf = 0.0
# # # #             ai_loc = "Unknown"
# # # #             ai_label = "Anomaly"
            
# # # #             # 1. Get AI Confidence
# # # #             if len(results[0].boxes) > 0:
# # # #                 box = results[0].boxes[0]
# # # #                 ai_conf = float(box.conf[0])
# # # #                 ai_loc = get_location_text(box, w, h, detected_type)
# # # #                 ai_label = model.names[int(box.cls[0])]

# # # #             # 2. Calculate Final Probability (Fusion)
# # # #             boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
# # # #                 syms, detected_type, ai_conf, ai_loc
# # # #             )
# # # #             final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
            
# # # #             # 3. Determine Status/Color (Dark Red vs Dark Blue)
# # # #             if final_prob > 0.60:
# # # #                 risk_label = "HIGH PRIORITY"
# # # #                 risk_color = "#cc0000"  # Dark Red
# # # #                 action_text = "Isolate Patient & Order Sputum Test" if "Chest" in detected_type else "Splint Limb & Order Ortho Consult"
# # # #             else:
# # # #                 risk_label = "ROUTINE REVIEW"
# # # #                 risk_color = "#004085"  # Dark Navy Blue
# # # #                 action_text = "Observation / Follow-up X-ray in 2 weeks"
            
# # # #             # 4. Draw Overlay (With correct color)
# # # #             final_img = image
# # # #             if ai_conf > 0:
# # # #                 final_img = draw_smart_overlay(image, box, ai_label, ai_conf, risk_label)

# # # #             st.image(final_img, caption="AI Enhanced View (Contour Map)", use_container_width=True)

# # # #     # --- D. EXPLAINABILITY DASHBOARD (FIXED VISIBILITY) ---
# # # #     st.markdown("---")

# # # #     st.markdown(f"""
# # # #     <div class="reasoning-card">
# # # #         <h3>🤖 System Reasoning Report</h3>
# # # #         <p><b>1. Visual Analysis:</b> AI highlighted a specific region in the <span class="highlight-text">{ai_loc}</span>.</p>
# # # #         <p><b>2. Clinical Context:</b> {narrative}</p>
# # # #         <p><b>3. Why this decision?</b> The system correlated the visual opacity ({ai_conf*100:.0f}%) with {len(findings)} clinical risk factor(s). {'The low probability suggests Latent/Early stage.' if final_prob <= 0.6 else 'The high probability indicates Active pathology.'}</p>
# # # #         <hr style="border-top: 1px solid #ccc;">
        
# # # #         <h2 style="color:{risk_color} !important; text-align:center; margin-bottom:5px; font-weight: 900; text-transform: uppercase;">
# # # #             {risk_label} ({final_prob*100:.1f}%)
# # # #         </h2>
        
# # # #         <p style="text-align:center; font-style:italic; font-weight:bold; color: #333;">
# # # #             Recommended Action: {action_text}
# # # #         </p>
# # # #     </div>
# # # #     """, unsafe_allow_html=True)














# # # import streamlit as st
# # # from PIL import Image, ImageDraw, ImageFont
# # # import numpy as np
# # # import tensorflow as tf
# # # from ultralytics import YOLO
# # # import pre_symptoms_solver  # Ensure this file is in the same folder

# # # # --- CONFIGURATION ---
# # # st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# # # # --- CUSTOM CSS (High Contrast & Black Text) ---
# # # st.markdown("""
# # #     <style>
# # #     /* Main Report Card Styling */
# # #     .reasoning-card {
# # #         background-color: #f8f9fa; 
# # #         border-left: 6px solid #007bff;
# # #         padding: 25px;
# # #         border-radius: 8px;
# # #         margin-top: 20px;
# # #         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
# # #     }
    
# # #     /* FORCE BLACK TEXT GLOBAL OVERRIDE */
# # #     .reasoning-card h3 { color: #000000 !important; margin-top: 0; }
# # #     .reasoning-card p { color: #333333 !important; font-size: 16px; line-height: 1.5; }
# # #     .reasoning-card li { color: #333333 !important; }
# # #     .highlight-text { font-weight: bold; color: #0056b3 !important; }
    
# # #     /* Streamlit Metric Fixes */
# # #     div[data-testid="stMetricValue"] { color: #000 !important; }
# # #     div[data-testid="stMetricLabel"] { color: #333 !important; }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # # --- LOAD MODELS ---
# # # @st.cache_resource
# # # def load_models():
# # #     try:
# # #         router = tf.keras.models.load_model('router_model.h5')
# # #         bone_model = YOLO('bone_model.pt')
# # #         lung_model = YOLO('lung_model.pt')
# # #         return router, bone_model, lung_model
# # #     except Exception as e:
# # #         return None, None, None

# # # router, bone_model, lung_model = load_models()

# # # # --- 1. SMART OVERLAY (Blue/Red Logic) ---
# # # def draw_smart_overlay(image, box, label, conf, risk_status):
# # #     """
# # #     Draws a contour. Color changes based on Risk Status.
# # #     RED = Active/High Priority
# # #     BLUE = Latent/Routine Review
# # #     """
# # #     overlay = image.copy().convert("RGBA")
# # #     draw = ImageDraw.Draw(overlay)
# # #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# # #     # --- COLOR LOGIC ---
# # #     if risk_status == "HIGH PRIORITY":
# # #         # RED (Active TB / Fracture)
# # #         fill_color = (255, 0, 0, 70)     # Red with transparency
# # #         outline_color = (255, 0, 0, 200) # Solid Red
# # #         text_bg = (255, 0, 0, 255)       # Solid Red for text
# # #         tag_text = f"ACTIVE: {conf*100:.1f}%"
# # #     else:
# # #         # BLUE (Latent TB / Hairline)
# # #         fill_color = (0, 120, 255, 70)   # Blue with transparency
# # #         outline_color = (0, 120, 255, 200)
# # #         text_bg = (0, 120, 255, 255)
# # #         tag_text = f"LATENT/EARLY: {conf*100:.1f}%"
    
# # #     # Draw The Box
# # #     draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=4)
    
# # #     # Draw The Text Label
# # #     try: font = ImageFont.truetype("arial.ttf", 20)
# # #     except: font = ImageFont.load_default()
    
# # #     text_bbox = draw.textbbox((x1, y1), tag_text, font=font)
# # #     text_w = text_bbox[2] - text_bbox[0] + 10
# # #     text_h = text_bbox[3] - text_bbox[1] + 10
    
# # #     # Draw text background rectangle (above the box)
# # #     draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=text_bg)
# # #     # Draw White Text
# # #     draw.text((x1 + 5, y1 - text_h + 5), tag_text, fill="white", font=font)
    
# # #     # Blend back
# # #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # # # --- 2. SPATIAL LOCALIZER ---
# # # def get_location_text(box, w, h, scan_type):
# # #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
# # #     if scan_type == "Chest":
# # #         if cy < h/3: zone = "Apical/Upper Zone"
# # #         elif cy < 2*h/3: zone = "Middle Zone"
# # #         else: zone = "Basal/Lower Zone"
# # #         side = "Right" if cx < w/2 else "Left" 
# # #         return f"{side} {zone}"
# # #     else:
# # #         loc_y = "Distal" if cy > h/2 else "Proximal"
# # #         side = "Right" if cx > w/2 else "Left"
# # #         return f"{loc_y} {side} Region"

# # # # --- MAIN APP UI ---
# # # st.title("Dayflow: Explainable Medical AI")
# # # st.markdown("**Mission:** Assisting healthcare workers with automated triage and transparent clinical reasoning.")

# # # if not router:
# # #     st.error("⚠️ Models offline. Please check file paths.")
# # #     st.stop()

# # # # --- SIDEBAR (Dynamic Symptoms) ---
# # # with st.sidebar:
# # #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# # #     st.header("Patient Intake")
# # #     sym_placeholder = st.empty()
# # #     st.info("Upload a scan to activate the specific symptom checklist.")

# # # # --- UPLOAD & ROUTING ---
# # # uploaded_file = st.file_uploader("Upload X-Ray (Chest or Bone)", type=['jpg', 'png', 'jpeg'])

# # # if uploaded_file:
# # #     image = Image.open(uploaded_file).convert('RGB')
# # #     w, h = image.size
    
# # #     # A. ROUTER PHASE (Automatic)
# # #     img_arr = np.array(image.resize((224,224)))/255.0
# # #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# # #     route_idx = np.argmax(pred)
# # #     detected_type = ["Bone", "Chest", "Invalid"][route_idx]
    
# # #     if detected_type == "Invalid":
# # #         st.error("🚫 Image Rejected: Not a recognized medical scan.")
# # #         st.stop()
        
# # #     st.success(f"✅ **Auto-Detected:** {detected_type} Scan")

# # #     # B. DYNAMIC SYMPTOMS (Based on Type)
# # #     syms = {}
# # #     with sym_placeholder.container():
# # #         st.subheader(f"{detected_type} Clinical Signs")
# # #         if detected_type == "Bone":
# # #             syms['deformity'] = st.checkbox("Visible Deformity")
# # #             syms['immobility'] = st.checkbox("Function Loss")
# # #             syms['trauma'] = st.checkbox("Trauma History")
# # #             syms['pain'] = st.checkbox("Severe Pain")
# # #         else:
# # #             syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
# # #             syms['cough'] = st.checkbox("Chronic Cough")
# # #             syms['weight_loss'] = st.checkbox("Weight Loss")
# # #             syms['night_sweats'] = st.checkbox("Night Sweats")

# # #     # --- C. PRE-CALCULATE RISK ---
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         st.image(image, caption="Original Input", use_container_width=True)
        
# # #     with col2:
# # #         with st.spinner(f"Scanning for {detected_type} anomalies..."):
# # #             model = bone_model if detected_type == "Bone" else lung_model
# # #             results = model.predict(image, conf=0.15, augment=True, verbose=False)
            
# # #             ai_conf = 0.0
# # #             ai_loc = "Unknown"
# # #             ai_label = "Anomaly"
            
# # #             # 1. Get AI Confidence
# # #             if len(results[0].boxes) > 0:
# # #                 box = results[0].boxes[0]
# # #                 ai_conf = float(box.conf[0])
# # #                 ai_loc = get_location_text(box, w, h, detected_type)
# # #                 ai_label = model.names[int(box.cls[0])]

# # #             # 2. Calculate Final Probability (Fusion)
# # #             boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
# # #                 syms, detected_type, ai_conf, ai_loc
# # #             )
# # #             final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
            
# # #             # 3. Determine Status/Color (Dark Red vs Dark Blue)
# # #             if final_prob > 0.60:
# # #                 risk_label = "HIGH PRIORITY"
# # #                 risk_color = "#cc0000"  # Dark Red
# # #                 bg_color = "#ffe6e6"    # Light Red Background
# # #                 action_text = "Isolate Patient & Order Sputum Test" if "Chest" in detected_type else "Splint Limb & Order Ortho Consult"
# # #             else:
# # #                 risk_label = "ROUTINE REVIEW"
# # #                 risk_color = "#004085"  # Dark Navy Blue
# # #                 bg_color = "#e6f0ff"    # Light Blue Background
# # #                 action_text = "Observation / Follow-up X-ray in 2 weeks"
            
# # #             # 4. Draw Overlay (With correct color)
# # #             final_img = image
# # #             if ai_conf > 0:
# # #                 final_img = draw_smart_overlay(image, box, ai_label, ai_conf, risk_label)

# # #             st.image(final_img, caption="AI Enhanced View (Contour Map)", use_container_width=True)

# # #     # --- D. EXPLAINABILITY DASHBOARD (FIXED RENDERING) ---
# # #     st.markdown("---")

# # #     # This HTML block is the "Robust Fix" for the label visibility issue
# # #     st.markdown(f"""
# # #     <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 6px solid {risk_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
# # #         <h3 style="color: black; margin-top: 0; font-family: sans-serif;">🤖 System Reasoning Report</h3>
        
# # #         <p style="color: #333; font-size: 16px;">
# # #             <b>1. Visual Analysis:</b> AI highlighted a specific region in the <span style="color: #0056b3; font-weight: bold;">{ai_loc}</span>.
# # #         </p>
        
# # #         <p style="color: #333; font-size: 16px;">
# # #             <b>2. Clinical Context:</b> {narrative}
# # #         </p>
        
# # #         <p style="color: #333; font-size: 16px;">
# # #             <b>3. Why this decision?</b> The system correlated the visual opacity ({ai_conf*100:.0f}%) with {len(findings)} clinical risk factor(s). {'The low probability suggests Latent/Early stage.' if final_prob <= 0.6 else 'The high probability indicates Active pathology.'}
# # #         </p>
        
# # #         <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
        
# # #         <div style="background-color: {bg_color}; padding: 20px; border-radius: 8px; text-align: center;">
# # #             <h2 style="color: {risk_color}; margin: 0; font-weight: 900; font-size: 28px; letter-spacing: 1px; font-family: sans-serif;">
# # #                 {risk_label} ({final_prob*100:.1f}%)
# # #             </h2>
# # #             <p style="color: #333; margin-top: 10px; font-style: italic; font-weight: 600; font-size: 16px;">
# # #                 Recommended Action: {action_text}
# # #             </p>
# # #         </div>
        
# # #     </div>
# # #     """, unsafe_allow_html=True)









# # import streamlit as st
# # from PIL import Image, ImageDraw, ImageFont
# # import numpy as np
# # import tensorflow as tf
# # from ultralytics import YOLO
# # import pre_symptoms_solver  # Ensure this file is in the same folder

# # # --- CONFIGURATION ---
# # st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# # # --- CUSTOM CSS (High Contrast & Black Text) ---
# # st.markdown("""
# #     <style>
# #     /* Main Report Card Styling */
# #     .reasoning-card {
# #         background-color: #f8f9fa; 
# #         border-left: 6px solid #007bff;
# #         padding: 25px;
# #         border-radius: 8px;
# #         margin-top: 20px;
# #         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
# #     }
    
# #     /* FORCE BLACK TEXT GLOBAL OVERRIDE */
# #     .reasoning-card h3 { color: #000000 !important; margin-top: 0; }
# #     .reasoning-card p { color: #333333 !important; font-size: 16px; line-height: 1.5; }
# #     .reasoning-card li { color: #333333 !important; }
# #     .highlight-text { font-weight: bold; color: #0056b3 !important; }
    
# #     /* Streamlit Metric Fixes */
# #     div[data-testid="stMetricValue"] { color: #000 !important; }
# #     div[data-testid="stMetricLabel"] { color: #333 !important; }
# #     </style>
# # """, unsafe_allow_html=True)

# # # --- LOAD MODELS ---
# # @st.cache_resource
# # def load_models():
# #     try:
# #         router = tf.keras.models.load_model('router_model.h5')
# #         bone_model = YOLO('bone_model.pt')
# #         lung_model = YOLO('lung_model.pt')
# #         return router, bone_model, lung_model
# #     except Exception as e:
# #         return None, None, None

# # router, bone_model, lung_model = load_models()

# # # --- 1. SMART OVERLAY (Blue/Red Logic) ---
# # def draw_smart_overlay(image, box, label, conf, risk_status):
# #     """
# #     Draws a contour. Color changes based on Risk Status.
# #     RED = Active/High Priority
# #     BLUE = Latent/Routine Review
# #     """
# #     overlay = image.copy().convert("RGBA")
# #     draw = ImageDraw.Draw(overlay)
# #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# #     # --- COLOR LOGIC ---
# #     if risk_status == "HIGH PRIORITY":
# #         # RED (Active TB / Fracture)
# #         fill_color = (255, 0, 0, 70)     # Red with transparency
# #         outline_color = (255, 0, 0, 200) # Solid Red
# #         text_bg = (255, 0, 0, 255)       # Solid Red for text
# #         tag_text = f"ACTIVE: {conf*100:.1f}%"
# #     else:
# #         # BLUE (Latent TB / Hairline)
# #         fill_color = (0, 120, 255, 70)   # Blue with transparency
# #         outline_color = (0, 120, 255, 200)
# #         text_bg = (0, 120, 255, 255)
# #         tag_text = f"LATENT/EARLY: {conf*100:.1f}%"
    
# #     # Draw The Box
# #     draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=4)
    
# #     # Draw The Text Label
# #     try: font = ImageFont.truetype("arial.ttf", 20)
# #     except: font = ImageFont.load_default()
    
# #     text_bbox = draw.textbbox((x1, y1), tag_text, font=font)
# #     text_w = text_bbox[2] - text_bbox[0] + 10
# #     text_h = text_bbox[3] - text_bbox[1] + 10
    
# #     # Draw text background rectangle (above the box)
# #     draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=text_bg)
# #     # Draw White Text
# #     draw.text((x1 + 5, y1 - text_h + 5), tag_text, fill="white", font=font)
    
# #     # Blend back
# #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # # --- 2. SPATIAL LOCALIZER ---
# # def get_location_text(box, w, h, scan_type):
# #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
# #     if scan_type == "Chest":
# #         if cy < h/3: zone = "Apical/Upper Zone"
# #         elif cy < 2*h/3: zone = "Middle Zone"
# #         else: zone = "Basal/Lower Zone"
# #         side = "Right" if cx < w/2 else "Left" 
# #         return f"{side} {zone}"
# #     else:
# #         loc_y = "Distal" if cy > h/2 else "Proximal"
# #         side = "Right" if cx > w/2 else "Left"
# #         return f"{loc_y} {side} Region"

# # # --- MAIN APP UI ---
# # st.title("Dayflow: Explainable Medical AI")
# # st.markdown("**Mission:** Assisting healthcare workers with automated triage and transparent clinical reasoning.")

# # if not router:
# #     st.error("⚠️ Models offline. Please check file paths.")
# #     st.stop()

# # # --- SIDEBAR (Dynamic Symptoms) ---
# # with st.sidebar:
# #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# #     st.header("Patient Intake")
# #     sym_placeholder = st.empty()
# #     st.info("Upload a scan to activate the specific symptom checklist.")

# # # --- UPLOAD & ROUTING ---
# # uploaded_file = st.file_uploader("Upload X-Ray (Chest or Bone)", type=['jpg', 'png', 'jpeg'])

# # if uploaded_file:
# #     image = Image.open(uploaded_file).convert('RGB')
# #     w, h = image.size
    
# #     # A. ROUTER PHASE (Automatic)
# #     img_arr = np.array(image.resize((224,224)))/255.0
# #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# #     route_idx = np.argmax(pred)
# #     detected_type = ["Bone", "Chest", "Invalid"][route_idx]
    
# #     if detected_type == "Invalid":
# #         st.error("🚫 Image Rejected: Not a recognized medical scan.")
# #         st.stop()
        
# #     st.success(f"✅ **Auto-Detected:** {detected_type} Scan")

# #     # B. DYNAMIC SYMPTOMS (Based on Type)
# #     syms = {}
# #     with sym_placeholder.container():
# #         st.subheader(f"{detected_type} Clinical Signs")
# #         if detected_type == "Bone":
# #             syms['deformity'] = st.checkbox("Visible Deformity")
# #             syms['immobility'] = st.checkbox("Function Loss")
# #             syms['trauma'] = st.checkbox("Trauma History")
# #             syms['pain'] = st.checkbox("Severe Pain")
# #         else:
# #             syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
# #             syms['cough'] = st.checkbox("Chronic Cough")
# #             syms['weight_loss'] = st.checkbox("Weight Loss")
# #             syms['night_sweats'] = st.checkbox("Night Sweats")

# #     # --- C. PRE-CALCULATE RISK ---
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.image(image, caption="Original Input", use_container_width=True)
        
# #     with col2:
# #         with st.spinner(f"Scanning for {detected_type} anomalies..."):
# #             model = bone_model if detected_type == "Bone" else lung_model
# #             results = model.predict(image, conf=0.15, augment=True, verbose=False)
            
# #             ai_conf = 0.0
# #             ai_loc = "Unknown"
# #             ai_label = "Anomaly"
            
# #             # 1. Get AI Confidence
# #             if len(results[0].boxes) > 0:
# #                 box = results[0].boxes[0]
# #                 ai_conf = float(box.conf[0])
# #                 ai_loc = get_location_text(box, w, h, detected_type)
# #                 ai_label = model.names[int(box.cls[0])]

# #             # 2. Calculate Final Probability (Fusion)
# #             boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
# #                 syms, detected_type, ai_conf, ai_loc
# #             )
# #             final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
            
# #             # 3. Determine Status/Color (Dark Red vs Dark Blue)
# #             if final_prob > 0.60:
# #                 risk_label = "HIGH PRIORITY"
# #                 risk_color = "#cc0000"  # Dark Red
# #                 bg_color = "#ffe6e6"    # Light Red Background
# #                 action_text = "Isolate Patient & Order Sputum Test" if "Chest" in detected_type else "Splint Limb & Order Ortho Consult"
# #             else:
# #                 risk_label = "ROUTINE REVIEW"
# #                 risk_color = "#004085"  # Dark Navy Blue
# #                 bg_color = "#e6f0ff"    # Light Blue Background
# #                 action_text = "Observation / Follow-up X-ray in 2 weeks"
            
# #             # 4. Draw Overlay (With correct color)
# #             final_img = image
# #             if ai_conf > 0:
# #                 final_img = draw_smart_overlay(image, box, ai_label, ai_conf, risk_label)

# #             st.image(final_img, caption="AI Enhanced View (Contour Map)", use_container_width=True)

# #     # --- D. EXPLAINABILITY DASHBOARD (FIXED RENDERING) ---
# #     st.markdown("---")

# #     # This HTML block is the "Robust Fix" for the label visibility issue
# #     st.markdown(f"""
# #     <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 6px solid {risk_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
# #         <h3 style="color: black; margin-top: 0; font-family: sans-serif;">🤖 System Reasoning Report</h3>
        
# #         <p style="color: #333; font-size: 16px;">
# #             <b>1. Visual Analysis:</b> AI highlighted a specific region in the <span style="color: #0056b3; font-weight: bold;">{ai_loc}</span>.
# #         </p>
        
# #         <p style="color: #333; font-size: 16px;">
# #             <b>2. Clinical Context:</b> {narrative}
# #         </p>
        
# #         <p style="color: #333; font-size: 16px;">
# #             <b>3. Why this decision?</b> The system correlated the visual opacity ({ai_conf*100:.0f}%) with {len(findings)} clinical risk factor(s). {'The low probability suggests Latent/Early stage.' if final_prob <= 0.6 else 'The high probability indicates Active pathology.'}
# #         </p>
        
# #         <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
        
# #         <div style="background-color: {bg_color}; padding: 20px; border-radius: 8px; text-align: center;">
# #             <h2 style="color: {risk_color}; margin: 0; font-weight: 900; font-size: 28px; letter-spacing: 1px; font-family: sans-serif;">
# #                 {risk_label} ({final_prob*100:.1f}%)
# #             </h2>
# #             <p style="color: #333; margin-top: 10px; font-style: italic; font-weight: 600; font-size: 16px;">
# #                 Recommended Action: {action_text}
# #             </p>
# #         </div>
        
# #     </div>
# #     """, unsafe_allow_html=True)








# import streamlit as st
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# import pre_symptoms_solver  # Ensure this file is in the same folder

# # --- CONFIGURATION ---
# st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# # --- CUSTOM CSS (High Contrast & Black Text) ---
# st.markdown("""
#     <style>
#     /* Main Report Card Styling */
#     .reasoning-card {
#         background-color: #f8f9fa; 
#         border-left: 6px solid #007bff;
#         padding: 25px;
#         border-radius: 8px;
#         margin-top: 20px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     /* FORCE BLACK TEXT GLOBAL OVERRIDE */
#     .reasoning-card h3 { color: #000000 !important; margin-top: 0; }
#     .reasoning-card p { color: #333333 !important; font-size: 16px; line-height: 1.5; }
#     .reasoning-card li { color: #333333 !important; }
#     .highlight-text { font-weight: bold; color: #0056b3 !important; }
    
#     /* Streamlit Metric Fixes */
#     div[data-testid="stMetricValue"] { color: #000 !important; }
#     div[data-testid="stMetricLabel"] { color: #333 !important; }
#     </style>
# """, unsafe_allow_html=True)

# # --- LOAD MODELS ---
# @st.cache_resource
# def load_models():
#     try:
#         router = tf.keras.models.load_model('router_model.h5')
#         bone_model = YOLO('bone_model.pt')
#         lung_model = YOLO('lung_model.pt')
#         return router, bone_model, lung_model
#     except Exception as e:
#         return None, None, None

# router, bone_model, lung_model = load_models()

# # --- 1. SMART OVERLAY (Blue/Red Logic) ---
# def draw_smart_overlay(image, box, label, conf, risk_status):
#     """
#     Draws a contour. Color changes based on Risk Status.
#     RED = Active/High Priority
#     BLUE = Latent/Routine Review
#     """
#     overlay = image.copy().convert("RGBA")
#     draw = ImageDraw.Draw(overlay)
#     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
#     # --- COLOR LOGIC ---
#     if risk_status == "HIGH PRIORITY":
#         # RED (Active TB / Fracture)
#         fill_color = (255, 0, 0, 70)     # Red with transparency
#         outline_color = (255, 0, 0, 200) # Solid Red
#         text_bg = (255, 0, 0, 255)       # Solid Red for text
#         tag_text = f"ACTIVE: {conf*100:.1f}%"
#     else:
#         # BLUE (Latent TB / Hairline)
#         fill_color = (0, 120, 255, 70)   # Blue with transparency
#         outline_color = (0, 120, 255, 200)
#         text_bg = (0, 120, 255, 255)
#         tag_text = f"LATENT/EARLY: {conf*100:.1f}%"
    
#     # Draw The Box
#     draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=4)
    
#     # Draw The Text Label
#     try: font = ImageFont.truetype("arial.ttf", 20)
#     except: font = ImageFont.load_default()
    
#     text_bbox = draw.textbbox((x1, y1), tag_text, font=font)
#     text_w = text_bbox[2] - text_bbox[0] + 10
#     text_h = text_bbox[3] - text_bbox[1] + 10
    
#     # Draw text background rectangle (above the box)
#     draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=text_bg)
#     # Draw White Text
#     draw.text((x1 + 5, y1 - text_h + 5), tag_text, fill="white", font=font)
    
#     # Blend back
#     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # --- 2. SPATIAL LOCALIZER ---
# def get_location_text(box, w, h, scan_type):
#     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
#     if scan_type == "Chest":
#         if cy < h/3: zone = "Apical/Upper Zone"
#         elif cy < 2*h/3: zone = "Middle Zone"
#         else: zone = "Basal/Lower Zone"
#         side = "Right" if cx < w/2 else "Left" 
#         return f"{side} {zone}"
#     else:
#         loc_y = "Distal" if cy > h/2 else "Proximal"
#         side = "Right" if cx > w/2 else "Left"
#         return f"{loc_y} {side} Region"

# # --- MAIN APP UI ---
# st.title("Dayflow: Explainable Medical AI")
# st.markdown("**Mission:** Assisting healthcare workers with automated triage and transparent clinical reasoning.")

# if not router:
#     st.error("⚠️ Models offline. Please check file paths.")
#     st.stop()

# # --- SIDEBAR (Dynamic Symptoms) ---
# with st.sidebar:
#     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
#     st.header("Patient Intake")
#     sym_placeholder = st.empty()
#     st.info("Upload a scan to activate the specific symptom checklist.")

# # --- UPLOAD & ROUTING ---
# uploaded_file = st.file_uploader("Upload X-Ray (Chest or Bone)", type=['jpg', 'png', 'jpeg'])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert('RGB')
#     w, h = image.size
    
#     # A. ROUTER PHASE (Automatic)
#     img_arr = np.array(image.resize((224,224)))/255.0
#     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
#     route_idx = np.argmax(pred)
#     detected_type = ["Bone", "Chest", "Invalid"][route_idx]
    
#     if detected_type == "Invalid":
#         st.error("🚫 Image Rejected: Not a recognized medical scan.")
#         st.stop()
        
#     st.success(f"✅ **Auto-Detected:** {detected_type} Scan")

#     # B. DYNAMIC SYMPTOMS (Based on Type)
#     syms = {}
#     with sym_placeholder.container():
#         st.subheader(f"{detected_type} Clinical Signs")
#         if detected_type == "Bone":
#             syms['deformity'] = st.checkbox("Visible Deformity")
#             syms['immobility'] = st.checkbox("Function Loss")
#             syms['trauma'] = st.checkbox("Trauma History")
#             syms['pain'] = st.checkbox("Severe Pain")
#         else:
#             syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
#             syms['cough'] = st.checkbox("Chronic Cough")
#             syms['weight_loss'] = st.checkbox("Weight Loss")
#             syms['night_sweats'] = st.checkbox("Night Sweats")

#     # --- C. PRE-CALCULATE RISK ---
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(image, caption="Original Input", use_container_width=True)
        
#     with col2:
#         with st.spinner(f"Scanning for {detected_type} anomalies..."):
#             model = bone_model if detected_type == "Bone" else lung_model
#             results = model.predict(image, conf=0.15, augment=True, verbose=False)
            
#             ai_conf = 0.0
#             ai_loc = "Unknown"
#             ai_label = "Anomaly"
            
#             # 1. Get AI Confidence
#             if len(results[0].boxes) > 0:
#                 box = results[0].boxes[0]
#                 ai_conf = float(box.conf[0])
#                 ai_loc = get_location_text(box, w, h, detected_type)
#                 ai_label = model.names[int(box.cls[0])]

#             # 2. Calculate Final Probability (Fusion)
#             boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
#                 syms, detected_type, ai_conf, ai_loc
#             )
#             final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
            
#             # 3. Determine Status/Color (Dark Red vs Dark Blue)
#             if final_prob > 0.60:
#                 risk_label = "HIGH PRIORITY"
#                 risk_color = "#cc0000"  # Dark Red
#                 bg_color = "#ffe6e6"    # Light Red Background
#                 action_text = "Isolate Patient & Order Sputum Test" if "Chest" in detected_type else "Splint Limb & Order Ortho Consult"
#             else:
#                 risk_label = "ROUTINE REVIEW"
#                 risk_color = "#004085"  # Dark Navy Blue
#                 bg_color = "#e6f0ff"    # Light Blue Background
#                 action_text = "Observation / Follow-up X-ray in 2 weeks"
            
#             # 4. Draw Overlay (With correct color)
#             final_img = image
#             if ai_conf > 0:
#                 final_img = draw_smart_overlay(image, box, ai_label, ai_conf, risk_label)

#             st.image(final_img, caption="AI Enhanced View (Contour Map)", use_container_width=True)

#    # --- E. EXPLAINABILITY DASHBOARD (FIXED RENDERING) ---
#     st.markdown("---")
    
#     # 1. Determine Status & Colors
#     if final_prob > 0.60:
#         risk_label = "HIGH PRIORITY"
#         risk_color = "#cc0000"  # Dark Red
#         bg_color = "#ffe6e6"    # Light Red Background
#         action_text = "Isolate Patient & Order Sputum Test" if "Chest" in detected_type else "Splint Limb & Order Ortho Consult"
#     else:
#         risk_label = "ROUTINE REVIEW"
#         risk_color = "#004085"  # Navy Blue
#         bg_color = "#e6f0ff"    # Light Blue Background
#         action_text = "Observation / Follow-up X-ray in 2 weeks"

#     # 2. Render Report
#     # IMPORTANT: The HTML string below is NOT indented to prevent it from turning into a code block.
#     html_code = f"""
# <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
#     <h3 style="color: black; margin-top: 0; font-family: sans-serif;">🤖 System Reasoning Report</h3>
#     <p style="color: #333; font-size: 16px; margin-bottom: 10px;">
#         <b>1. Visual Analysis:</b> AI highlighted a specific region in the <span style="color: #0056b3; font-weight: bold;">{ai_loc}</span>.
#     </p>
#     <p style="color: #333; font-size: 16px; margin-bottom: 10px;">
#         <b>2. Clinical Context:</b> {narrative}
#     </p>
#     <p style="color: #333; font-size: 16px; margin-bottom: 20px;">
#         <b>3. Why this decision?</b> The system correlated the visual opacity ({ai_conf*100:.0f}%) with {len(findings)} clinical risk factor(s).
#     </p>
#     <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
#     <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; text-align: center; margin-top: 15px;">
#         <h2 style="color: {risk_color}; margin: 0; font-weight: 900; font-size: 24px; font-family: sans-serif;">
#             {risk_label} ({final_prob*100:.1f}%)
#         </h2>
#         <p style="color: #333; margin-top: 5px; font-style: italic; font-weight: 600; font-size: 14px;">
#             Recommended Action: {action_text}
#         </p>
#     </div>
# </div>
# """
#     st.markdown(html_code, unsafe_allow_html=True)
    







# import streamlit as st
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# import pre_symptoms_solver  # Ensure this file is in the same folder

# # --- CONFIGURATION ---
# st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# # --- LOAD MODELS ---
# @st.cache_resource
# def load_models():
#     try:
#         router = tf.keras.models.load_model('router_model.h5')
#         bone_model = YOLO('bone_model.pt')
#         lung_model = YOLO('lung_model.pt')
#         return router, bone_model, lung_model
#     except Exception as e:
#         return None, None, None

# router, bone_model, lung_model = load_models()

# # --- 1. SMART OVERLAY (Blue/Red Logic) ---
# def draw_smart_overlay(image, box, label, conf, risk_status):
#     overlay = image.copy().convert("RGBA")
#     draw = ImageDraw.Draw(overlay)
#     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
#     # Simple Color Logic
#     if risk_status == "HIGH PRIORITY":
#         color = (255, 0, 0) # Red
#     else:
#         color = (0, 120, 255) # Blue
    
#     # Draw Box (Transparency handled by fill color)
#     draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
    
#     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # --- 2. LOCATION TEXT ---
# def get_location_text(box, w, h, scan_type):
#     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
#     side = "Right" if cx > w/2 else "Left" # Simplified logic
#     if scan_type == "Chest":
#         zone = "Upper" if cy < h/3 else "Middle" if cy < 2*h/3 else "Lower"
#         return f"{side} {zone} Zone"
#     else:
#         return f"{side} Limb Region"

# # --- MAIN APP UI ---
# st.title("Dayflow: Explainable Medical AI")

# if not router:
#     st.error("⚠️ Models offline. Please check file paths.")
#     st.stop()

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("Patient Intake")
#     sym_placeholder = st.empty()
#     st.info("Upload a scan to activate the specific symptom checklist.")

# # --- UPLOAD & ROUTING ---
# uploaded_file = st.file_uploader("Upload X-Ray (Chest or Bone)", type=['jpg', 'png', 'jpeg'])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert('RGB')
#     w, h = image.size
    
#     # A. ROUTER
#     img_arr = np.array(image.resize((224,224)))/255.0
#     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
#     route_idx = np.argmax(pred)
#     detected_type = ["Bone", "Chest", "Invalid"][route_idx]
    
#     if detected_type == "Invalid":
#         st.error("🚫 Image Rejected: Not a recognized medical scan.")
#         st.stop()
        
#     st.success(f"✅ **Auto-Detected:** {detected_type} Scan")

#     # B. DYNAMIC SYMPTOMS
#     syms = {}
#     with sym_placeholder.container():
#         st.subheader(f"{detected_type} Clinical Signs")
#         if detected_type == "Bone":
#             syms['deformity'] = st.checkbox("Visible Deformity")
#             syms['immobility'] = st.checkbox("Function Loss")
#             syms['trauma'] = st.checkbox("Trauma History")
#             syms['pain'] = st.checkbox("Severe Pain")
#         else:
#             syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
#             syms['cough'] = st.checkbox("Chronic Cough")
#             syms['weight_loss'] = st.checkbox("Weight Loss")
#             syms['night_sweats'] = st.checkbox("Night Sweats")

#     # --- C. ANALYSIS ---
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(image, caption="Original Input", use_container_width=True)
        
#     with col2:
#         with st.spinner(f"Scanning..."):
#             model = bone_model if detected_type == "Bone" else lung_model
#             results = model.predict(image, conf=0.15, augment=True, verbose=False)
            
#             ai_conf = 0.0
#             ai_loc = "Unknown"
            
#             if len(results[0].boxes) > 0:
#                 box = results[0].boxes[0]
#                 ai_conf = float(box.conf[0])
#                 ai_loc = get_location_text(box, w, h, detected_type)

#             # Calculate Risk
#             boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
#                 syms, detected_type, ai_conf, ai_loc
#             )
#             final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
            
#             # Determine Status
#             if final_prob > 0.60:
#                 risk_label = "HIGH PRIORITY"
#                 action = "Isolate Patient & Order Test"
#             else:
#                 risk_label = "ROUTINE REVIEW"
#                 action = "Observation / Follow-up"
            
#             # Draw Overlay
#             final_img = image
#             if ai_conf > 0:
#                 final_img = draw_smart_overlay(image, box, "Anomaly", ai_conf, risk_label)

#             st.image(final_img, caption="AI Enhanced View", use_container_width=True)

#     # --- D. SIMPLE EXPLAINABILITY DASHBOARD ---
#     st.markdown("---")
    
#     # 1. Create a container for the report
#     with st.container(border=True):
#         st.subheader("🤖 System Reasoning Report")
        
#         # 2. Simple text output
#         st.write(f"**1. Visual Analysis:** AI highlighted region: **{ai_loc}**")
#         st.write(f"**2. Clinical Context:** {narrative}")
#         st.write(f"**3. Why?** Visual opacity ({ai_conf*100:.0f}%) combined with {len(findings)} clinical factors.")
        
#         st.divider() # Adds a thin line
        
#         # 3. Final Result using built-in Streamlit boxes
#         result_text = f"**{risk_label} ({final_prob*100:.1f}%)**\n\n*Action: {action}*"
        
#         if risk_label == "HIGH PRIORITY":
#             st.error(result_text, icon="🚨") # Red Box
#         else:
#             st.info(result_text, icon="ℹ️")  # Blue Box



























































































import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import pre_symptoms_solver  # Ensure this file is in the same folder

# --- CONFIGURATION ---
st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        router = tf.keras.models.load_model('router_model.h5')
        bone_model = YOLO('bone_model.pt')
        lung_model = YOLO('lung-tb-model.pt')
        # lung_model = YOLO('lung_model.pt')
        return router, bone_model, lung_model
    except Exception as e:
        return None, None, None

router, bone_model, lung_model = load_models()

# --- 1. SMART OVERLAY (Handles MULTIPLE Boxes) ---
def draw_smart_overlay(image, boxes, risk_status):
    """
    Draws ALL detected boxes on the image.
    """
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    
    # Simple Color Logic
    if risk_status == "HIGH PRIORITY":
        color = (255, 0, 0) # Red
    else:
        color = (0, 120, 255) # Blue
    
    # LOOP: Iterate through every box found by the AI
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        # Draw Box (Transparency handled by fill color)
        draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
    
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# --- 2. LOCATION TEXT (Handles MULTIPLE Locations) ---
def get_all_locations(boxes, w, h, scan_type):
    """
    Returns a list of location strings for ALL boxes.
    """
    locations = []
    for box in boxes:
        cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
        # Note: In medical imaging, the left side of the image is usually the patient's right.
        # We stick to "Viewer's Left/Right" for simplicity unless you want strict medical anatomical sides.
        side = "Right" if cx > w/2 else "Left" 
        
        if scan_type == "Chest":
            zone = "Upper" if cy < h/3 else "Middle" if cy < 2*h/3 else "Lower"
            locations.append(f"{side} {zone} Zone")
        else:
            region = "Distal" if cy > h/2 else "Proximal"
            locations.append(f"{side} {region} Region")
            
    # Remove duplicates (e.g., if two boxes are in "Right Lower Zone")
    return list(set(locations))

# --- MAIN APP UI ---
st.title("Dayflow: Explainable Medical AI")

if not router:
    st.error("⚠️ Models offline. Please check file paths.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Patient Intake")
    sym_placeholder = st.empty()
    st.info("Upload a scan to activate the specific symptom checklist.")

# --- UPLOAD & ROUTING ---
uploaded_file = st.file_uploader("Upload X-Ray (Chest or Bone)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    w, h = image.size
    
    # A. ROUTER
    img_arr = np.array(image.resize((224,224)))/255.0
    pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
    route_idx = np.argmax(pred)
    detected_type = ["Bone", "Chest", "Invalid"][route_idx]
    
    if detected_type == "Invalid":
        st.error("🚫 Image Rejected: Not a recognized medical scan.")
        st.stop()
        
    st.success(f"✅ **Auto-Detected:** {detected_type} Scan")

    # B. DYNAMIC SYMPTOMS
    syms = {}
    with sym_placeholder.container():
        st.subheader(f"{detected_type} Clinical Signs")
        if detected_type == "Bone":
            syms['deformity'] = st.checkbox("Visible Deformity")
            syms['immobility'] = st.checkbox("Function Loss")
            syms['trauma'] = st.checkbox("Trauma History")
            syms['pain'] = st.checkbox("Severe Pain")
        else:
            syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
            syms['cough'] = st.checkbox("Chronic Cough")
            syms['weight_loss'] = st.checkbox("Weight Loss")
            syms['night_sweats'] = st.checkbox("Night Sweats")

    # --- C. ANALYSIS ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Input", use_container_width=True)
        
    with col2:
        with st.spinner(f"Scanning..."):
            model = bone_model if detected_type == "Bone" else lung_model
            # Get results
            results = model.predict(image, conf=0.15, augment=True, verbose=False)
            
            # Init Analysis Variables
            ai_conf = 0.0
            ai_locs = []
            detected_boxes = []
            
            # Check if any boxes exist
            if len(results[0].boxes) > 0:
                detected_boxes = results[0].boxes
                
                # Use the highest confidence score for the risk calculation
                ai_conf = float(detected_boxes[0].conf[0])
                
                # Get text description for ALL boxes
                ai_locs = get_all_locations(detected_boxes, w, h, detected_type)

            # Generate Narrative
            # We join multiple locations with commas for the text report
            loc_string = ", ".join(ai_locs) if ai_locs else "Unknown"
            
            boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
                syms, detected_type, ai_conf, loc_string
            )
            final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
            
            # Determine Status
            if final_prob > 0.60:
                risk_label = "HIGH PRIORITY"
                action = "Isolate Patient & Order Test"
            else:
                risk_label = "ROUTINE REVIEW"
                action = "Observation / Follow-up"
            
            # Draw Overlay (PASSING ALL BOXES NOW)
            final_img = image
            if len(detected_boxes) > 0:
                final_img = draw_smart_overlay(image, detected_boxes, risk_label)

            st.image(final_img, caption=f"AI Detected: {len(detected_boxes)} anomalies", use_container_width=True)
            

    # --- D. EXPLAINABILITY REPORT ---
    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("🤖 System Reasoning Report")
        
        # 1. Visual Analysis Report
        if ai_locs:
            st.write(f"**1. Visual Analysis:** AI highlighted **{len(ai_locs)}** region(s): **{', '.join(ai_locs)}**")
        else:
             st.write(f"**1. Visual Analysis:** No anomalies detected.")
        
        # 2. Clinical Context
        st.write(f"**2. Clinical Context:** {narrative}")
        st.write(f"**3. Why?** Visual opacity ({ai_conf*100:.0f}%) combined with {len(findings)} clinical factors.")
        
        st.divider()
        
        # 3. Final Verdict
        result_text = f"**{risk_label} ({final_prob*100:.1f}%)**\n\n*Action: {action}*"
        
        if risk_label == "HIGH PRIORITY":
            st.error(result_text, icon="🚨") 
        else:
            st.info(result_text, icon="ℹ️")











# import streamlit as st
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# import cv2  # [ADDED] Required for enhancement
# import pre_symptoms_solver  # Ensure this file is in the same folder

# # --- CONFIGURATION ---
# st.set_page_config(page_title="Dayflow Explainable AI", page_icon="🩺", layout="wide")

# # --- LOAD MODELS ---
# @st.cache_resource
# def load_models():
#     try:
#         router = tf.keras.models.load_model('router_model.h5')
#         bone_model = YOLO('bone_model.pt')
#         lung_model = YOLO('lung-tb-model.pt')
#         return router, bone_model, lung_model
#     except Exception as e:
#         return None, None, None

# router, bone_model, lung_model = load_models()

# # --- [ADDED] AUTO-ENHANCEMENT FUNCTION ---
# def auto_enhance_image(pil_image):
#     """
#     Automatically improves contrast and reduces noise for better AI detection.
#     """
#     # Convert PIL to OpenCV format (RGB -> BGR)
#     img = np.array(pil_image)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     # This fixes glare and brings out bone details
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl, a, b))
#     enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

#     # 2. Denoise (Removes grain/noise)
#     clean = cv2.fastNlMeansDenoisingColored(enhanced_bgr, None, 10, 10, 7, 21)

#     # Convert back to PIL (BGR -> RGB)
#     final_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
#     return Image.fromarray(final_rgb)

# # --- 1. SMART OVERLAY (Handles MULTIPLE Boxes) ---
# def draw_smart_overlay(image, boxes, risk_status):
#     """
#     Draws ALL detected boxes on the image.
#     """
#     overlay = image.copy().convert("RGBA")
#     draw = ImageDraw.Draw(overlay)
    
#     # Simple Color Logic
#     if risk_status == "HIGH PRIORITY":
#         color = (255, 0, 0) # Red
#     else:
#         color = (0, 120, 255) # Blue
    
#     # LOOP: Iterate through every box found by the AI
#     for box in boxes:
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         # Draw Box (Transparency handled by fill color)
#         draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
    
#     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # --- 2. LOCATION TEXT (Handles MULTIPLE Locations) ---
# def get_all_locations(boxes, w, h, scan_type):
#     """
#     Returns a list of location strings for ALL boxes.
#     """
#     locations = []
#     for box in boxes:
#         cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
#         # Note: In medical imaging, the left side of the image is usually the patient's right.
#         side = "Right" if cx > w/2 else "Left" 
        
#         if scan_type == "Chest":
#             zone = "Upper" if cy < h/3 else "Middle" if cy < 2*h/3 else "Lower"
#             locations.append(f"{side} {zone} Zone")
#         else:
#             region = "Distal" if cy > h/2 else "Proximal"
#             locations.append(f"{side} {region} Region")
            
#     # Remove duplicates
#     return list(set(locations))

# # --- MAIN APP UI ---
# st.title("Dayflow: Explainable Medical AI")

# if not router:
#     st.error("⚠️ Models offline. Please check file paths.")
#     st.stop()

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("Patient Intake")
#     sym_placeholder = st.empty()
#     st.info("Upload a scan to activate the specific symptom checklist.")

# # --- UPLOAD & ROUTING ---
# uploaded_file = st.file_uploader("Upload X-Ray (Chest or Bone)", type=['jpg', 'png', 'jpeg'])

# if uploaded_file:
#     # 1. LOAD RAW IMAGE
#     raw_image = Image.open(uploaded_file).convert('RGB')
    
#     # 2. [ADDED] APPLY ENHANCEMENT IMMEDIATELY
#     # The image is now cleaner before the AI ever sees it
#     image = auto_enhance_image(raw_image)
    
#     w, h = image.size
    
#     # A. ROUTER (Uses Enhanced Image)
#     img_arr = np.array(image.resize((224,224)))/255.0
#     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
#     route_idx = np.argmax(pred)
#     detected_type = ["Bone", "Chest", "Invalid"][route_idx]
    
#     if detected_type == "Invalid":
#         st.error("🚫 Image Rejected: Not a recognized medical scan.")
#         st.stop()
        
#     st.success(f"✅ **Auto-Detected:** {detected_type} Scan")

#     # B. DYNAMIC SYMPTOMS
#     syms = {}
#     with sym_placeholder.container():
#         st.subheader(f"{detected_type} Clinical Signs")
#         if detected_type == "Bone":
#             syms['deformity'] = st.checkbox("Visible Deformity")
#             syms['immobility'] = st.checkbox("Function Loss")
#             syms['trauma'] = st.checkbox("Trauma History")
#             syms['pain'] = st.checkbox("Severe Pain")
#         else:
#             syms['blood_sputum'] = st.checkbox("Hemoptysis (Blood)")
#             syms['cough'] = st.checkbox("Chronic Cough")
#             syms['weight_loss'] = st.checkbox("Weight Loss")
#             syms['night_sweats'] = st.checkbox("Night Sweats")

#     # --- C. ANALYSIS ---
#     col1, col2 = st.columns(2)
#     with col1:
#         # Display the ENHANCED image so user sees what AI sees
#         st.image(image, caption="Enhanced Input (AI View)", use_container_width=True)
        
#     with col2:
#         with st.spinner(f"Scanning..."):
#             model = bone_model if detected_type == "Bone" else lung_model
            
#             # Get results (Uses Enhanced Image)
#             results = model.predict(image, conf=0.15, augment=True, verbose=False)
            
#             # Init Analysis Variables
#             ai_conf = 0.0
#             ai_locs = []
#             detected_boxes = []
            
#             # Check if any boxes exist
#             if len(results[0].boxes) > 0:
#                 detected_boxes = results[0].boxes
                
#                 # Use the highest confidence score for the risk calculation
#                 ai_conf = float(detected_boxes[0].conf[0])
                
#                 # Get text description for ALL boxes
#                 ai_locs = get_all_locations(detected_boxes, w, h, detected_type)

#             # Generate Narrative
#             loc_string = ", ".join(ai_locs) if ai_locs else "Unknown"
            
#             boost, findings, narrative = pre_symptoms_solver.generate_diagnostic_reasoning(
#                 syms, detected_type, ai_conf, loc_string
#             )
#             final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
            
#             # Determine Status
#             if final_prob > 0.60:
#                 risk_label = "HIGH PRIORITY"
#                 action = "Isolate Patient & Order Test"
#             else:
#                 risk_label = "ROUTINE REVIEW"
#                 action = "Observation / Follow-up"
            
#             # Draw Overlay (Uses Enhanced Image)
#             final_img = image
#             if len(detected_boxes) > 0:
#                 final_img = draw_smart_overlay(image, detected_boxes, risk_label)

#             st.image(final_img, caption=f"AI Detected: {len(detected_boxes)} anomalies", use_container_width=True)
            

#     # --- D. EXPLAINABILITY REPORT ---
#     st.markdown("---")
    
#     with st.container(border=True):
#         st.subheader("🤖 System Reasoning Report")
        
#         # 1. Visual Analysis Report
#         if ai_locs:
#             st.write(f"**1. Visual Analysis:** AI highlighted **{len(ai_locs)}** region(s): **{', '.join(ai_locs)}**")
#         else:
#              st.write(f"**1. Visual Analysis:** No anomalies detected.")
        
#         # 2. Clinical Context
#         st.write(f"**2. Clinical Context:** {narrative}")
#         st.write(f"**3. Why?** Visual opacity ({ai_conf*100:.0f}%) combined with {len(findings)} clinical factors.")
        
#         st.divider()
        
#         # 3. Final Verdict
#         result_text = f"**{risk_label} ({final_prob*100:.1f}%)**\n\n*Action: {action}*"
        
#         if risk_label == "HIGH PRIORITY":
#             st.error(result_text, icon="🚨") 
#         else:
#             st.info(result_text, icon="ℹ️")














# import streamlit as st
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# import cv2
# import pre_pre_symptoms_solver  # Logic File

# # --- CONFIGURATION ---
# st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# st.markdown("""
#     <style>
#     .reasoning-card { background-color: #f8f9fa; border-left: 6px solid #007bff; padding: 25px; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
#     .reasoning-card h3 { color: #000 !important; margin-top: 0; }
#     .reasoning-card p { color: #333 !important; font-size: 16px; }
#     </style>
# """, unsafe_allow_html=True)

# # --- LOAD MODELS ---
# @st.cache_resource
# def load_models():
#     try:
#         router = tf.keras.models.load_model('router_model.h5')
#         bone_model = YOLO('bone_model.pt')
#         tb_model = YOLO('lung_model.pt')
#         try: pneu_model = YOLO('pneumonia_model.pt')
#         except: pneu_model = None
#         return router, bone_model, tb_model, pneu_model
#     except: return None, None, None, None

# router, bone_model, tb_model, pneu_model = load_models()

# # --- HELPERS ---

# def enhance_phone_image(pil_image):
#     """Phone Mode: Fixes glare and contrast using CLAHE."""
#     img = np.array(pil_image)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)
#     enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
#     final_img = cv2.merge([enhanced, enhanced, enhanced])
#     return Image.fromarray(final_img)

# def run_tiled_inference(model, image, conf_thresh=0.10):
#     """
#     THE ZOOM ENGINE: Splits image into 4 quadrants to find hairline cracks.
#     Returns the best detection from the tiles OR the full image.
#     """
#     w, h = image.size
#     best_box = None
#     best_conf = 0.0
#     best_label = "Anomaly"
    
#     # 1. Run on FULL Image first
#     results = model.predict(image, conf=conf_thresh, augment=True, verbose=False)
#     if len(results[0].boxes) > 0:
#         best_box = results[0].boxes[0]
#         best_conf = float(best_box.conf[0])
#         best_label = model.names[int(best_box.cls[0])]

#     # 2. Run on TILES (Quadrants) for Zoom
#     # Define tiles: (left, upper, right, lower)
#     tiles = [
#         (0, 0, w//2, h//2),       # Top-Left
#         (w//2, 0, w, h//2),       # Top-Right
#         (0, h//2, w//2, h),       # Bottom-Left
#         (w//2, h//2, w, h)        # Bottom-Right
#     ]
    
#     for i, tile_coords in enumerate(tiles):
#         tile_img = image.crop(tile_coords)
#         t_results = model.predict(tile_img, conf=conf_thresh, verbose=False) # No augment for speed
        
#         if len(t_results[0].boxes) > 0:
#             t_box = t_results[0].boxes[0]
#             t_conf = float(t_box.conf[0])
            
#             # If this tile found something with higher confidence, keep it
#             if t_conf > best_conf:
#                 # IMPORTANT: Map tile coordinates back to Global coordinates
#                 x1, y1, x2, y2 = t_box.xyxy[0].tolist()
#                 offset_x, offset_y = tile_coords[0], tile_coords[1]
                
#                 # Create a fake box object with global coords (Manual fix)
#                 # We reuse the box structure but hack the values
#                 best_box = t_box
#                 # Store the global coordinates in a custom attribute we read later
#                 best_box.global_xyxy = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
#                 best_conf = t_conf
#                 best_label = model.names[int(t_box.cls[0])]
                
#     return best_box, best_conf, best_label

# def draw_smart_overlay(image, box, label, conf, risk_status):
#     overlay = image.copy().convert("RGBA")
#     draw = ImageDraw.Draw(overlay)
    
#     # Check if we have Tiled Global Coords (from Zoom Engine)
#     if hasattr(box, 'global_xyxy'):
#         x1, y1, x2, y2 = box.global_xyxy
#     else:
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
    
#     color = (255, 0, 0) if risk_status == "HIGH PRIORITY" else (0, 120, 255)
#     draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
#     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# def get_location_text(box, w, h, scan_type):
#     # Handle Global Coords for Tiling
#     if hasattr(box, 'global_xyxy'):
#         cx = (box.global_xyxy[0] + box.global_xyxy[2]) / 2
#         cy = (box.global_xyxy[1] + box.global_xyxy[3]) / 2
#     else:
#         cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
        
#     if scan_type == "Chest":
#         zone = "Upper" if cy < h/3 else "Middle" if cy < 2*h/3 else "Lower"
#         side = "Right" if cx < w/2 else "Left" 
#         return f"{side} {zone} Zone"
#     else:
#         # Bone Zones
#         loc_y = "Distal" if cy > h/2 else "Proximal"
#         return f"{loc_y} Region"

# # --- SIDEBAR & UI ---
# with st.sidebar:
#     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
#     st.header("Patient Intake")
    
#     if "symptom_state" not in st.session_state: st.session_state.symptom_state = {}

#     chest_mode = "Tuberculosis (TB)"
#     if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
#         chest_mode = st.radio("Protocol:", ["Tuberculosis (TB)", "Pneumonia"])
    
#     st.markdown("---")
#     st.subheader("🗣️ AI Scribe")
#     user_notes = st.text_area("Describe symptoms...", height=80)
    
#     if st.button("⚡ Auto-Fill"):
#         if "detected_type" in st.session_state:
#             detected = pre_pre_symptoms_solver.parse_symptoms_from_text(user_notes, st.session_state.detected_type)
#             for key in detected: st.session_state.symptom_state[key] = True
#             if detected: st.success(f"Detected: {', '.join(detected)}")
#             else: st.warning("No keywords found.")

#     st.markdown("---")
#     syms = {}
#     if "detected_type" in st.session_state:
#         dtype = st.session_state.detected_type
#         def smart_chk(lbl, key):
#             val = st.checkbox(lbl, value=st.session_state.symptom_state.get(key, False), key=key)
#             st.session_state.symptom_state[key] = val
#             return val

#         if dtype == "Bone":
#             st.caption("Fracture Signs")
#             syms['deformity'] = smart_chk("Deformity", 'deformity')
#             syms['bone_poke'] = smart_chk("Bone Exposed", 'bone_poke')
#             syms['cannot_move'] = smart_chk("Cannot Move", 'cannot_move')
#             syms['snap_sound'] = smart_chk("Heard Snap", 'snap_sound')
#             syms['trauma'] = smart_chk("Trauma/Fall", 'trauma')
#             st.markdown("---")
#             syms['respiratory_trap'] = smart_chk("Patient Coughing?", 'respiratory_trap')
            
#         elif dtype == "Chest":
#             st.caption("Pneumonia (Acute)")
#             syms['high_fever'] = smart_chk("Fever >39°C", 'high_fever')
#             syms['hard_breathe'] = smart_chk("Dyspnea", 'hard_breathe')
#             st.caption("TB (Chronic)")
#             syms['cough_blood'] = smart_chk("Cough Blood", 'cough_blood')
#             syms['weight_loss'] = smart_chk("Weight Loss", 'weight_loss')
#             syms['night_sweats'] = smart_chk("Night Sweats", 'night_sweats')

#     # Guardrails
#     if "detected_type" in st.session_state:
#         if st.session_state.detected_type == "Chest":
#             if chest_mode == "Tuberculosis (TB)" and syms.get('high_fever'):
#                 st.warning("⚠️ High Fever is usually Pneumonia, not TB.")
#         elif st.session_state.detected_type == "Bone" and syms.get('respiratory_trap'):
#             st.error("🚨 Mismatch: Respiratory signs in Bone patient.")

# # --- MAIN LOGIC ---
# st.title("Dayflow: AI Diagnosis Assistant")

# if not router: st.error("⚠️ Models Offline."); st.stop()

# uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png'])

# if uploaded_file:
#     original_image = Image.open(uploaded_file).convert('RGB')
    
#     col_opt1, col_opt2 = st.columns([3, 1])
#     with col_opt2: use_enhancer = st.checkbox("📸 Phone Mode", value=True)
    
#     if use_enhancer:
#         image_for_ai = enhance_phone_image(original_image)
#         st.caption("✅ Image Enhanced (CLAHE + Denoise)")
#     else:
#         image_for_ai = original_image
    
#     w, h = image_for_ai.size
    
#     # 1. ROUTER
#     img_arr = np.array(image_for_ai.resize((224,224)))/255.0
#     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
#     detected_type = ["Bone", "Chest", "Invalid"][np.argmax(pred)]
#     st.session_state.detected_type = detected_type
    
#     if detected_type == "Invalid": st.error("🚫 Invalid Image"); st.stop()
#     st.success(f"✅ Detected: {detected_type}")

#     # 2. SELECT MODEL
#     model = None
#     condition_name = "Fracture"
#     if detected_type == "Bone": model = bone_model
#     else:
#         if chest_mode == "Pneumonia": model = pneu_model; condition_name = "Pneumonia"
#         else: model = tb_model; condition_name = "Tuberculosis"

#     # 3. ANALYSIS
#     col1, col2 = st.columns(2)
#     with col1: st.image(original_image, caption="Original", use_container_width=True)
    
#     with col2:
#         with st.spinner(f"Analyzing {condition_name}..."):
#             if model:
#                 # --- FRACTURE ZOOM LOGIC ---
#                 if detected_type == "Bone":
#                     # Run the NEW Tiling Engine
#                     box, ai_conf, ai_label = run_tiled_inference(model, image_for_ai)
#                 else:
#                     # Standard Chest Scan
#                     results = model.predict(image_for_ai, conf=0.10, augment=True, verbose=False)
#                     box, ai_conf, ai_label = None, 0.0, "Anomaly"
#                     if len(results[0].boxes) > 0:
#                         box = results[0].boxes[0]
#                         ai_conf = float(box.conf[0])
#                         ai_label = model.names[int(box.cls[0])]

#                 ai_loc = get_location_text(box, w, h, detected_type) if ai_conf > 0 else "Unknown"

#                 # Reasoning
#                 boost, findings, narrative = pre_pre_symptoms_solver.generate_diagnostic_reasoning(
#                     syms, detected_type, ai_conf, ai_loc
#                 )
#                 final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
                
#                 # Colors
#                 if final_prob > 0.60:
#                     risk_lbl = "HIGH PRIORITY"; risk_col = "#cc0000"; bg_col = "#ffe6e6"
#                     act = "Isolate" if "Chest" in detected_type else "Splint"
#                 else:
#                     risk_lbl = "ROUTINE"; risk_col = "#004085"; bg_col = "#e6f0ff"
#                     act = "Observe"
                
#                 # Draw
#                 final_img = image_for_ai
#                 if ai_conf > 0:
#                     final_img = draw_smart_overlay(image_for_ai, box, ai_label, ai_conf, risk_lbl)
                
#                 st.image(final_img, caption=f"AI Result: {condition_name}", use_container_width=True)

#     # 4. REPORT
#     st.markdown("---")
#     html_code = f"""
#     <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 6px solid {risk_col};">
#         <h3 style="color: black; margin: 0;">🤖 Diagnosis Report</h3>
#         <p style="color: #333;"><b>Visual:</b> {ai_label} at <span style="color: #0056b3;">{ai_loc}</span>.</p>
#         <p style="color: #333;"><b>Clinical:</b> {narrative}</p>
#         <hr>
#         <div style="background-color: {bg_col}; padding: 15px; text-align: center; border-radius: 5px;">
#             <h2 style="color: {risk_col}; margin: 0;">{risk_lbl} ({final_prob*100:.1f}%)</h2>
#             <p style="color: #333; margin-top: 5px;"><b>Action:</b> {act}</p>
#         </div>
#     </div>
#     """
#     st.markdown(html_code, unsafe_allow_html=True)