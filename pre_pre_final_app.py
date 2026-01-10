# # # # import streamlit as st
# # # # from PIL import Image, ImageDraw, ImageFont
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from ultralytics import YOLO
# # # # import cv2
# # # # import pre_pre_symptoms_solver  # Importing the logic file above

# # # # # --- CONFIGURATION ---
# # # # st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# # # # # --- CSS (Clean & Readable) ---
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
# # # #     .reasoning-card h3 { color: #000 !important; margin-top: 0; }
# # # #     .reasoning-card p { color: #333 !important; font-size: 16px; }
# # # #     </style>
# # # # """, unsafe_allow_html=True)

# # # # # --- LOAD MODELS ---
# # # # @st.cache_resource
# # # # def load_models():
# # # #     try:
# # # #         router = tf.keras.models.load_model('router_model.h5')
# # # #         bone_model = YOLO('bone_model.pt')
# # # #         tb_model = YOLO('lung_model.pt')
# # # #         # Only load pneumonia if it exists (Kaggle download might be pending)
# # # #         try:
# # # #             pneu_model = YOLO('pneumonia_model.pt')
# # # #         except:
# # # #             pneu_model = None
            
# # # #         return router, bone_model, tb_model, pneu_model
# # # #     except:
# # # #         return None, None, None, None

# # # # router, bone_model, tb_model, pneu_model = load_models()

# # # # # --- HELPER 1: PHONE MODE ENHANCER ---
# # # # def enhance_phone_image(pil_image):
# # # #     """
# # # #     Turns a messy phone photo into a clean, high-contrast X-ray.
# # # #     """
# # # #     img = np.array(pil_image)
# # # #     # Convert RGB to BGR for OpenCV
# # # #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# # # #     # Convert to Grayscale
# # # #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # #     # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
# # # #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# # # #     enhanced = clahe.apply(gray)

# # # #     # Denoise
# # # #     enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

# # # #     # Convert back to fake RGB (3 channels) for YOLO
# # # #     final_img = cv2.merge([enhanced, enhanced, enhanced])
    
# # # #     return Image.fromarray(final_img)

# # # # # --- HELPER 2: VISUALIZATION ---
# # # # def draw_smart_overlay(image, box, label, conf, risk_status):
# # # #     overlay = image.copy().convert("RGBA")
# # # #     draw = ImageDraw.Draw(overlay)
# # # #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# # # #     if risk_status == "HIGH PRIORITY":
# # # #         color = (255, 0, 0) # Red
# # # #     else:
# # # #         color = (0, 120, 255) # Blue
    
# # # #     draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
# # # #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # # # def get_location_text(box, w, h, scan_type):
# # # #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
# # # #     if scan_type == "Chest":
# # # #         zone = "Upper Chest" if cy < h/3 else "Middle Chest" if cy < 2*h/3 else "Lower Chest"
# # # #         side = "Right" if cx < w/2 else "Left" 
# # # #         return f"{side} {zone}"
# # # #     else:
# # # #         return "Bone/Limb Area"

# # # # # --- SIDEBAR LOGIC (AI SCRIBE & GUARDRAILS) ---
# # # # with st.sidebar:
# # # #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# # # #     st.header("Patient Intake")
    
# # # #     # 1. INITIALIZE STATE
# # # #     if "symptom_state" not in st.session_state:
# # # #         st.session_state.symptom_state = {}

# # # #     # 2. MODE SELECTION
# # # #     chest_mode = "Tuberculosis (TB)"
# # # #     if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
# # # #         st.subheader("What do you want to check?")
# # # #         chest_mode = st.radio("Select Protocol:", ["Tuberculosis (TB)", "Pneumonia"])
    
# # # #     st.markdown("---")
    
# # # #     # 3. AI SCRIBE TEXT BOX
# # # #     st.subheader("🗣️ AI Scribe")
# # # #     user_notes = st.text_area("Type symptoms (e.g., 'Coughing blood since tuesday')", height=80)
    
# # # #     if st.button("⚡ Auto-Fill Form"):
# # # #         if "detected_type" in st.session_state:
# # # #             detected = pre_pre_symptoms_solver.parse_symptoms_from_text(user_notes, st.session_state.detected_type)
# # # #             for key in detected:
# # # #                 st.session_state.symptom_state[key] = True
# # # #             if detected:
# # # #                 st.success(f"AI detected: {', '.join(detected)}")
# # # #             else:
# # # #                 st.warning("No keywords found. Try simpler terms.")

# # # #     st.markdown("---")
    
# # # #     # 4. SMART CHECKLIST
# # # #     syms = {}
# # # #     if "detected_type" in st.session_state:
# # # #         dtype = st.session_state.detected_type
        
# # # #         def smart_checkbox(label, key):
# # # #             is_checked = st.session_state.symptom_state.get(key, False)
# # # #             val = st.checkbox(label, value=is_checked, key=key)
# # # #             st.session_state.symptom_state[key] = val
# # # #             return val

# # # #         if dtype == "Bone":
# # # #             st.caption("Broken Bone Signs")
# # # #             syms['deformity'] = smart_checkbox("Bone looks bent / wrong shape", 'deformity')
# # # #             syms['bone_poke'] = smart_checkbox("Bone poking through skin", 'bone_poke')
# # # #             syms['cannot_move'] = smart_checkbox("Cannot move the part", 'cannot_move')
# # # #             syms['snap_sound'] = smart_checkbox("Heard a 'Snap' or 'Crack'", 'snap_sound')
# # # #             syms['swelling'] = smart_checkbox("Bad swelling or bruising", 'swelling')
# # # #             syms['trauma'] = smart_checkbox("Hit by something hard / Fell down", 'trauma')
# # # #             st.markdown("---")
# # # #             syms['respiratory_trap'] = smart_checkbox("Patient is also coughing?", 'respiratory_trap')
            
# # # #         elif dtype == "Chest":
# # # #             st.caption("Pneumonia Signs (Fast & Hot)")
# # # #             syms['high_fever'] = smart_checkbox("Very Hot Fever (>39°C)", 'high_fever')
# # # #             syms['shaking_chills'] = smart_checkbox("Shaking Chills / Shivering", 'shaking_chills')
# # # #             syms['hard_breathe'] = smart_checkbox("Hard to breathe / Gasping", 'hard_breathe')
# # # #             syms['green_phlegm'] = smart_checkbox("Coughing green/yellow slime", 'green_phlegm')
            
# # # #             st.caption("TB Signs (Slow & Wasting)")
# # # #             syms['cough_blood'] = smart_checkbox("Coughing up Blood", 'cough_blood')
# # # #             syms['weight_loss'] = smart_checkbox("Losing weight without trying", 'weight_loss')
# # # #             syms['night_sweats'] = smart_checkbox("Sweating huge amounts at night", 'night_sweats')
# # # #             syms['long_cough'] = smart_checkbox("Cough >3 weeks", 'long_cough')

# # # #     else:
# # # #         st.info("Upload X-Ray to unlock.")

# # # #     # 5. GUARDRAIL ALERTS
# # # #     if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
# # # #         warning_msg = None
# # # #         if chest_mode == "Tuberculosis (TB)" and (syms.get('high_fever') or syms.get('hard_breathe')):
# # # #             warning_msg = "⚠️ **Wait!** High Fever/Gasping are **Pneumonia** signs. \n\n👉 Suggested: Switch to **Pneumonia** above."
# # # #         elif chest_mode == "Pneumonia" and (syms.get('cough_blood') or syms.get('weight_loss')):
# # # #             warning_msg = "⚠️ **Wait!** Coughing Blood/Weight Loss are **TB** signs. \n\n👉 Suggested: Switch to **Tuberculosis** above."
# # # #         if warning_msg: st.warning(warning_msg)

# # # #     if "detected_type" in st.session_state and st.session_state.detected_type == "Bone":
# # # #         if syms.get('respiratory_trap'):
# # # #             st.error("🚨 **Error:** BONE X-ray but patient is coughing? Check patient ID.")

# # # # # --- MAIN APP ---
# # # # st.title("Dayflow: AI Diagnosis Assistant")

# # # # if not router:
# # # #     st.error("⚠️ AI Models are offline.")
# # # #     st.stop()

# # # # uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png', 'jpeg'])

# # # # if uploaded_file:
# # # #     original_image = Image.open(uploaded_file).convert('RGB')
    
# # # #     # --- A. PHONE MODE TOGGLE ---
# # # #     col_opt1, col_opt2 = st.columns([3, 1])
# # # #     with col_opt2:
# # # #         use_enhancer = st.checkbox("📸 Phone Mode", value=True, help="Fixes glare/blur from phone photos.")
    
# # # #     if use_enhancer:
# # # #         image_for_ai = enhance_phone_image(original_image)
# # # #         st.caption("✅ Image automatically enhanced for clarity.")
# # # #     else:
# # # #         image_for_ai = original_image
    
# # # #     w, h = image_for_ai.size
    
# # # #     # --- B. ROUTER ---
# # # #     img_arr = np.array(image_for_ai.resize((224,224)))/255.0
# # # #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# # # #     route_idx = np.argmax(pred)
# # # #     detected_type = ["Bone", "Chest", "Invalid"][route_idx]
# # # #     st.session_state.detected_type = detected_type
    
# # # #     if detected_type == "Invalid":
# # # #         st.error("🚫 This doesn't look like an X-ray."); st.stop()

# # # #     st.success(f"✅ **Detected:** {detected_type} Scan")

# # # #     # --- C. MODEL SELECTION ---
# # # #     condition_name = "Fracture"
# # # #     model = None
    
# # # #     if detected_type == "Bone":
# # # #         model = bone_model
# # # #     else:
# # # #         if chest_mode == "Pneumonia":
# # # #             model = pneu_model; condition_name = "Pneumonia"
# # # #         else:
# # # #             model = tb_model; condition_name = "Tuberculosis"

# # # #     # --- D. ANALYSIS ---
# # # #     col1, col2 = st.columns(2)
# # # #     with col1: 
# # # #         st.image(original_image, caption="Original Upload", use_container_width=True)
    
# # # #     with col2:
# # # #         with st.spinner(f"AI is checking for {condition_name}..."):
# # # #             if model:
# # # #                 # Lower confidence for phone photos to catch subtle issues
# # # #                 results = model.predict(image_for_ai, conf=0.10, augment=True, verbose=False)
                
# # # #                 ai_conf = 0.0; ai_loc = "Unknown"; ai_label = "Anomaly"
# # # #                 if len(results[0].boxes) > 0:
# # # #                     box = results[0].boxes[0]
# # # #                     ai_conf = float(box.conf[0])
# # # #                     ai_loc = get_location_text(box, w, h, detected_type)
# # # #                     ai_label = model.names[int(box.cls[0])]

# # # #                 # LOGIC ENGINE
# # # #                 boost, findings, narrative = pre_pre_symptoms_solver.generate_diagnostic_reasoning(
# # # #                     syms, detected_type, ai_conf, ai_loc
# # # #                 )
# # # #                 final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
                
# # # #                 # RISK COLORS
# # # #                 if final_prob > 0.60:
# # # #                     risk_label = "HIGH PRIORITY"; risk_color = "#cc0000"; bg_color = "#ffe6e6"
# # # #                     action = "Isolate Patient & Test Sputum" if "Chest" in detected_type else "Splint & Refer to Doctor"
# # # #                 else:
# # # #                     risk_label = "ROUTINE CHECK"; risk_color = "#004085"; bg_color = "#e6f0ff"
# # # #                     action = "Watch & Wait / Check again in 2 weeks"
                
# # # #                 # DRAW OVERLAY
# # # #                 final_img = image_for_ai
# # # #                 if ai_conf > 0:
# # # #                     final_img = draw_smart_overlay(image_for_ai, box, ai_label, ai_conf, risk_label)
                
# # # #                 st.image(final_img, caption=f"AI Result: {condition_name}", use_container_width=True)
# # # #             else:
# # # #                 st.warning("⚠️ Model not loaded yet.")

# # # #     # --- E. REPORT CARD ---
# # # #     st.markdown("---")
# # # #     html_code = f"""
# # # #     <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 6px solid {risk_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
# # # #         <h3 style="color: black; margin-top: 0; font-family: sans-serif;">🤖 Diagnosis Report</h3>
# # # #         <p style="color: #333; font-size: 16px;"><b>1. What AI Saw:</b> {ai_label} found in the <span style="color: #0056b3; font-weight: bold;">{ai_loc}</span>.</p>
# # # #         <p style="color: #333; font-size: 16px;"><b>2. Clinical Story:</b> {narrative}</p>
# # # #         <p style="color: #333; font-size: 16px;"><b>3. Why?</b> AI Confidence ({ai_conf*100:.0f}%) + {len(findings)} Danger Signs.</p>
# # # #         <hr style="margin: 20px 0; border-top: 1px solid #ddd;">
# # # #         <div style="background-color: {bg_color}; padding: 20px; border-radius: 8px; text-align: center;">
# # # #             <h2 style="color: {risk_color}; margin: 0; font-weight: 900; font-size: 28px; font-family: sans-serif;">{risk_label} ({final_prob*100:.1f}%)</h2>
# # # #             <p style="color: #333; margin-top: 10px; font-style: italic; font-weight: 600;">Next Step: {action}</p>
# # # #         </div>
# # # #     </div>
# # # #     """
# # # #     st.markdown(html_code, unsafe_allow_html=True)







# # # import streamlit as st
# # # from PIL import Image, ImageDraw, ImageFont
# # # import numpy as np
# # # import tensorflow as tf
# # # from ultralytics import YOLO
# # # import cv2
# # # import pre_pre_symptoms_solver  # Logic File

# # # # --- CONFIGURATION ---
# # # st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# # # st.markdown("""
# # #     <style>
# # #     .reasoning-card { background-color: #f8f9fa; border-left: 6px solid #007bff; padding: 25px; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
# # #     .reasoning-card h3 { color: #000 !important; margin-top: 0; }
# # #     .reasoning-card p { color: #333 !important; font-size: 16px; }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # # --- LOAD MODELS ---
# # # @st.cache_resource
# # # def load_models():
# # #     try:
# # #         router = tf.keras.models.load_model('router_model.h5')
# # #         bone_model = YOLO('bone_model.pt')
# # #         tb_model = YOLO('lung_model.pt')
# # #         try: pneu_model = YOLO('pneumonia_model.pt')
# # #         except: pneu_model = None
# # #         return router, bone_model, tb_model, pneu_model
# # #     except: return None, None, None, None

# # # router, bone_model, tb_model, pneu_model = load_models()

# # # # --- HELPERS ---

# # # def enhance_phone_image(pil_image):
# # #     """Phone Mode: Fixes glare and contrast using CLAHE."""
# # #     img = np.array(pil_image)
# # #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# # #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# # #     enhanced = clahe.apply(gray)
# # #     enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
# # #     final_img = cv2.merge([enhanced, enhanced, enhanced])
# # #     return Image.fromarray(final_img)

# # # def run_tiled_inference(model, image, conf_thresh=0.10):
# # #     """
# # #     THE ZOOM ENGINE: Splits image into 4 quadrants to find hairline cracks.
# # #     Returns the best detection from the tiles OR the full image.
# # #     """
# # #     w, h = image.size
# # #     best_box = None
# # #     best_conf = 0.0
# # #     best_label = "Anomaly"
    
# # #     # 1. Run on FULL Image first
# # #     results = model.predict(image, conf=conf_thresh, augment=True, verbose=False)
# # #     if len(results[0].boxes) > 0:
# # #         best_box = results[0].boxes[0]
# # #         best_conf = float(best_box.conf[0])
# # #         best_label = model.names[int(best_box.cls[0])]

# # #     # 2. Run on TILES (Quadrants) for Zoom
# # #     # Define tiles: (left, upper, right, lower)
# # #     tiles = [
# # #         (0, 0, w//2, h//2),       # Top-Left
# # #         (w//2, 0, w, h//2),       # Top-Right
# # #         (0, h//2, w//2, h),       # Bottom-Left
# # #         (w//2, h//2, w, h)        # Bottom-Right
# # #     ]
    
# # #     for i, tile_coords in enumerate(tiles):
# # #         tile_img = image.crop(tile_coords)
# # #         t_results = model.predict(tile_img, conf=conf_thresh, verbose=False) # No augment for speed
        
# # #         if len(t_results[0].boxes) > 0:
# # #             t_box = t_results[0].boxes[0]
# # #             t_conf = float(t_box.conf[0])
            
# # #             # If this tile found something with higher confidence, keep it
# # #             if t_conf > best_conf:
# # #                 # IMPORTANT: Map tile coordinates back to Global coordinates
# # #                 x1, y1, x2, y2 = t_box.xyxy[0].tolist()
# # #                 offset_x, offset_y = tile_coords[0], tile_coords[1]
                
# # #                 # Create a fake box object with global coords (Manual fix)
# # #                 # We reuse the box structure but hack the values
# # #                 best_box = t_box
# # #                 # Store the global coordinates in a custom attribute we read later
# # #                 best_box.global_xyxy = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
# # #                 best_conf = t_conf
# # #                 best_label = model.names[int(t_box.cls[0])]
                
# # #     return best_box, best_conf, best_label

# # # def draw_smart_overlay(image, box, label, conf, risk_status):
# # #     overlay = image.copy().convert("RGBA")
# # #     draw = ImageDraw.Draw(overlay)
    
# # #     # Check if we have Tiled Global Coords (from Zoom Engine)
# # #     if hasattr(box, 'global_xyxy'):
# # #         x1, y1, x2, y2 = box.global_xyxy
# # #     else:
# # #         x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# # #     color = (255, 0, 0) if risk_status == "HIGH PRIORITY" else (0, 120, 255)
# # #     draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
# # #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # # def get_location_text(box, w, h, scan_type):
# # #     # Handle Global Coords for Tiling
# # #     if hasattr(box, 'global_xyxy'):
# # #         cx = (box.global_xyxy[0] + box.global_xyxy[2]) / 2
# # #         cy = (box.global_xyxy[1] + box.global_xyxy[3]) / 2
# # #     else:
# # #         cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
        
# # #     if scan_type == "Chest":
# # #         zone = "Upper" if cy < h/3 else "Middle" if cy < 2*h/3 else "Lower"
# # #         side = "Right" if cx < w/2 else "Left" 
# # #         return f"{side} {zone} Zone"
# # #     else:
# # #         # Bone Zones
# # #         loc_y = "Distal" if cy > h/2 else "Proximal"
# # #         return f"{loc_y} Region"

# # # # --- SIDEBAR & UI ---
# # # with st.sidebar:
# # #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# # #     st.header("Patient Intake")
    
# # #     if "symptom_state" not in st.session_state: st.session_state.symptom_state = {}

# # #     chest_mode = "Tuberculosis (TB)"
# # #     if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
# # #         chest_mode = st.radio("Protocol:", ["Tuberculosis (TB)", "Pneumonia"])
    
# # #     st.markdown("---")
# # #     st.subheader("🗣️ AI Scribe")
# # #     user_notes = st.text_area("Describe symptoms...", height=80)
    
# # #     if st.button("⚡ Auto-Fill"):
# # #         if "detected_type" in st.session_state:
# # #             detected = pre_pre_symptoms_solver.parse_symptoms_from_text(user_notes, st.session_state.detected_type)
# # #             for key in detected: st.session_state.symptom_state[key] = True
# # #             if detected: st.success(f"Detected: {', '.join(detected)}")
# # #             else: st.warning("No keywords found.")

# # #     st.markdown("---")
# # #     syms = {}
# # #     if "detected_type" in st.session_state:
# # #         dtype = st.session_state.detected_type
# # #         def smart_chk(lbl, key):
# # #             val = st.checkbox(lbl, value=st.session_state.symptom_state.get(key, False), key=key)
# # #             st.session_state.symptom_state[key] = val
# # #             return val

# # #         if dtype == "Bone":
# # #             st.caption("Fracture Signs")
# # #             syms['deformity'] = smart_chk("Deformity", 'deformity')
# # #             syms['bone_poke'] = smart_chk("Bone Exposed", 'bone_poke')
# # #             syms['cannot_move'] = smart_chk("Cannot Move", 'cannot_move')
# # #             syms['snap_sound'] = smart_chk("Heard Snap", 'snap_sound')
# # #             syms['trauma'] = smart_chk("Trauma/Fall", 'trauma')
# # #             st.markdown("---")
# # #             syms['respiratory_trap'] = smart_chk("Patient Coughing?", 'respiratory_trap')
            
# # #         elif dtype == "Chest":
# # #             st.caption("Pneumonia (Acute)")
# # #             syms['high_fever'] = smart_chk("Fever >39°C", 'high_fever')
# # #             syms['hard_breathe'] = smart_chk("Dyspnea", 'hard_breathe')
# # #             st.caption("TB (Chronic)")
# # #             syms['cough_blood'] = smart_chk("Cough Blood", 'cough_blood')
# # #             syms['weight_loss'] = smart_chk("Weight Loss", 'weight_loss')
# # #             syms['night_sweats'] = smart_chk("Night Sweats", 'night_sweats')

# # #     # Guardrails
# # #     if "detected_type" in st.session_state:
# # #         if st.session_state.detected_type == "Chest":
# # #             if chest_mode == "Tuberculosis (TB)" and syms.get('high_fever'):
# # #                 st.warning("⚠️ High Fever is usually Pneumonia, not TB.")
# # #         elif st.session_state.detected_type == "Bone" and syms.get('respiratory_trap'):
# # #             st.error("🚨 Mismatch: Respiratory signs in Bone patient.")

# # # # --- MAIN LOGIC ---
# # # st.title("Dayflow: AI Diagnosis Assistant")

# # # if not router: st.error("⚠️ Models Offline."); st.stop()

# # # uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png'])

# # # if uploaded_file:
# # #     original_image = Image.open(uploaded_file).convert('RGB')
    
# # #     col_opt1, col_opt2 = st.columns([3, 1])
# # #     with col_opt2: use_enhancer = st.checkbox("📸 Phone Mode", value=True)
    
# # #     if use_enhancer:
# # #         image_for_ai = enhance_phone_image(original_image)
# # #         st.caption("✅ Image Enhanced (CLAHE + Denoise)")
# # #     else:
# # #         image_for_ai = original_image
    
# # #     w, h = image_for_ai.size
    
# # #     # 1. ROUTER
# # #     img_arr = np.array(image_for_ai.resize((224,224)))/255.0
# # #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# # #     detected_type = ["Bone", "Chest", "Invalid"][np.argmax(pred)]
# # #     st.session_state.detected_type = detected_type
    
# # #     if detected_type == "Invalid": st.error("🚫 Invalid Image"); st.stop()
# # #     st.success(f"✅ Detected: {detected_type}")

# # #     # 2. SELECT MODEL
# # #     model = None
# # #     condition_name = "Fracture"
# # #     if detected_type == "Bone": model = bone_model
# # #     else:
# # #         if chest_mode == "Pneumonia": model = pneu_model; condition_name = "Pneumonia"
# # #         else: model = tb_model; condition_name = "Tuberculosis"

# # #     # 3. ANALYSIS
# # #     col1, col2 = st.columns(2)
# # #     with col1: st.image(original_image, caption="Original", use_container_width=True)
    
# # #     with col2:
# # #         with st.spinner(f"Analyzing {condition_name}..."):
# # #             if model:
# # #                 # --- FRACTURE ZOOM LOGIC ---
# # #                 if detected_type == "Bone":
# # #                     # Run the NEW Tiling Engine
# # #                     box, ai_conf, ai_label = run_tiled_inference(model, image_for_ai)
# # #                 else:
# # #                     # Standard Chest Scan
# # #                     results = model.predict(image_for_ai, conf=0.10, augment=True, verbose=False)
# # #                     box, ai_conf, ai_label = None, 0.0, "Anomaly"
# # #                     if len(results[0].boxes) > 0:
# # #                         box = results[0].boxes[0]
# # #                         ai_conf = float(box.conf[0])
# # #                         ai_label = model.names[int(box.cls[0])]

# # #                 ai_loc = get_location_text(box, w, h, detected_type) if ai_conf > 0 else "Unknown"

# # #                 # Reasoning
# # #                 boost, findings, narrative = pre_pre_symptoms_solver.generate_diagnostic_reasoning(
# # #                     syms, detected_type, ai_conf, ai_loc
# # #                 )
# # #                 final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
                
# # #                 # Colors
# # #                 if final_prob > 0.60:
# # #                     risk_lbl = "HIGH PRIORITY"; risk_col = "#cc0000"; bg_col = "#ffe6e6"
# # #                     act = "Isolate" if "Chest" in detected_type else "Splint"
# # #                 else:
# # #                     risk_lbl = "ROUTINE"; risk_col = "#004085"; bg_col = "#e6f0ff"
# # #                     act = "Observe"
                
# # #                 # Draw
# # #                 final_img = image_for_ai
# # #                 if ai_conf > 0:
# # #                     final_img = draw_smart_overlay(image_for_ai, box, ai_label, ai_conf, risk_lbl)
                
# # #                 st.image(final_img, caption=f"AI Result: {condition_name}", use_container_width=True)

# # #     # 4. REPORT
# # #     st.markdown("---")
# # #     html_code = f"""
# # #     <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 6px solid {risk_col};">
# # #         <h3 style="color: black; margin: 0;">🤖 Diagnosis Report</h3>
# # #         <p style="color: #333;"><b>Visual:</b> {ai_label} at <span style="color: #0056b3;">{ai_loc}</span>.</p>
# # #         <p style="color: #333;"><b>Clinical:</b> {narrative}</p>
# # #         <hr>
# # #         <div style="background-color: {bg_col}; padding: 15px; text-align: center; border-radius: 5px;">
# # #             <h2 style="color: {risk_col}; margin: 0;">{risk_lbl} ({final_prob*100:.1f}%)</h2>
# # #             <p style="color: #333; margin-top: 5px;"><b>Action:</b> {act}</p>
# # #         </div>
# # #     </div>
# # #     """
# # #     st.markdown(html_code, unsafe_allow_html=True)













# # import streamlit as st
# # from streamlit_cropper import st_cropper  # NEW LIBRARY
# # from PIL import Image, ImageDraw, ImageFont
# # import numpy as np
# # import tensorflow as tf
# # from ultralytics import YOLO
# # import cv2
# # import pre_pre_symptoms_solver  # Your Logic File

# # # --- CONFIGURATION ---
# # st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# # # --- CSS ---
# # st.markdown("""
# #     <style>
# #     .reasoning-card { background-color: #f8f9fa; border-left: 6px solid #007bff; padding: 25px; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
# #     .reasoning-card h3 { color: #000 !important; margin-top: 0; }
# #     .reasoning-card p { color: #333 !important; font-size: 16px; }
# #     </style>
# # """, unsafe_allow_html=True)

# # # --- LOAD MODELS ---
# # @st.cache_resource
# # def load_models():
# #     try:
# #         router = tf.keras.models.load_model('router_model.h5')
# #         bone_model = YOLO('bone_model.pt')
# #         tb_model = YOLO('lung_model.pt')
# #         try: pneu_model = YOLO('pneumonia_model.pt')
# #         except: pneu_model = None
# #         return router, bone_model, tb_model, pneu_model
# #     except: return None, None, None, None

# # router, bone_model, tb_model, pneu_model = load_models()

# # # --- HELPER 1: PHONE MODE ENHANCER ---
# # def enhance_phone_image(pil_image):
# #     """
# #     Fixes glare, contrast, and noise from phone photos.
# #     """
# #     # Convert PIL to OpenCV (BGR)
# #     img = np.array(pil_image)
# #     if img.shape[-1] == 4: # Handle PNG with Alpha channel
# #         img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
# #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# #     # Convert to Grayscale
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #     # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
# #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# #     enhanced = clahe.apply(gray)

# #     # Denoise
# #     enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

# #     # Convert back to fake RGB (3 channels) for YOLO
# #     final_img = cv2.merge([enhanced, enhanced, enhanced])
    
# #     return Image.fromarray(final_img)

# # # --- HELPER 2: ZOOM ENGINE (TILING) ---
# # def run_tiled_inference(model, image, conf_thresh=0.10):
# #     """
# #     Splits image into 4 quadrants to find small hairline fractures.
# #     """
# #     w, h = image.size
# #     best_box = None
# #     best_conf = 0.0
# #     best_label = "Anomaly"
    
# #     # 1. Run on FULL Image first
# #     results = model.predict(image, conf=conf_thresh, augment=True, verbose=False)
# #     if len(results[0].boxes) > 0:
# #         best_box = results[0].boxes[0]
# #         best_conf = float(best_box.conf[0])
# #         best_label = model.names[int(best_box.cls[0])]

# #     # 2. Run on TILES (Zoom)
# #     tiles = [
# #         (0, 0, w//2, h//2), (w//2, 0, w, h//2),
# #         (0, h//2, w//2, h), (w//2, h//2, w, h)
# #     ]
    
# #     for tile_coords in tiles:
# #         tile_img = image.crop(tile_coords)
# #         t_results = model.predict(tile_img, conf=conf_thresh, verbose=False)
        
# #         if len(t_results[0].boxes) > 0:
# #             t_box = t_results[0].boxes[0]
# #             t_conf = float(t_box.conf[0])
            
# #             if t_conf > best_conf:
# #                 # Map back to global coordinates
# #                 x1, y1, x2, y2 = t_box.xyxy[0].tolist()
# #                 off_x, off_y = tile_coords[0], tile_coords[1]
                
# #                 # Hack: Store global coords in the box object
# #                 best_box = t_box
# #                 best_box.global_xyxy = [x1+off_x, y1+off_y, x2+off_x, y2+off_y]
# #                 best_conf = t_conf
# #                 best_label = model.names[int(t_box.cls[0])]
                
# #     return best_box, best_conf, best_label

# # # --- HELPER 3: VISUALIZATION ---
# # def draw_smart_overlay(image, box, label, conf, risk_status):
# #     overlay = image.copy().convert("RGBA")
# #     draw = ImageDraw.Draw(overlay)
    
# #     # Handle Global Coords for Tiling
# #     if hasattr(box, 'global_xyxy'):
# #         x1, y1, x2, y2 = box.global_xyxy
# #     else:
# #         x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# #     color = (255, 0, 0) if risk_status == "HIGH PRIORITY" else (0, 120, 255)
# #     draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
# #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # def get_location_text(box, w, h, scan_type):
# #     if hasattr(box, 'global_xyxy'):
# #         cx = (box.global_xyxy[0] + box.global_xyxy[2]) / 2
# #         cy = (box.global_xyxy[1] + box.global_xyxy[3]) / 2
# #     else:
# #         cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
        
# #     if scan_type == "Chest":
# #         zone = "Upper" if cy < h/3 else "Middle" if cy < 2*h/3 else "Lower"
# #         side = "Right" if cx < w/2 else "Left" 
# #         return f"{side} {zone} Zone"
# #     else:
# #         loc_y = "Distal" if cy > h/2 else "Proximal"
# #         return f"{loc_y} Region"

# # # --- SIDEBAR ---
# # with st.sidebar:
# #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# #     st.header("Patient Intake")
    
# #     if "symptom_state" not in st.session_state: st.session_state.symptom_state = {}

# #     chest_mode = "Tuberculosis (TB)"
# #     if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
# #         chest_mode = st.radio("Protocol:", ["Tuberculosis (TB)", "Pneumonia"])
    
# #     st.markdown("---")
# #     st.subheader("🗣️ AI Scribe")
# #     user_notes = st.text_area("Describe symptoms...", height=80)
    
# #     if st.button("⚡ Auto-Fill"):
# #         if "detected_type" in st.session_state:
# #             detected = pre_pre_symptoms_solver.parse_symptoms_from_text(user_notes, st.session_state.detected_type)
# #             for key in detected: st.session_state.symptom_state[key] = True
# #             if detected: st.success(f"Detected: {', '.join(detected)}")
# #             else: st.warning("No keywords found.")

# #     st.markdown("---")
# #     syms = {}
# #     if "detected_type" in st.session_state:
# #         dtype = st.session_state.detected_type
# #         def smart_chk(lbl, key):
# #             val = st.checkbox(lbl, value=st.session_state.symptom_state.get(key, False), key=key)
# #             st.session_state.symptom_state[key] = val
# #             return val

# #         if dtype == "Bone":
# #             st.caption("Fracture Signs")
# #             syms['deformity'] = smart_chk("Deformity", 'deformity')
# #             syms['bone_poke'] = smart_chk("Bone Exposed", 'bone_poke')
# #             syms['cannot_move'] = smart_chk("Cannot Move", 'cannot_move')
# #             syms['trauma'] = smart_chk("Trauma/Fall", 'trauma')
# #             st.markdown("---")
# #             syms['respiratory_trap'] = smart_chk("Patient Coughing?", 'respiratory_trap')
            
# #         elif dtype == "Chest":
# #             st.caption("Pneumonia (Acute)")
# #             syms['high_fever'] = smart_chk("Fever >39°C", 'high_fever')
# #             syms['hard_breathe'] = smart_chk("Dyspnea", 'hard_breathe')
# #             st.caption("TB (Chronic)")
# #             syms['cough_blood'] = smart_chk("Cough Blood", 'cough_blood')
# #             syms['weight_loss'] = smart_chk("Weight Loss", 'weight_loss')

# #     # Guardrails
# #     if "detected_type" in st.session_state:
# #         if st.session_state.detected_type == "Chest":
# #             if chest_mode == "Tuberculosis (TB)" and syms.get('high_fever'):
# #                 st.warning("⚠️ High Fever is usually Pneumonia, not TB.")
# #         elif st.session_state.detected_type == "Bone" and syms.get('respiratory_trap'):
# #             st.error("🚨 Mismatch: Respiratory signs in Bone patient.")

# # # --- MAIN APP ---
# # st.title("Dayflow: AI Diagnosis Assistant")

# # if not router: st.error("⚠️ Models Offline."); st.stop()

# # uploaded_file = st.file_uploader("Upload X-Ray (Phone Photo or Scan)", type=['jpg', 'png'])

# # if uploaded_file:
# #     # 1. LOAD RAW IMAGE
# #     raw_image = Image.open(uploaded_file).convert('RGB')
    
# #     # 2. CROPPER (The "Option A" Fix)
# #     st.write("✂️ **Step 1: Crop the Image** (Remove the background!)")
    
# #     # This creates the interactive cropper box
# #     cropped_image = st_cropper(
# #         raw_image,
# #         realtime_update=True,
# #         box_color='#0000FF',
# #         aspect_ratio=None,
# #         should_resize_image=True
# #     )
    
# #     # 3. ENHANCER TOGGLE
# #     col_opt1, col_opt2 = st.columns([3, 1])
# #     with col_opt2: 
# #         use_enhancer = st.checkbox("📸 Enhance Contrast", value=True)
    
# #     if use_enhancer:
# #         # Enhance the CROPPED image, not the raw one
# #         image_for_ai = enhance_phone_image(cropped_image)
# #         st.caption("✅ Image Enhanced (CLAHE + Denoise)")
# #     else:
# #         image_for_ai = cropped_image
    
# #     w, h = image_for_ai.size
    
# #     # 4. ROUTER
# #     # We must resize for the Router (which expects 224x224)
# #     img_arr = np.array(image_for_ai.resize((224,224)))/255.0
# #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# #     detected_type = ["Bone", "Chest", "Invalid"][np.argmax(pred)]
# #     st.session_state.detected_type = detected_type
    
# #     if detected_type == "Invalid": st.error("🚫 Invalid Image"); st.stop()
# #     st.success(f"✅ Detected: {detected_type}")

# #     # 5. SELECT MODEL
# #     model = None
# #     condition_name = "Fracture"
# #     if detected_type == "Bone": model = bone_model
# #     else:
# #         if chest_mode == "Pneumonia": model = pneu_model; condition_name = "Pneumonia"
# #         else: model = tb_model; condition_name = "Tuberculosis"

# #     # 6. ANALYSIS
# #     col1, col2 = st.columns(2)
# #     with col1: 
# #         st.image(image_for_ai, caption="Processed Input", use_container_width=True)
    
# #     with col2:
# #         with st.spinner(f"Analyzing {condition_name}..."):
# #             if model:
# #                 # --- FRACTURE ZOOM LOGIC ---
# #                 if detected_type == "Bone":
# #                     box, ai_conf, ai_label = run_tiled_inference(model, image_for_ai)
# #                 else:
# #                     # Chest Logic
# #                     results = model.predict(image_for_ai, conf=0.10, augment=True, verbose=False)
# #                     box, ai_conf, ai_label = None, 0.0, "Anomaly"
# #                     if len(results[0].boxes) > 0:
# #                         box = results[0].boxes[0]
# #                         ai_conf = float(box.conf[0])
# #                         ai_label = model.names[int(box.cls[0])]

# #                 ai_loc = get_location_text(box, w, h, detected_type) if ai_conf > 0 else "Unknown"

# #                 # Reasoning
# #                 boost, findings, narrative = pre_pre_symptoms_solver.generate_diagnostic_reasoning(
# #                     syms, detected_type, ai_conf, ai_loc
# #                 )
# #                 final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
                
# #                 # Colors
# #                 if final_prob > 0.60:
# #                     risk_lbl = "HIGH PRIORITY"; risk_col = "#cc0000"; bg_col = "#ffe6e6"
# #                     act = "Isolate" if "Chest" in detected_type else "Splint"
# #                 else:
# #                     risk_lbl = "ROUTINE"; risk_col = "#004085"; bg_col = "#e6f0ff"
# #                     act = "Observe"
                
# #                 # Draw
# #                 final_img = image_for_ai
# #                 if ai_conf > 0:
# #                     final_img = draw_smart_overlay(image_for_ai, box, ai_label, ai_conf, risk_lbl)
                
# #                 st.image(final_img, caption=f"AI Result: {condition_name}", use_container_width=True)

# #     # 7. REPORT
# #     st.markdown("---")
# #     html_code = f"""
# #     <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 6px solid {risk_col};">
# #         <h3 style="color: black; margin: 0;">🤖 Diagnosis Report</h3>
# #         <p style="color: #333;"><b>Visual:</b> {ai_label} at <span style="color: #0056b3;">{ai_loc}</span>.</p>
# #         <p style="color: #333;"><b>Clinical:</b> {narrative}</p>
# #         <hr>
# #         <div style="background-color: {bg_col}; padding: 15px; text-align: center; border-radius: 5px;">
# #             <h2 style="color: {risk_col}; margin: 0;">{risk_lbl} ({final_prob*100:.1f}%)</h2>
# #             <p style="color: #333; margin-top: 5px;"><b>Action:</b> {act}</p>
# #         </div>
# #     </div>
# #     """
# #     st.markdown(html_code, unsafe_allow_html=True)





# import streamlit as st
# from streamlit_cropper import st_cropper
# from PIL import Image, ImageDraw, ImageFont, ImageFilter
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# import cv2
# import pre_pre_symptoms_solver

# # --- CONFIG ---
# st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# # --- CSS ---
# st.markdown("""
#     <style>
#     .reasoning-card { background-color: #f8f9fa; border-left: 6px solid #007bff; padding: 25px; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
#     </style>
# """, unsafe_allow_html=True)

# # --- LOAD MODELS ---
# @st.cache_resource
# def load_models():
#     try:
#         router = tf.keras.models.load_model('router_model.h5')
#         # Ensure these are your FINE-TUNED models
#         bone_model = YOLO('bone_model.pt') 
#         tb_model = YOLO('lung_model.pt')
#         try: pneu_model = YOLO('pneumonia_model.pt')
#         except: pneu_model = None
#         return router, bone_model, tb_model, pneu_model
#     except: return None, None, None, None

# router, bone_model, tb_model, pneu_model = load_models()

# # --- NEW: DUAL ENHANCEMENT ENGINES 🛠️ ---

# def enhance_film_photo(pil_image):
#     """
#     OPTIMIZED FOR FILMS (Chest/Leg images):
#     - Aggressive Contrast (CLAHE)
#     - Medium Denoising (Grain removal)
#     """
#     img = np.array(pil_image)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
#     # 1. Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 2. Strong CLAHE (Fixes the light box glare)
#     clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)) # Increased clipLimit
#     enhanced = clahe.apply(gray)

#     # 3. Medium Denoising
#     enhanced = cv2.fastNlMeansDenoising(enhanced, None, 15, 7, 21)

#     return Image.fromarray(cv2.merge([enhanced, enhanced, enhanced]))

# def enhance_screen_photo(pil_image):
#     """
#     OPTIMIZED FOR SCREENS (Toe image):
#     - Gaussian Blur (Kills the Moiré grid patterns)
#     - Sharpening (Brings bone back)
#     """
#     img = np.array(pil_image)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # 1. Gaussian Blur (The Anti-Moiré Weapon)
#     # We blur slightly to merge the screen pixels together
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # 2. Light CLAHE (Screens already have high contrast)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(blurred)
    
#     # 3. Sharpening (To fix the blur we added)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     sharpened = cv2.filter2D(enhanced, -1, kernel)

#     return Image.fromarray(cv2.merge([sharpened, sharpened, sharpened]))

# # --- HELPER: ZOOM ENGINE ---
# def run_tiled_inference(model, image, conf_thresh=0.10):
#     w, h = image.size
#     best_box = None; best_conf = 0.0; best_label = "Anomaly"
    
#     # Full Pass
#     results = model.predict(image, conf=conf_thresh, augment=True, verbose=False)
#     if len(results[0].boxes) > 0:
#         best_box = results[0].boxes[0]; best_conf = float(best_box.conf[0]); best_label = model.names[int(best_box.cls[0])]

#     # Tile Pass
#     tiles = [(0,0,w//2,h//2), (w//2,0,w,h//2), (0,h//2,w//2,h), (w//2,h//2,w,h)]
#     for tc in tiles:
#         tile_img = image.crop(tc)
#         # Lower confidence for tiles to catch small toes
#         res = model.predict(tile_img, conf=0.05, verbose=False) 
#         if len(res[0].boxes) > 0:
#             box = res[0].boxes[0]
#             if float(box.conf[0]) > best_conf:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 best_box = box
#                 best_box.global_xyxy = [x1+tc[0], y1+tc[1], x2+tc[0], y2+tc[1]]
#                 best_conf = float(box.conf[0])
#                 best_label = model.names[int(box.cls[0])]
#     return best_box, best_conf, best_label

# # --- HELPER: DRAWING ---
# def draw_smart_overlay(image, box, label, conf, risk_status):
#     overlay = image.copy().convert("RGBA")
#     draw = ImageDraw.Draw(overlay)
#     x1, y1, x2, y2 = box.global_xyxy if hasattr(box, 'global_xyxy') else box.xyxy[0].tolist()
#     color = (255, 0, 0) if risk_status == "HIGH PRIORITY" else (0, 120, 255)
#     draw.rectangle([x1, y1, x2, y2], fill=color+(60,), outline=color+(255,), width=5)
#     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# def get_location_text(box, w, h, scan_type):
#     if hasattr(box, 'global_xyxy'):
#         cx = (box.global_xyxy[0] + box.global_xyxy[2])/2
#         cy = (box.global_xyxy[1] + box.global_xyxy[3])/2
#     else:
#         cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
#     if scan_type == "Chest":
#         zone = "Upper" if cy < h/3 else "Middle" if cy < 2*h/3 else "Lower"
#         return f"{'Right' if cx < w/2 else 'Left'} {zone} Zone"
#     return "Distal Region" if cy > h/2 else "Proximal Region"

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("Patient Intake")
#     if "symptom_state" not in st.session_state: st.session_state.symptom_state = {}
    
#     chest_mode = "Tuberculosis (TB)"
#     if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
#         chest_mode = st.radio("Protocol:", ["Tuberculosis (TB)", "Pneumonia"])
    
#     st.markdown("---")
#     user_notes = st.text_area("Symptoms...", height=80)
#     if st.button("⚡ Auto-Fill"):
#         if "detected_type" in st.session_state:
#             detected = pre_pre_symptoms_solver.parse_symptoms_from_text(user_notes, st.session_state.detected_type)
#             for k in detected: st.session_state.symptom_state[k] = True

#     # Checkboxes (Simplified)
#     syms = {}
#     if "detected_type" in st.session_state:
#         dtype = st.session_state.detected_type
#         def chk(l, k): st.session_state.symptom_state[k] = st.checkbox(l, value=st.session_state.symptom_state.get(k,False), key=k); return st.session_state.symptom_state[k]
#         if dtype == "Bone":
#             syms['deformity'] = chk("Deformity", 'deformity')
#             syms['cannot_move'] = chk("Cannot Move", 'cannot_move')
#             syms['trauma'] = chk("Trauma", 'trauma')
#         elif dtype == "Chest":
#             syms['high_fever'] = chk("High Fever", 'high_fever')
#             syms['cough_blood'] = chk("Cough Blood", 'cough_blood')

# # --- MAIN APP ---
# st.title("Dayflow: AI Diagnosis Assistant")
# if not router: st.error("⚠️ Models Offline."); st.stop()

# uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png', 'jpeg'])

# if uploaded_file:
#     raw_image = Image.open(uploaded_file).convert('RGB')
    
#     # 1. SOURCE SELECTOR (The Fix!)
#     st.info("👇 **Where did this image come from?** (Crucial for accuracy)")
#     col_s1, col_s2, col_s3 = st.columns(3)
#     with col_s1:
#         source_mode = st.radio("Select Source:", ["Direct Upload (Clean)", "Photo of Film", "Photo of Screen/Monitor"])

#     # 2. CROPPER
#     st.write("✂️ **Step 2: Crop (Remove Background)**")
#     cropped_preview = st_cropper(raw_image, realtime_update=True, box_color='#0000FF', aspect_ratio=None)
    
#     if st.button("🚀 Run Analysis"):
#         # 3. APPLY SPECIFIC ENHANCEMENT
#         if source_mode == "Photo of Film":
#             image_for_ai = enhance_film_photo(cropped_preview)
#             st.caption("✅ Applied Film Correction (Glare Removal)")
#         elif source_mode == "Photo of Screen/Monitor":
#             image_for_ai = enhance_screen_photo(cropped_preview)
#             st.caption("✅ Applied Screen Correction (Moiré Removal)")
#         else:
#             image_for_ai = cropped_preview
            
#         w, h = image_for_ai.size
        
#         # 4. ROUTER
#         img_arr = np.array(image_for_ai.resize((224,224)))/255.0
#         pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
#         detected_type = ["Bone", "Chest", "Invalid"][np.argmax(pred)]
#         st.session_state.detected_type = detected_type
        
#         if detected_type == "Invalid": st.error("🚫 Invalid Image"); st.stop()
        
#         col1, col2 = st.columns(2)
#         with col1: st.image(image_for_ai, caption="AI View (Enhanced)", use_container_width=True)
        
#         # 5. INFERENCE
#         model = None; cond = "Fracture"
#         if detected_type == "Bone": model = bone_model
#         else:
#             if chest_mode == "Pneumonia": model = pneu_model; cond = "Pneumonia"
#             else: model = tb_model; cond = "TB"
            
#         with col2:
#             with st.spinner(f"Scanning for {cond}..."):
#                 if detected_type == "Bone":
#                     # Run Zoom Engine
#                     box, ai_conf, ai_label = run_tiled_inference(model, image_for_ai)
#                 else:
#                     res = model.predict(image_for_ai, conf=0.10, augment=True, verbose=False)
#                     box, ai_conf, ai_label = None, 0.0, "Anomaly"
#                     if len(res[0].boxes)>0: 
#                         box = res[0].boxes[0]; ai_conf = float(box.conf[0]); ai_label = model.names[int(box.cls[0])]
                
#                 # Report
#                 ai_loc = get_location_text(box, w, h, detected_type) if ai_conf > 0 else "Unknown"
#                 boost, findings, narrative = pre_pre_symptoms_solver.generate_diagnostic_reasoning(syms, detected_type, ai_conf, ai_loc)
#                 final_prob = min(ai_conf+boost, 1.0) if ai_conf > 0 else 0.0
                
#                 if final_prob > 0.6: 
#                     lbl="HIGH PRIORITY"; clr="#cc0000"; bg="#ffe6e6"; act="Splint & Refer"
#                 else: 
#                     lbl="ROUTINE"; clr="#004085"; bg="#e6f0ff"; act="Observe"
                
#                 final_img = image_for_ai
#                 if ai_conf > 0: final_img = draw_smart_overlay(image_for_ai, box, ai_label, ai_conf, lbl)
#                 st.image(final_img, caption="AI Result", use_container_width=True)
                
#         # 6. REPORT
#         st.markdown(f"""
#         <div style="background-color:#f9f9f9; padding:20px; border-left:6px solid {clr};">
#             <h3 style="color:black;">Diagnosis Report</h3>
#             <p style="color:#333;"><b>Finding:</b> {narrative}</p>
#             <div style="background-color:{bg}; padding:10px; text-align:center;">
#                 <h2 style="color:{clr};">{lbl} ({final_prob*100:.1f}%)</h2>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)







# # # import streamlit as st
# # # from PIL import Image, ImageDraw, ImageFont
# # # import numpy as np
# # # import tensorflow as tf
# # # from ultralytics import YOLO
# # # import cv2
# # # import pre_pre_symptoms_solver  # Importing the logic file above

# # # # --- CONFIGURATION ---
# # # st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# # # # --- CSS (Clean & Readable) ---
# # # st.markdown("""
# # #     <style>
# # #     .reasoning-card {
# # #         background-color: #f8f9fa; 
# # #         border-left: 6px solid #007bff;
# # #         padding: 25px;
# # #         border-radius: 8px;
# # #         margin-top: 20px;
# # #         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
# # #     }
# # #     .reasoning-card h3 { color: #000 !important; margin-top: 0; }
# # #     .reasoning-card p { color: #333 !important; font-size: 16px; }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # # --- LOAD MODELS ---
# # # @st.cache_resource
# # # def load_models():
# # #     try:
# # #         router = tf.keras.models.load_model('router_model.h5')
# # #         bone_model = YOLO('bone_model.pt')
# # #         tb_model = YOLO('lung_model.pt')
# # #         # Only load pneumonia if it exists (Kaggle download might be pending)
# # #         try:
# # #             pneu_model = YOLO('pneumonia_model.pt')
# # #         except:
# # #             pneu_model = None
            
# # #         return router, bone_model, tb_model, pneu_model
# # #     except:
# # #         return None, None, None, None

# # # router, bone_model, tb_model, pneu_model = load_models()

# # # # --- HELPER 1: PHONE MODE ENHANCER ---
# # # def enhance_phone_image(pil_image):
# # #     """
# # #     Turns a messy phone photo into a clean, high-contrast X-ray.
# # #     """
# # #     img = np.array(pil_image)
# # #     # Convert RGB to BGR for OpenCV
# # #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# # #     # Convert to Grayscale
# # #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # #     # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
# # #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# # #     enhanced = clahe.apply(gray)

# # #     # Denoise
# # #     enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

# # #     # Convert back to fake RGB (3 channels) for YOLO
# # #     final_img = cv2.merge([enhanced, enhanced, enhanced])
    
# # #     return Image.fromarray(final_img)

# # # # --- HELPER 2: VISUALIZATION ---
# # # def draw_smart_overlay(image, box, label, conf, risk_status):
# # #     overlay = image.copy().convert("RGBA")
# # #     draw = ImageDraw.Draw(overlay)
# # #     x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# # #     if risk_status == "HIGH PRIORITY":
# # #         color = (255, 0, 0) # Red
# # #     else:
# # #         color = (0, 120, 255) # Blue
    
# # #     draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
# # #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # # def get_location_text(box, w, h, scan_type):
# # #     cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
# # #     if scan_type == "Chest":
# # #         zone = "Upper Chest" if cy < h/3 else "Middle Chest" if cy < 2*h/3 else "Lower Chest"
# # #         side = "Right" if cx < w/2 else "Left" 
# # #         return f"{side} {zone}"
# # #     else:
# # #         return "Bone/Limb Area"

# # # # --- SIDEBAR LOGIC (AI SCRIBE & GUARDRAILS) ---
# # # with st.sidebar:
# # #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# # #     st.header("Patient Intake")
    
# # #     # 1. INITIALIZE STATE
# # #     if "symptom_state" not in st.session_state:
# # #         st.session_state.symptom_state = {}

# # #     # 2. MODE SELECTION
# # #     chest_mode = "Tuberculosis (TB)"
# # #     if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
# # #         st.subheader("What do you want to check?")
# # #         chest_mode = st.radio("Select Protocol:", ["Tuberculosis (TB)", "Pneumonia"])
    
# # #     st.markdown("---")
    
# # #     # 3. AI SCRIBE TEXT BOX
# # #     st.subheader("🗣️ AI Scribe")
# # #     user_notes = st.text_area("Type symptoms (e.g., 'Coughing blood since tuesday')", height=80)
    
# # #     if st.button("⚡ Auto-Fill Form"):
# # #         if "detected_type" in st.session_state:
# # #             detected = pre_pre_symptoms_solver.parse_symptoms_from_text(user_notes, st.session_state.detected_type)
# # #             for key in detected:
# # #                 st.session_state.symptom_state[key] = True
# # #             if detected:
# # #                 st.success(f"AI detected: {', '.join(detected)}")
# # #             else:
# # #                 st.warning("No keywords found. Try simpler terms.")

# # #     st.markdown("---")
    
# # #     # 4. SMART CHECKLIST
# # #     syms = {}
# # #     if "detected_type" in st.session_state:
# # #         dtype = st.session_state.detected_type
        
# # #         def smart_checkbox(label, key):
# # #             is_checked = st.session_state.symptom_state.get(key, False)
# # #             val = st.checkbox(label, value=is_checked, key=key)
# # #             st.session_state.symptom_state[key] = val
# # #             return val

# # #         if dtype == "Bone":
# # #             st.caption("Broken Bone Signs")
# # #             syms['deformity'] = smart_checkbox("Bone looks bent / wrong shape", 'deformity')
# # #             syms['bone_poke'] = smart_checkbox("Bone poking through skin", 'bone_poke')
# # #             syms['cannot_move'] = smart_checkbox("Cannot move the part", 'cannot_move')
# # #             syms['snap_sound'] = smart_checkbox("Heard a 'Snap' or 'Crack'", 'snap_sound')
# # #             syms['swelling'] = smart_checkbox("Bad swelling or bruising", 'swelling')
# # #             syms['trauma'] = smart_checkbox("Hit by something hard / Fell down", 'trauma')
# # #             st.markdown("---")
# # #             syms['respiratory_trap'] = smart_checkbox("Patient is also coughing?", 'respiratory_trap')
            
# # #         elif dtype == "Chest":
# # #             st.caption("Pneumonia Signs (Fast & Hot)")
# # #             syms['high_fever'] = smart_checkbox("Very Hot Fever (>39°C)", 'high_fever')
# # #             syms['shaking_chills'] = smart_checkbox("Shaking Chills / Shivering", 'shaking_chills')
# # #             syms['hard_breathe'] = smart_checkbox("Hard to breathe / Gasping", 'hard_breathe')
# # #             syms['green_phlegm'] = smart_checkbox("Coughing green/yellow slime", 'green_phlegm')
            
# # #             st.caption("TB Signs (Slow & Wasting)")
# # #             syms['cough_blood'] = smart_checkbox("Coughing up Blood", 'cough_blood')
# # #             syms['weight_loss'] = smart_checkbox("Losing weight without trying", 'weight_loss')
# # #             syms['night_sweats'] = smart_checkbox("Sweating huge amounts at night", 'night_sweats')
# # #             syms['long_cough'] = smart_checkbox("Cough >3 weeks", 'long_cough')

# # #     else:
# # #         st.info("Upload X-Ray to unlock.")

# # #     # 5. GUARDRAIL ALERTS
# # #     if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
# # #         warning_msg = None
# # #         if chest_mode == "Tuberculosis (TB)" and (syms.get('high_fever') or syms.get('hard_breathe')):
# # #             warning_msg = "⚠️ **Wait!** High Fever/Gasping are **Pneumonia** signs. \n\n👉 Suggested: Switch to **Pneumonia** above."
# # #         elif chest_mode == "Pneumonia" and (syms.get('cough_blood') or syms.get('weight_loss')):
# # #             warning_msg = "⚠️ **Wait!** Coughing Blood/Weight Loss are **TB** signs. \n\n👉 Suggested: Switch to **Tuberculosis** above."
# # #         if warning_msg: st.warning(warning_msg)

# # #     if "detected_type" in st.session_state and st.session_state.detected_type == "Bone":
# # #         if syms.get('respiratory_trap'):
# # #             st.error("🚨 **Error:** BONE X-ray but patient is coughing? Check patient ID.")

# # # # --- MAIN APP ---
# # # st.title("Dayflow: AI Diagnosis Assistant")

# # # if not router:
# # #     st.error("⚠️ AI Models are offline.")
# # #     st.stop()

# # # uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png', 'jpeg'])

# # # if uploaded_file:
# # #     original_image = Image.open(uploaded_file).convert('RGB')
    
# # #     # --- A. PHONE MODE TOGGLE ---
# # #     col_opt1, col_opt2 = st.columns([3, 1])
# # #     with col_opt2:
# # #         use_enhancer = st.checkbox("📸 Phone Mode", value=True, help="Fixes glare/blur from phone photos.")
    
# # #     if use_enhancer:
# # #         image_for_ai = enhance_phone_image(original_image)
# # #         st.caption("✅ Image automatically enhanced for clarity.")
# # #     else:
# # #         image_for_ai = original_image
    
# # #     w, h = image_for_ai.size
    
# # #     # --- B. ROUTER ---
# # #     img_arr = np.array(image_for_ai.resize((224,224)))/255.0
# # #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# # #     route_idx = np.argmax(pred)
# # #     detected_type = ["Bone", "Chest", "Invalid"][route_idx]
# # #     st.session_state.detected_type = detected_type
    
# # #     if detected_type == "Invalid":
# # #         st.error("🚫 This doesn't look like an X-ray."); st.stop()

# # #     st.success(f"✅ **Detected:** {detected_type} Scan")

# # #     # --- C. MODEL SELECTION ---
# # #     condition_name = "Fracture"
# # #     model = None
    
# # #     if detected_type == "Bone":
# # #         model = bone_model
# # #     else:
# # #         if chest_mode == "Pneumonia":
# # #             model = pneu_model; condition_name = "Pneumonia"
# # #         else:
# # #             model = tb_model; condition_name = "Tuberculosis"

# # #     # --- D. ANALYSIS ---
# # #     col1, col2 = st.columns(2)
# # #     with col1: 
# # #         st.image(original_image, caption="Original Upload", use_container_width=True)
    
# # #     with col2:
# # #         with st.spinner(f"AI is checking for {condition_name}..."):
# # #             if model:
# # #                 # Lower confidence for phone photos to catch subtle issues
# # #                 results = model.predict(image_for_ai, conf=0.10, augment=True, verbose=False)
                
# # #                 ai_conf = 0.0; ai_loc = "Unknown"; ai_label = "Anomaly"
# # #                 if len(results[0].boxes) > 0:
# # #                     box = results[0].boxes[0]
# # #                     ai_conf = float(box.conf[0])
# # #                     ai_loc = get_location_text(box, w, h, detected_type)
# # #                     ai_label = model.names[int(box.cls[0])]

# # #                 # LOGIC ENGINE
# # #                 boost, findings, narrative = pre_pre_symptoms_solver.generate_diagnostic_reasoning(
# # #                     syms, detected_type, ai_conf, ai_loc
# # #                 )
# # #                 final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
                
# # #                 # RISK COLORS
# # #                 if final_prob > 0.60:
# # #                     risk_label = "HIGH PRIORITY"; risk_color = "#cc0000"; bg_color = "#ffe6e6"
# # #                     action = "Isolate Patient & Test Sputum" if "Chest" in detected_type else "Splint & Refer to Doctor"
# # #                 else:
# # #                     risk_label = "ROUTINE CHECK"; risk_color = "#004085"; bg_color = "#e6f0ff"
# # #                     action = "Watch & Wait / Check again in 2 weeks"
                
# # #                 # DRAW OVERLAY
# # #                 final_img = image_for_ai
# # #                 if ai_conf > 0:
# # #                     final_img = draw_smart_overlay(image_for_ai, box, ai_label, ai_conf, risk_label)
                
# # #                 st.image(final_img, caption=f"AI Result: {condition_name}", use_container_width=True)
# # #             else:
# # #                 st.warning("⚠️ Model not loaded yet.")

# # #     # --- E. REPORT CARD ---
# # #     st.markdown("---")
# # #     html_code = f"""
# # #     <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 6px solid {risk_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
# # #         <h3 style="color: black; margin-top: 0; font-family: sans-serif;">🤖 Diagnosis Report</h3>
# # #         <p style="color: #333; font-size: 16px;"><b>1. What AI Saw:</b> {ai_label} found in the <span style="color: #0056b3; font-weight: bold;">{ai_loc}</span>.</p>
# # #         <p style="color: #333; font-size: 16px;"><b>2. Clinical Story:</b> {narrative}</p>
# # #         <p style="color: #333; font-size: 16px;"><b>3. Why?</b> AI Confidence ({ai_conf*100:.0f}%) + {len(findings)} Danger Signs.</p>
# # #         <hr style="margin: 20px 0; border-top: 1px solid #ddd;">
# # #         <div style="background-color: {bg_color}; padding: 20px; border-radius: 8px; text-align: center;">
# # #             <h2 style="color: {risk_color}; margin: 0; font-weight: 900; font-size: 28px; font-family: sans-serif;">{risk_label} ({final_prob*100:.1f}%)</h2>
# # #             <p style="color: #333; margin-top: 10px; font-style: italic; font-weight: 600;">Next Step: {action}</p>
# # #         </div>
# # #     </div>
# # #     """
# # #     st.markdown(html_code, unsafe_allow_html=True)







# # import streamlit as st
# # from PIL import Image, ImageDraw, ImageFont
# # import numpy as np
# # import tensorflow as tf
# # from ultralytics import YOLO
# # import cv2
# # import pre_pre_symptoms_solver  # Logic File

# # # --- CONFIGURATION ---
# # st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# # st.markdown("""
# #     <style>
# #     .reasoning-card { background-color: #f8f9fa; border-left: 6px solid #007bff; padding: 25px; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
# #     .reasoning-card h3 { color: #000 !important; margin-top: 0; }
# #     .reasoning-card p { color: #333 !important; font-size: 16px; }
# #     </style>
# # """, unsafe_allow_html=True)

# # # --- LOAD MODELS ---
# # @st.cache_resource
# # def load_models():
# #     try:
# #         router = tf.keras.models.load_model('router_model.h5')
# #         bone_model = YOLO('bone_model.pt')
# #         tb_model = YOLO('lung_model.pt')
# #         try: pneu_model = YOLO('pneumonia_model.pt')
# #         except: pneu_model = None
# #         return router, bone_model, tb_model, pneu_model
# #     except: return None, None, None, None

# # router, bone_model, tb_model, pneu_model = load_models()

# # # --- HELPERS ---

# # def enhance_phone_image(pil_image):
# #     """Phone Mode: Fixes glare and contrast using CLAHE."""
# #     img = np.array(pil_image)
# #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# #     enhanced = clahe.apply(gray)
# #     enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
# #     final_img = cv2.merge([enhanced, enhanced, enhanced])
# #     return Image.fromarray(final_img)

# # def run_tiled_inference(model, image, conf_thresh=0.10):
# #     """
# #     THE ZOOM ENGINE: Splits image into 4 quadrants to find hairline cracks.
# #     Returns the best detection from the tiles OR the full image.
# #     """
# #     w, h = image.size
# #     best_box = None
# #     best_conf = 0.0
# #     best_label = "Anomaly"
    
# #     # 1. Run on FULL Image first
# #     results = model.predict(image, conf=conf_thresh, augment=True, verbose=False)
# #     if len(results[0].boxes) > 0:
# #         best_box = results[0].boxes[0]
# #         best_conf = float(best_box.conf[0])
# #         best_label = model.names[int(best_box.cls[0])]

# #     # 2. Run on TILES (Quadrants) for Zoom
# #     # Define tiles: (left, upper, right, lower)
# #     tiles = [
# #         (0, 0, w//2, h//2),       # Top-Left
# #         (w//2, 0, w, h//2),       # Top-Right
# #         (0, h//2, w//2, h),       # Bottom-Left
# #         (w//2, h//2, w, h)        # Bottom-Right
# #     ]
    
# #     for i, tile_coords in enumerate(tiles):
# #         tile_img = image.crop(tile_coords)
# #         t_results = model.predict(tile_img, conf=conf_thresh, verbose=False) # No augment for speed
        
# #         if len(t_results[0].boxes) > 0:
# #             t_box = t_results[0].boxes[0]
# #             t_conf = float(t_box.conf[0])
            
# #             # If this tile found something with higher confidence, keep it
# #             if t_conf > best_conf:
# #                 # IMPORTANT: Map tile coordinates back to Global coordinates
# #                 x1, y1, x2, y2 = t_box.xyxy[0].tolist()
# #                 offset_x, offset_y = tile_coords[0], tile_coords[1]
                
# #                 # Create a fake box object with global coords (Manual fix)
# #                 # We reuse the box structure but hack the values
# #                 best_box = t_box
# #                 # Store the global coordinates in a custom attribute we read later
# #                 best_box.global_xyxy = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
# #                 best_conf = t_conf
# #                 best_label = model.names[int(t_box.cls[0])]
                
# #     return best_box, best_conf, best_label

# # def draw_smart_overlay(image, box, label, conf, risk_status):
# #     overlay = image.copy().convert("RGBA")
# #     draw = ImageDraw.Draw(overlay)
    
# #     # Check if we have Tiled Global Coords (from Zoom Engine)
# #     if hasattr(box, 'global_xyxy'):
# #         x1, y1, x2, y2 = box.global_xyxy
# #     else:
# #         x1, y1, x2, y2 = box.xyxy[0].tolist()
    
# #     color = (255, 0, 0) if risk_status == "HIGH PRIORITY" else (0, 120, 255)
# #     draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
# #     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# # def get_location_text(box, w, h, scan_type):
# #     # Handle Global Coords for Tiling
# #     if hasattr(box, 'global_xyxy'):
# #         cx = (box.global_xyxy[0] + box.global_xyxy[2]) / 2
# #         cy = (box.global_xyxy[1] + box.global_xyxy[3]) / 2
# #     else:
# #         cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
        
# #     if scan_type == "Chest":
# #         zone = "Upper" if cy < h/3 else "Middle" if cy < 2*h/3 else "Lower"
# #         side = "Right" if cx < w/2 else "Left" 
# #         return f"{side} {zone} Zone"
# #     else:
# #         # Bone Zones
# #         loc_y = "Distal" if cy > h/2 else "Proximal"
# #         return f"{loc_y} Region"

# # # --- SIDEBAR & UI ---
# # with st.sidebar:
# #     st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
# #     st.header("Patient Intake")
    
# #     if "symptom_state" not in st.session_state: st.session_state.symptom_state = {}

# #     chest_mode = "Tuberculosis (TB)"
# #     if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
# #         chest_mode = st.radio("Protocol:", ["Tuberculosis (TB)", "Pneumonia"])
    
# #     st.markdown("---")
# #     st.subheader("🗣️ AI Scribe")
# #     user_notes = st.text_area("Describe symptoms...", height=80)
    
# #     if st.button("⚡ Auto-Fill"):
# #         if "detected_type" in st.session_state:
# #             detected = pre_pre_symptoms_solver.parse_symptoms_from_text(user_notes, st.session_state.detected_type)
# #             for key in detected: st.session_state.symptom_state[key] = True
# #             if detected: st.success(f"Detected: {', '.join(detected)}")
# #             else: st.warning("No keywords found.")

# #     st.markdown("---")
# #     syms = {}
# #     if "detected_type" in st.session_state:
# #         dtype = st.session_state.detected_type
# #         def smart_chk(lbl, key):
# #             val = st.checkbox(lbl, value=st.session_state.symptom_state.get(key, False), key=key)
# #             st.session_state.symptom_state[key] = val
# #             return val

# #         if dtype == "Bone":
# #             st.caption("Fracture Signs")
# #             syms['deformity'] = smart_chk("Deformity", 'deformity')
# #             syms['bone_poke'] = smart_chk("Bone Exposed", 'bone_poke')
# #             syms['cannot_move'] = smart_chk("Cannot Move", 'cannot_move')
# #             syms['snap_sound'] = smart_chk("Heard Snap", 'snap_sound')
# #             syms['trauma'] = smart_chk("Trauma/Fall", 'trauma')
# #             st.markdown("---")
# #             syms['respiratory_trap'] = smart_chk("Patient Coughing?", 'respiratory_trap')
            
# #         elif dtype == "Chest":
# #             st.caption("Pneumonia (Acute)")
# #             syms['high_fever'] = smart_chk("Fever >39°C", 'high_fever')
# #             syms['hard_breathe'] = smart_chk("Dyspnea", 'hard_breathe')
# #             st.caption("TB (Chronic)")
# #             syms['cough_blood'] = smart_chk("Cough Blood", 'cough_blood')
# #             syms['weight_loss'] = smart_chk("Weight Loss", 'weight_loss')
# #             syms['night_sweats'] = smart_chk("Night Sweats", 'night_sweats')

# #     # Guardrails
# #     if "detected_type" in st.session_state:
# #         if st.session_state.detected_type == "Chest":
# #             if chest_mode == "Tuberculosis (TB)" and syms.get('high_fever'):
# #                 st.warning("⚠️ High Fever is usually Pneumonia, not TB.")
# #         elif st.session_state.detected_type == "Bone" and syms.get('respiratory_trap'):
# #             st.error("🚨 Mismatch: Respiratory signs in Bone patient.")

# # # --- MAIN LOGIC ---
# # st.title("Dayflow: AI Diagnosis Assistant")

# # if not router: st.error("⚠️ Models Offline."); st.stop()

# # uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png'])

# # if uploaded_file:
# #     original_image = Image.open(uploaded_file).convert('RGB')
    
# #     col_opt1, col_opt2 = st.columns([3, 1])
# #     with col_opt2: use_enhancer = st.checkbox("📸 Phone Mode", value=True)
    
# #     if use_enhancer:
# #         image_for_ai = enhance_phone_image(original_image)
# #         st.caption("✅ Image Enhanced (CLAHE + Denoise)")
# #     else:
# #         image_for_ai = original_image
    
# #     w, h = image_for_ai.size
    
# #     # 1. ROUTER
# #     img_arr = np.array(image_for_ai.resize((224,224)))/255.0
# #     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
# #     detected_type = ["Bone", "Chest", "Invalid"][np.argmax(pred)]
# #     st.session_state.detected_type = detected_type
    
# #     if detected_type == "Invalid": st.error("🚫 Invalid Image"); st.stop()
# #     st.success(f"✅ Detected: {detected_type}")

# #     # 2. SELECT MODEL
# #     model = None
# #     condition_name = "Fracture"
# #     if detected_type == "Bone": model = bone_model
# #     else:
# #         if chest_mode == "Pneumonia": model = pneu_model; condition_name = "Pneumonia"
# #         else: model = tb_model; condition_name = "Tuberculosis"

# #     # 3. ANALYSIS
# #     col1, col2 = st.columns(2)
# #     with col1: st.image(original_image, caption="Original", use_container_width=True)
    
# #     with col2:
# #         with st.spinner(f"Analyzing {condition_name}..."):
# #             if model:
# #                 # --- FRACTURE ZOOM LOGIC ---
# #                 if detected_type == "Bone":
# #                     # Run the NEW Tiling Engine
# #                     box, ai_conf, ai_label = run_tiled_inference(model, image_for_ai)
# #                 else:
# #                     # Standard Chest Scan
# #                     results = model.predict(image_for_ai, conf=0.10, augment=True, verbose=False)
# #                     box, ai_conf, ai_label = None, 0.0, "Anomaly"
# #                     if len(results[0].boxes) > 0:
# #                         box = results[0].boxes[0]
# #                         ai_conf = float(box.conf[0])
# #                         ai_label = model.names[int(box.cls[0])]

# #                 ai_loc = get_location_text(box, w, h, detected_type) if ai_conf > 0 else "Unknown"

# #                 # Reasoning
# #                 boost, findings, narrative = pre_pre_symptoms_solver.generate_diagnostic_reasoning(
# #                     syms, detected_type, ai_conf, ai_loc
# #                 )
# #                 final_prob = min(ai_conf + boost, 1.0) if ai_conf > 0 else 0.0
                
# #                 # Colors
# #                 if final_prob > 0.60:
# #                     risk_lbl = "HIGH PRIORITY"; risk_col = "#cc0000"; bg_col = "#ffe6e6"
# #                     act = "Isolate" if "Chest" in detected_type else "Splint"
# #                 else:
# #                     risk_lbl = "ROUTINE"; risk_col = "#004085"; bg_col = "#e6f0ff"
# #                     act = "Observe"
                
# #                 # Draw
# #                 final_img = image_for_ai
# #                 if ai_conf > 0:
# #                     final_img = draw_smart_overlay(image_for_ai, box, ai_label, ai_conf, risk_lbl)
                
# #                 st.image(final_img, caption=f"AI Result: {condition_name}", use_container_width=True)

# #     # 4. REPORT
# #     st.markdown("---")
# #     html_code = f"""
# #     <div style="background-color: #f9f9f9; padding: 25px; border-radius: 10px; border-left: 6px solid {risk_col};">
# #         <h3 style="color: black; margin: 0;">🤖 Diagnosis Report</h3>
# #         <p style="color: #333;"><b>Visual:</b> {ai_label} at <span style="color: #0056b3;">{ai_loc}</span>.</p>
# #         <p style="color: #333;"><b>Clinical:</b> {narrative}</p>
# #         <hr>
# #         <div style="background-color: {bg_col}; padding: 15px; text-align: center; border-radius: 5px;">
# #             <h2 style="color: {risk_col}; margin: 0;">{risk_lbl} ({final_prob*100:.1f}%)</h2>
# #             <p style="color: #333; margin-top: 5px;"><b>Action:</b> {act}</p>
# #         </div>
# #     </div>
# #     """
# #     st.markdown(html_code, unsafe_allow_html=True)













# import streamlit as st
# from streamlit_cropper import st_cropper  # NEW LIBRARY
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# import cv2
# import pre_pre_symptoms_solver  # Your Logic File

# # --- CONFIGURATION ---
# st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# # --- CSS ---
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

# # --- HELPER 1: PHONE MODE ENHANCER ---
# def enhance_phone_image(pil_image):
#     """
#     Fixes glare, contrast, and noise from phone photos.
#     """
#     # Convert PIL to OpenCV (BGR)
#     img = np.array(pil_image)
#     if img.shape[-1] == 4: # Handle PNG with Alpha channel
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     # Convert to Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)

#     # Denoise
#     enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

#     # Convert back to fake RGB (3 channels) for YOLO
#     final_img = cv2.merge([enhanced, enhanced, enhanced])
    
#     return Image.fromarray(final_img)

# # --- HELPER 2: ZOOM ENGINE (TILING) ---
# def run_tiled_inference(model, image, conf_thresh=0.10):
#     """
#     Splits image into 4 quadrants to find small hairline fractures.
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

#     # 2. Run on TILES (Zoom)
#     tiles = [
#         (0, 0, w//2, h//2), (w//2, 0, w, h//2),
#         (0, h//2, w//2, h), (w//2, h//2, w, h)
#     ]
    
#     for tile_coords in tiles:
#         tile_img = image.crop(tile_coords)
#         t_results = model.predict(tile_img, conf=conf_thresh, verbose=False)
        
#         if len(t_results[0].boxes) > 0:
#             t_box = t_results[0].boxes[0]
#             t_conf = float(t_box.conf[0])
            
#             if t_conf > best_conf:
#                 # Map back to global coordinates
#                 x1, y1, x2, y2 = t_box.xyxy[0].tolist()
#                 off_x, off_y = tile_coords[0], tile_coords[1]
                
#                 # Hack: Store global coords in the box object
#                 best_box = t_box
#                 best_box.global_xyxy = [x1+off_x, y1+off_y, x2+off_x, y2+off_y]
#                 best_conf = t_conf
#                 best_label = model.names[int(t_box.cls[0])]
                
#     return best_box, best_conf, best_label

# # --- HELPER 3: VISUALIZATION ---
# def draw_smart_overlay(image, box, label, conf, risk_status):
#     overlay = image.copy().convert("RGBA")
#     draw = ImageDraw.Draw(overlay)
    
#     # Handle Global Coords for Tiling
#     if hasattr(box, 'global_xyxy'):
#         x1, y1, x2, y2 = box.global_xyxy
#     else:
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
    
#     color = (255, 0, 0) if risk_status == "HIGH PRIORITY" else (0, 120, 255)
#     draw.rectangle([x1, y1, x2, y2], fill=color + (70,), outline=color + (255,), width=4)
#     return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# def get_location_text(box, w, h, scan_type):
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
#         loc_y = "Distal" if cy > h/2 else "Proximal"
#         return f"{loc_y} Region"

# # --- SIDEBAR ---
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

#     # Guardrails
#     if "detected_type" in st.session_state:
#         if st.session_state.detected_type == "Chest":
#             if chest_mode == "Tuberculosis (TB)" and syms.get('high_fever'):
#                 st.warning("⚠️ High Fever is usually Pneumonia, not TB.")
#         elif st.session_state.detected_type == "Bone" and syms.get('respiratory_trap'):
#             st.error("🚨 Mismatch: Respiratory signs in Bone patient.")

# # --- MAIN APP ---
# st.title("Dayflow: AI Diagnosis Assistant")

# if not router: st.error("⚠️ Models Offline."); st.stop()

# uploaded_file = st.file_uploader("Upload X-Ray (Phone Photo or Scan)", type=['jpg', 'png'])

# if uploaded_file:
#     # 1. LOAD RAW IMAGE
#     raw_image = Image.open(uploaded_file).convert('RGB')
    
#     # 2. CROPPER (The "Option A" Fix)
#     st.write("✂️ **Step 1: Crop the Image** (Remove the background!)")
    
#     # This creates the interactive cropper box
#     cropped_image = st_cropper(
#         raw_image,
#         realtime_update=True,
#         box_color='#0000FF',
#         aspect_ratio=None,
#         should_resize_image=True
#     )
    
#     # 3. ENHANCER TOGGLE
#     col_opt1, col_opt2 = st.columns([3, 1])
#     with col_opt2: 
#         use_enhancer = st.checkbox("📸 Enhance Contrast", value=True)
    
#     if use_enhancer:
#         # Enhance the CROPPED image, not the raw one
#         image_for_ai = enhance_phone_image(cropped_image)
#         st.caption("✅ Image Enhanced (CLAHE + Denoise)")
#     else:
#         image_for_ai = cropped_image
    
#     w, h = image_for_ai.size
    
#     # 4. ROUTER
#     # We must resize for the Router (which expects 224x224)
#     img_arr = np.array(image_for_ai.resize((224,224)))/255.0
#     pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
#     detected_type = ["Bone", "Chest", "Invalid"][np.argmax(pred)]
#     st.session_state.detected_type = detected_type
    
#     if detected_type == "Invalid": st.error("🚫 Invalid Image"); st.stop()
#     st.success(f"✅ Detected: {detected_type}")

#     # 5. SELECT MODEL
#     model = None
#     condition_name = "Fracture"
#     if detected_type == "Bone": model = bone_model
#     else:
#         if chest_mode == "Pneumonia": model = pneu_model; condition_name = "Pneumonia"
#         else: model = tb_model; condition_name = "Tuberculosis"

#     # 6. ANALYSIS
#     col1, col2 = st.columns(2)
#     with col1: 
#         st.image(image_for_ai, caption="Processed Input", use_container_width=True)
    
#     with col2:
#         with st.spinner(f"Analyzing {condition_name}..."):
#             if model:
#                 # --- FRACTURE ZOOM LOGIC ---
#                 if detected_type == "Bone":
#                     box, ai_conf, ai_label = run_tiled_inference(model, image_for_ai)
#                 else:
#                     # Chest Logic
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

#     # 7. REPORT
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





import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import cv2
import pre_pre_symptoms_solver

# --- CONFIG ---
st.set_page_config(page_title="Dayflow Unified AI", page_icon="🏥", layout="wide")

# --- CSS ---
st.markdown("""
    <style>
    .reasoning-card { background-color: #f8f9fa; border-left: 6px solid #007bff; padding: 25px; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        router = tf.keras.models.load_model('router_model.h5')
        # Ensure these are your FINE-TUNED models
        bone_model = YOLO('bone_model.pt') 
        tb_model = YOLO('lung_model.pt')
        try: pneu_model = YOLO('pneumonia_model.pt')
        except: pneu_model = None
        return router, bone_model, tb_model, pneu_model
    except: return None, None, None, None

router, bone_model, tb_model, pneu_model = load_models()

# --- NEW: DUAL ENHANCEMENT ENGINES 🛠️ ---

def enhance_film_photo(pil_image):
    """
    OPTIMIZED FOR FILMS (Chest/Leg images):
    - Aggressive Contrast (CLAHE)
    - Medium Denoising (Grain removal)
    """
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Strong CLAHE (Fixes the light box glare)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)) # Increased clipLimit
    enhanced = clahe.apply(gray)

    # 3. Medium Denoising
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 15, 7, 21)

    return Image.fromarray(cv2.merge([enhanced, enhanced, enhanced]))

def enhance_screen_photo(pil_image):
    """
    OPTIMIZED FOR SCREENS (Toe image):
    - Gaussian Blur (Kills the Moiré grid patterns)
    - Sharpening (Brings bone back)
    """
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Gaussian Blur (The Anti-Moiré Weapon)
    # We blur slightly to merge the screen pixels together
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Light CLAHE (Screens already have high contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # 3. Sharpening (To fix the blur we added)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return Image.fromarray(cv2.merge([sharpened, sharpened, sharpened]))

# --- HELPER: ZOOM ENGINE ---
def run_tiled_inference(model, image, conf_thresh=0.10):
    w, h = image.size
    best_box = None; best_conf = 0.0; best_label = "Anomaly"
    
    # Full Pass
    results = model.predict(image, conf=conf_thresh, augment=True, verbose=False)
    if len(results[0].boxes) > 0:
        best_box = results[0].boxes[0]; best_conf = float(best_box.conf[0]); best_label = model.names[int(best_box.cls[0])]

    # Tile Pass
    tiles = [(0,0,w//2,h//2), (w//2,0,w,h//2), (0,h//2,w//2,h), (w//2,h//2,w,h)]
    for tc in tiles:
        tile_img = image.crop(tc)
        # Lower confidence for tiles to catch small toes
        res = model.predict(tile_img, conf=0.05, verbose=False) 
        if len(res[0].boxes) > 0:
            box = res[0].boxes[0]
            if float(box.conf[0]) > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best_box = box
                best_box.global_xyxy = [x1+tc[0], y1+tc[1], x2+tc[0], y2+tc[1]]
                best_conf = float(box.conf[0])
                best_label = model.names[int(box.cls[0])]
    return best_box, best_conf, best_label

# --- HELPER: DRAWING ---
def draw_smart_overlay(image, box, label, conf, risk_status):
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = box.global_xyxy if hasattr(box, 'global_xyxy') else box.xyxy[0].tolist()
    color = (255, 0, 0) if risk_status == "HIGH PRIORITY" else (0, 120, 255)
    draw.rectangle([x1, y1, x2, y2], fill=color+(60,), outline=color+(255,), width=5)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

def get_location_text(box, w, h, scan_type):
    if hasattr(box, 'global_xyxy'):
        cx = (box.global_xyxy[0] + box.global_xyxy[2])/2
        cy = (box.global_xyxy[1] + box.global_xyxy[3])/2
    else:
        cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
    if scan_type == "Chest":
        zone = "Upper" if cy < h/3 else "Middle" if cy < 2*h/3 else "Lower"
        return f"{'Right' if cx < w/2 else 'Left'} {zone} Zone"
    return "Distal Region" if cy > h/2 else "Proximal Region"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Patient Intake")
    if "symptom_state" not in st.session_state: st.session_state.symptom_state = {}
    
    chest_mode = "Tuberculosis (TB)"
    if "detected_type" in st.session_state and st.session_state.detected_type == "Chest":
        chest_mode = st.radio("Protocol:", ["Tuberculosis (TB)", "Pneumonia"])
    
    st.markdown("---")
    user_notes = st.text_area("Symptoms...", height=80)
    if st.button("⚡ Auto-Fill"):
        if "detected_type" in st.session_state:
            detected = pre_pre_symptoms_solver.parse_symptoms_from_text(user_notes, st.session_state.detected_type)
            for k in detected: st.session_state.symptom_state[k] = True

    # Checkboxes (Simplified)
    syms = {}
    if "detected_type" in st.session_state:
        dtype = st.session_state.detected_type
        def chk(l, k): st.session_state.symptom_state[k] = st.checkbox(l, value=st.session_state.symptom_state.get(k,False), key=k); return st.session_state.symptom_state[k]
        if dtype == "Bone":
            syms['deformity'] = chk("Deformity", 'deformity')
            syms['cannot_move'] = chk("Cannot Move", 'cannot_move')
            syms['trauma'] = chk("Trauma", 'trauma')
        elif dtype == "Chest":
            syms['high_fever'] = chk("High Fever", 'high_fever')
            syms['cough_blood'] = chk("Cough Blood", 'cough_blood')

# --- MAIN APP ---
st.title("Dayflow: AI Diagnosis Assistant")
if not router: st.error("⚠️ Models Offline."); st.stop()

uploaded_file = st.file_uploader("Upload X-Ray", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    raw_image = Image.open(uploaded_file).convert('RGB')
    
    # 1. SOURCE SELECTOR (The Fix!)
    st.info("👇 **Where did this image come from?** (Crucial for accuracy)")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        source_mode = st.radio("Select Source:", ["Direct Upload (Clean)", "Photo of Film", "Photo of Screen/Monitor"])

    # 2. CROPPER
    st.write("✂️ **Step 2: Crop (Remove Background)**")
    cropped_preview = st_cropper(raw_image, realtime_update=True, box_color='#0000FF', aspect_ratio=None)
    
    if st.button("🚀 Run Analysis"):
        # 3. APPLY SPECIFIC ENHANCEMENT
        if source_mode == "Photo of Film":
            image_for_ai = enhance_film_photo(cropped_preview)
            st.caption("✅ Applied Film Correction (Glare Removal)")
        elif source_mode == "Photo of Screen/Monitor":
            image_for_ai = enhance_screen_photo(cropped_preview)
            st.caption("✅ Applied Screen Correction (Moiré Removal)")
        else:
            image_for_ai = cropped_preview
            
        w, h = image_for_ai.size
        
        # 4. ROUTER
        img_arr = np.array(image_for_ai.resize((224,224)))/255.0
        pred = router.predict(np.expand_dims(img_arr, axis=0), verbose=0)
        detected_type = ["Bone", "Chest", "Invalid"][np.argmax(pred)]
        st.session_state.detected_type = detected_type
        
        if detected_type == "Invalid": st.error("🚫 Invalid Image"); st.stop()
        
        col1, col2 = st.columns(2)
        with col1: st.image(image_for_ai, caption="AI View (Enhanced)", use_container_width=True)
        
        # 5. INFERENCE
        model = None; cond = "Fracture"
        if detected_type == "Bone": model = bone_model
        else:
            if chest_mode == "Pneumonia": model = pneu_model; cond = "Pneumonia"
            else: model = tb_model; cond = "TB"
            
        with col2:
            with st.spinner(f"Scanning for {cond}..."):
                if detected_type == "Bone":
                    # Run Zoom Engine
                    box, ai_conf, ai_label = run_tiled_inference(model, image_for_ai)
                else:
                    res = model.predict(image_for_ai, conf=0.10, augment=True, verbose=False)
                    box, ai_conf, ai_label = None, 0.0, "Anomaly"
                    if len(res[0].boxes)>0: 
                        box = res[0].boxes[0]; ai_conf = float(box.conf[0]); ai_label = model.names[int(box.cls[0])]
                
                # Report
                ai_loc = get_location_text(box, w, h, detected_type) if ai_conf > 0 else "Unknown"
                boost, findings, narrative = pre_pre_symptoms_solver.generate_diagnostic_reasoning(syms, detected_type, ai_conf, ai_loc)
                final_prob = min(ai_conf+boost, 1.0) if ai_conf > 0 else 0.0
                
                if final_prob > 0.6: 
                    lbl="HIGH PRIORITY"; clr="#cc0000"; bg="#ffe6e6"; act="Splint & Refer"
                else: 
                    lbl="ROUTINE"; clr="#004085"; bg="#e6f0ff"; act="Observe"
                
                final_img = image_for_ai
                if ai_conf > 0: final_img = draw_smart_overlay(image_for_ai, box, ai_label, ai_conf, lbl)
                st.image(final_img, caption="AI Result", use_container_width=True)
                
        # 6. REPORT
        st.markdown(f"""
        <div style="background-color:#f9f9f9; padding:20px; border-left:6px solid {clr};">
            <h3 style="color:black;">Diagnosis Report</h3>
            <p style="color:#333;"><b>Finding:</b> {narrative}</p>
            <div style="background-color:{bg}; padding:10px; text-align:center;">
                <h2 style="color:{clr};">{lbl} ({final_prob*100:.1f}%)</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)