# import re

# def parse_symptoms_from_text(text, scan_type):
#     """
#     The 'AI Scribe' Engine.
#     Scans free text for keywords and returns a list of checkboxes to AUTO-SELECT.
#     """
#     text = text.lower()
#     detected_keys = []

#     # --- 1. BONE KEYWORDS ---
#     if scan_type == "Bone":
#         if any(x in text for x in ['bent', 'crooked', 'shape', 'weird angle', 'deform']):
#             detected_keys.append('deformity')
#         if any(x in text for x in ['bone out', 'skin', 'poking', 'open', 'bleeding']):
#             detected_keys.append('bone_poke')
#         if any(x in text for x in ['cant move', 'stuck', 'paralyzed', 'frozen', 'lock']):
#             detected_keys.append('cannot_move')
#         if any(x in text for x in ['fell', 'hit', 'accident', 'crash', 'bike', 'car', 'trauma']):
#             detected_keys.append('trauma')
#         if any(x in text for x in ['snap', 'crack', 'pop', 'sound']):
#             detected_keys.append('snap_sound')
#         if any(x in text for x in ['swell', 'bruise', 'edema', 'blue']):
#             detected_keys.append('swelling')

#     # --- 2. CHEST KEYWORDS ---
#     else:
#         # TB Signs
#         if any(x in text for x in ['blood', 'red spit', 'hemoptysis']):
#             detected_keys.append('cough_blood')
#         if any(x in text for x in ['weight', 'skinny', 'thin', 'clothes loose']):
#             detected_keys.append('weight_loss')
#         if any(x in text for x in ['sweat', 'night', 'wet bed']):
#             detected_keys.append('night_sweats')
#         if any(x in text for x in ['weeks', 'month', 'long time', 'chronic']):
#             detected_keys.append('long_cough')

#         # Pneumonia Signs
#         if any(x in text for x in ['hot', 'burning', 'fever', 'high temp', '39', '40']):
#             detected_keys.append('high_fever')
#         if any(x in text for x in ['shake', 'shiver', 'cold', 'chills']):
#             detected_keys.append('shaking_chills')
#         if any(x in text for x in ['breath', 'air', 'gasp', 'pant']):
#             detected_keys.append('hard_breathe')
#         if any(x in text for x in ['green', 'yellow', 'slime', 'mucus', 'phlegm']):
#             detected_keys.append('green_phlegm')

#     return detected_keys

# def generate_diagnostic_reasoning(symptoms, scan_type, ai_confidence, ai_loc):
#     """
#     Generates reasoning using SIMPLE terms for non-medical users.
#     """
#     risk_score = 0.0
#     clinical_findings = []
    
#     # --- 1. BONE FRACTURE LOGIC 🦴 ---
#     if scan_type == "Bone":
#         # Critical Signs
#         if symptoms.get('deformity'): 
#             risk_score += 0.40; clinical_findings.append("bone looks bent/wrong shape")
#         if symptoms.get('bone_poke'): 
#             risk_score += 0.50; clinical_findings.append("bone poking through skin") # Critical!
#         if symptoms.get('cannot_move'): 
#             risk_score += 0.30; clinical_findings.append("cannot move the limb")
        
#         # Secondary Signs
#         if symptoms.get('snap_sound'): risk_score += 0.15; clinical_findings.append("heard a snap/crack sound")
#         if symptoms.get('trauma'): risk_score += 0.15; clinical_findings.append("bad fall or hit")
#         if symptoms.get('swelling'): risk_score += 0.10; clinical_findings.append("swelling or bruising")

#         # Narrative Generation
#         if ai_confidence > 0:
#             narrative = f"The break seen in the {ai_loc} matches the patient's report of {', '.join(clinical_findings)}, confirming a fracture."
#         else:
#             narrative = "No clear break seen on X-ray. However, if the patient cannot move the limb or is in severe pain, treat as a hidden fracture."

#     # --- 2. CHEST LOGIC (TB vs PNEUMONIA) 🫁 ---
#     else: 
#         # TB: Long-term/Wasting Signs
#         if symptoms.get('cough_blood'): 
#             risk_score += 0.45; clinical_findings.append("coughing up blood")
#         if symptoms.get('weight_loss'): 
#             risk_score += 0.25; clinical_findings.append("losing weight without trying")
#         if symptoms.get('long_cough'): 
#             risk_score += 0.20; clinical_findings.append("cough lasting >3 weeks")
#         if symptoms.get('night_sweats'): 
#             risk_score += 0.15; clinical_findings.append("heavy sweating at night")
        
#         # Pneumonia: Fast/Hot Signs
#         if symptoms.get('high_fever'): 
#             risk_score += 0.35; clinical_findings.append("very high fever (>39°C)")
#         if symptoms.get('hard_breathe'): 
#             risk_score += 0.30; clinical_findings.append("struggling to breathe")
#         if symptoms.get('shaking_chills'): 
#             risk_score += 0.20; clinical_findings.append("shaking chills")
#         if symptoms.get('green_phlegm'): 
#             risk_score += 0.15; clinical_findings.append("coughing green/yellow slime")

#         # Narrative Generation
#         if ai_confidence > 0:
#             if "coughing up blood" in clinical_findings:
#                 narrative = f"DANGER: The lung spot in {ai_loc} + coughing blood is a classic sign of Active TB."
#             elif "very high fever" in clinical_findings:
#                 narrative = f"The cloudy area in {ai_loc} + high fever looks like a Bacterial Pneumonia infection."
#             else:
#                 narrative = f"AI found a problem in {ai_loc}. Based on symptoms ({', '.join(clinical_findings)}), please check patient history."
#         else:
#             narrative = "Lungs look clear. If breathing is still hard, check for asthma or heart issues."

#     # Cap Clinical Boost (Max 50% extra confidence)
#     clinical_boost = min(risk_score, 0.50)
    
#     return clinical_boost, clinical_findings, narrative









import re

def parse_symptoms_from_text(text, scan_type):
    """
    AI Scribe: Extracts medical keywords from free text.
    """
    text = text.lower()
    detected_keys = []

    # --- 1. BONE KEYWORDS ---
    if scan_type == "Bone":
        if any(x in text for x in ['bent', 'crooked', 'shape', 'weird angle', 'deform']):
            detected_keys.append('deformity')
        if any(x in text for x in ['bone out', 'skin', 'poking', 'open', 'bleeding']):
            detected_keys.append('bone_poke')
        if any(x in text for x in ['cant move', 'stuck', 'paralyzed', 'frozen', 'lock']):
            detected_keys.append('cannot_move')
        if any(x in text for x in ['fell', 'hit', 'accident', 'crash', 'bike', 'car', 'trauma']):
            detected_keys.append('trauma')
        if any(x in text for x in ['snap', 'crack', 'pop', 'sound']):
            detected_keys.append('snap_sound')
        if any(x in text for x in ['swell', 'bruise', 'edema', 'blue']):
            detected_keys.append('swelling')

    # --- 2. CHEST KEYWORDS ---
    else:
        # TB Signs
        if any(x in text for x in ['blood', 'red spit', 'hemoptysis']):
            detected_keys.append('cough_blood')
        if any(x in text for x in ['weight', 'skinny', 'thin', 'clothes loose']):
            detected_keys.append('weight_loss')
        if any(x in text for x in ['sweat', 'night', 'wet bed']):
            detected_keys.append('night_sweats')
        if any(x in text for x in ['weeks', 'month', 'long time', 'chronic']):
            detected_keys.append('long_cough')

        # Pneumonia Signs
        if any(x in text for x in ['hot', 'burning', 'fever', 'high temp', '39', '40']):
            detected_keys.append('high_fever')
        if any(x in text for x in ['shake', 'shiver', 'cold', 'chills']):
            detected_keys.append('shaking_chills')
        if any(x in text for x in ['breath', 'air', 'gasp', 'pant']):
            detected_keys.append('hard_breathe')
        if any(x in text for x in ['green', 'yellow', 'slime', 'mucus', 'phlegm']):
            detected_keys.append('green_phlegm')

    return detected_keys

def generate_diagnostic_reasoning(symptoms, scan_type, ai_confidence, ai_loc):
    """
    Generates reasoning using SIMPLE terms for non-medical users.
    """
    risk_score = 0.0
    clinical_findings = []
    
    # --- 1. BONE FRACTURE LOGIC 🦴 ---
    if scan_type == "Bone":
        if symptoms.get('deformity'): 
            risk_score += 0.40; clinical_findings.append("visible deformity")
        if symptoms.get('bone_poke'): 
            risk_score += 0.50; clinical_findings.append("open fracture (bone visible)") 
        if symptoms.get('cannot_move'): 
            risk_score += 0.30; clinical_findings.append("functional loss")
        if symptoms.get('snap_sound'): risk_score += 0.15; clinical_findings.append("audible snap")
        if symptoms.get('trauma'): risk_score += 0.15; clinical_findings.append("trauma mechanism")

        # Narrative Generation
        if ai_confidence > 0:
            narrative = f"The structural break in the {ai_loc} aligns with {', '.join(clinical_findings)}, confirming a fracture."
        else:
            narrative = "No obvious break seen. However, if patient cannot move the limb, a 'hairline' fracture might still exist (occult fracture)."

    # --- 2. CHEST LOGIC 🫁 ---
    else: 
        if symptoms.get('cough_blood'): risk_score += 0.45; clinical_findings.append("coughing blood")
        if symptoms.get('weight_loss'): risk_score += 0.25; clinical_findings.append("weight loss")
        if symptoms.get('high_fever'): risk_score += 0.35; clinical_findings.append("high fever")
        if symptoms.get('hard_breathe'): risk_score += 0.30; clinical_findings.append("dyspnea")

        if ai_confidence > 0:
            if "coughing blood" in clinical_findings:
                narrative = f"DANGER: Opacity in {ai_loc} + blood suggests Active TB."
            elif "high fever" in clinical_findings:
                narrative = f"Consolidation in {ai_loc} + fever fits Bacterial Pneumonia."
            else:
                narrative = f"AI found an anomaly in {ai_loc}. Correlate with symptoms: {', '.join(clinical_findings)}."
        else:
            narrative = "Lungs appear clear. Monitor breathing."

    clinical_boost = min(risk_score, 0.50)
    return clinical_boost, clinical_findings, narrative