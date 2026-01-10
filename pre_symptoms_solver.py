# # def calculate_clinical_boost(symptoms, scan_type="Bone"):
# #     score = 0.0
# #     reasoning = []
    
# #     if scan_type == "Bone":
# #         # --- BONE FRACTURE LOGIC ---
# #         if symptoms.get('deformity'): score += 0.40; reasoning.append("Visible Deformity (+40%)")
# #         if symptoms.get('immobility'): score += 0.30; reasoning.append("Immobility (+30%)")
# #         if symptoms.get('trauma'): score += 0.15; reasoning.append("Trauma History (+15%)")
# #         if symptoms.get('swelling'): score += 0.10; reasoning.append("Swelling (+10%)")
# #         if symptoms.get('pain'): score += 0.05; reasoning.append("Localized Pain (+5%)")
        
# #     else:
# #         # --- LUNG / TB LOGIC ---
# #         if symptoms.get('blood_sputum'): score += 0.45; reasoning.append("Hemoptysis/Blood in Sputum (+45%)")
# #         if symptoms.get('weight_loss'): score += 0.20; reasoning.append("Unexplained Weight Loss (+20%)")
# #         if symptoms.get('cough'): score += 0.15; reasoning.append("Persistent Cough (+15%)")
# #         if symptoms.get('night_sweats'): score += 0.10; reasoning.append("Night Sweats (+10%)")
# #         if symptoms.get('fever'): score += 0.10; reasoning.append("High Fever (+10%)")

# #     final_boost = min(score, 0.50)
# #     return final_boost, reasoning







# def calculate_clinical_boost(symptoms, scan_type="Bone"):
#     """
#     Calculates risk boost and generates professional medical reasoning text.
#     """
#     score = 0.0
#     reasoning_list = []
    
#     if scan_type == "Bone":
#         # --- ORTHOPEDIC LOGIC ---
#         if symptoms.get('deformity'): 
#             score += 0.40
#             reasoning_list.append("visible gross deformity")
#         if symptoms.get('immobility'): 
#             score += 0.30
#             reasoning_list.append("functional loss (immobility)")
#         if symptoms.get('trauma'): 
#             score += 0.15
#             reasoning_list.append("recent high-impact trauma")
#         if symptoms.get('swelling'): 
#             score += 0.10
#             reasoning_list.append("soft tissue swelling")
#         if symptoms.get('pain'): 
#             score += 0.05
#             reasoning_list.append("localized tenderness")
        
#         # Contextual Text Generation
#         if reasoning_list:
#             reasoning_text = f"Orthopedic risk elevated due to {', '.join(reasoning_list)}."
#         else:
#             reasoning_text = "No significant orthopedic risk factors reported."
            
#     else:
#         # --- PULMONARY / TB LOGIC ---
#         if symptoms.get('blood_sputum'): 
#             score += 0.45
#             reasoning_list.append("hemoptysis (blood in sputum)")
#         if symptoms.get('weight_loss'): 
#             score += 0.20
#             reasoning_list.append("unexplained cachexia (weight loss)")
#         if symptoms.get('cough'): 
#             score += 0.15
#             reasoning_list.append("chronic cough (>2 weeks)")
#         if symptoms.get('night_sweats'): 
#             score += 0.10
#             reasoning_list.append("nocturnal diaphoresis")
#         if symptoms.get('fever'): 
#             score += 0.10
#             reasoning_list.append("pyrexia (fever)")

#         # Contextual Text Generation
#         if reasoning_list:
#             reasoning_text = f"Pulmonary risk elevated due to {', '.join(reasoning_list)}."
#         else:
#             reasoning_text = "No significant respiratory symptoms reported."

#     # Cap boost at 50%
#     final_boost = min(score, 0.50)
    
#     return final_boost, reasoning_list, reasoning_text











def generate_diagnostic_reasoning(symptoms, scan_type, ai_confidence, ai_loc):
    """
    Generates a natural-language 'Reasoning Block' that explains the diagnosis
    like a senior doctor teaching a junior doctor.
    """
    risk_score = 0.0
    clinical_findings = []
    
    # --- 1. CLINICAL ANALYSIS ---
    if scan_type == "Bone":
        if symptoms.get('deformity'): 
            risk_score += 0.40; clinical_findings.append("visible structural deformity")
        if symptoms.get('immobility'): 
            risk_score += 0.30; clinical_findings.append("total functional loss")
        if symptoms.get('trauma'): 
            risk_score += 0.15; clinical_findings.append("mechanism of injury (trauma)")
        if symptoms.get('swelling'): risk_score += 0.10
        if symptoms.get('pain'): risk_score += 0.05
        
        # Narrative Generation
        if ai_confidence > 0:
            if "deformity" in symptoms:
                explanation = f"The detected anomaly in the {ai_loc} aligns with the patient's visible deformity, strongly confirming a displaced fracture."
            elif "trauma" in symptoms:
                explanation = f"Given the history of trauma, the subtle discontinuity in the {ai_loc} is likely an acute fracture."
            else:
                explanation = f"AI has isolated a structural irregularity in the {ai_loc}. While clinical signs are minor, this warrants orthopedic review."
        else:
            explanation = "Imaging appears intact. However, if point tenderness persists, consider an occult (hidden) fracture not visible on X-ray."

    else: # Chest/Lung
        if symptoms.get('blood_sputum'): 
            risk_score += 0.45; clinical_findings.append("hemoptysis (critical)")
        if symptoms.get('weight_loss'): 
            risk_score += 0.20; clinical_findings.append("constitutional weight loss")
        if symptoms.get('cough'): 
            risk_score += 0.15; clinical_findings.append("chronic cough")
        if symptoms.get('night_sweats'): 
            risk_score += 0.10; clinical_findings.append("night sweats")
        if symptoms.get('fever'): risk_score += 0.10

        # Narrative Generation
        if ai_confidence > 0:
            if "Upper" in ai_loc and ("cough" in symptoms or "weight_loss" in symptoms):
                explanation = f"The lesion in the {ai_loc} is highly suspicious for Tuberculosis, especially given the patient's chronic symptoms. Immediate isolation recommended."
            elif "blood_sputum" in symptoms:
                explanation = f"Urgent: The opacity in the {ai_loc} combined with hemoptysis is a critical presentation requiring immediate sputum testing."
            else:
                explanation = f"AI detected a density in the {ai_loc}. Clinical correlation is needed to rule out pneumonia vs. early TB."
        else:
            explanation = "Lung fields appear clear. If symptoms like hemoptysis persist, refer for CT Scan as X-ray sensitivity may be limited."

    # Cap Clinical Boost
    clinical_boost = min(risk_score, 0.50)
    
    return clinical_boost, clinical_findings, explanation









# def generate_diagnostic_reasoning(symptoms, scan_type, ai_confidence, ai_loc):
#     """
#     Returns the risk score AND a dictionary explaining 'Why'.
#     """
#     # 1. Base Score (What the Camera Saw)
#     score_breakdown = {"Visual AI Evidence": round(ai_confidence, 2)}
#     risk_score = 0.0
#     clinical_findings = []
    
#     # --- BONE LOGIC ---
#     if scan_type == "Bone":
#         if symptoms.get('deformity'): 
#             risk_score += 0.40; clinical_findings.append("visible deformity")
#             score_breakdown["Symptom: Deformity"] = 0.40
#         if symptoms.get('bone_poke'): 
#             risk_score += 0.50; clinical_findings.append("open fracture") 
#             score_breakdown["Symptom: Bone Exposed"] = 0.50
#         if symptoms.get('cannot_move'): 
#             risk_score += 0.30; clinical_findings.append("functional loss")
#             score_breakdown["Symptom: Immobility"] = 0.30
            
#     # --- CHEST LOGIC ---
#     else: 
#         if symptoms.get('cough_blood'): 
#             risk_score += 0.45; clinical_findings.append("coughing blood")
#             score_breakdown["Symptom: Hemoptysis"] = 0.45
#         if symptoms.get('high_fever'): 
#             risk_score += 0.35; clinical_findings.append("high fever")
#             score_breakdown["Symptom: Fever"] = 0.35
            
#     # Cap the clinical boost at 0.50
#     clinical_boost = min(risk_score, 0.50)
    
#     # Narrative Logic (Same as before)
#     if ai_confidence > 0:
#         narrative = f"Visual anomaly at {ai_loc} correlates with {len(clinical_findings)} clinical signs."
#     else:
#         narrative = "No visual anomaly detected, but clinical signs suggest monitoring."

#     return clinical_boost, clinical_findings, narrative, score_breakdown