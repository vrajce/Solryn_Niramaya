def calculate_clinical_boost(symptoms):
    """
    Calculates a confidence boost score (0.0 to 0.5) based on clinical symptoms.
    Returns: 
        - boost_score (float): The amount to add to AI confidence.
        - reasoning (list): A list of strings explaining the boost.
    """
    score = 0.0
    reasoning = []
    
    # --- CRITICAL SYMPTOMS (High Boost) ---
    if symptoms.get('deformity'): 
        score += 0.40
        reasoning.append("Visible Bone Deformity (+40%)")
        
    if symptoms.get('immobility'): 
        score += 0.30
        reasoning.append("Functional Loss/Immobility (+30%)")
        
    # --- SUPPORTING SYMPTOMS (Moderate Boost) ---
    if symptoms.get('trauma'): 
        score += 0.15
        reasoning.append("History of Trauma (+15%)")
        
    if symptoms.get('swelling'): 
        score += 0.10
        reasoning.append("Soft Tissue Swelling (+10%)")
        
    if symptoms.get('pain'): 
        score += 0.05
        reasoning.append("Localized Pain (+5%)")
    
    # Cap the boost at 50% (0.50) so symptoms alone don't create a 100% false positive
    final_boost = min(score, 0.50)
    
    return final_boost, reasoning