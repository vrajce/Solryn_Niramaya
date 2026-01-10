// API Service for Dayflow Multi-Specialist CDSS Backend
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Check API health status
 */
export const checkHealth = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error('API not available');
        return await response.json();
    } catch (error) {
        console.error('Health check failed:', error);
        throw error;
    }
};

/**
 * Universal analysis - auto-routes to appropriate specialist
 * @param {File} file - The image file to analyze
 * @param {Object} symptoms - All symptom flags (bone + lung)
 * @param {boolean} includeImage - Whether to include annotated image in response
 * @returns {Promise<Object>} Analysis results
 */
export const analyzeXray = async (file, symptoms = {}, includeImage = true) => {
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        // Append all symptoms
        formData.append('pain', symptoms.pain || false);
        formData.append('fever', symptoms.fever || false);
        // Bone symptoms
        formData.append('deformity', symptoms.deformity || false);
        formData.append('immobility', symptoms.immobility || false);
        formData.append('trauma', symptoms.trauma || false);
        formData.append('swelling', symptoms.swelling || false);
        // Lung symptoms
        formData.append('cough', symptoms.cough || false);
        formData.append('blood_sputum', symptoms.blood_sputum || false);
        formData.append('night_sweats', symptoms.night_sweats || false);
        formData.append('weight_loss', symptoms.weight_loss || false);
        // Options
        formData.append('include_image', includeImage);

        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Analysis failed with status ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Analysis failed:', error);
        throw error;
    }
};

/**
 * Direct bone fracture analysis (bypasses router)
 */
export const analyzeBone = async (file, symptoms = {}, includeImage = true) => {
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('deformity', symptoms.deformity || false);
        formData.append('immobility', symptoms.immobility || false);
        formData.append('trauma', symptoms.trauma || false);
        formData.append('swelling', symptoms.swelling || false);
        formData.append('pain', symptoms.pain || false);
        formData.append('include_image', includeImage);

        const response = await fetch(`${API_BASE_URL}/analyze/bone`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Bone analysis failed`);
        }

        return await response.json();
    } catch (error) {
        console.error('Bone analysis failed:', error);
        throw error;
    }
};

/**
 * Direct lung/TB/Pneumonia analysis (bypasses router)
 */
export const analyzeLung = async (file, symptoms = {}, includeImage = true) => {
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('cough', symptoms.cough || false);
        formData.append('blood_sputum', symptoms.blood_sputum || false);
        formData.append('night_sweats', symptoms.night_sweats || false);
        formData.append('weight_loss', symptoms.weight_loss || false);
        formData.append('fever', symptoms.fever || false);
        formData.append('include_image', includeImage);

        const response = await fetch(`${API_BASE_URL}/analyze/lung`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Lung analysis failed`);
        }

        return await response.json();
    } catch (error) {
        console.error('Lung analysis failed:', error);
        throw error;
    }
};

/**
 * Route image to determine scan type without full analysis
 */
export const routeImage = async (file) => {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/route`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Routing failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Routing failed:', error);
        throw error;
    }
};

/**
 * Calculate clinical boost from symptoms only
 * @param {Object} symptoms - Symptom flags
 * @param {string} scanType - "Bone" or "Chest"
 * @returns {Promise<Object>} Clinical boost calculation
 */
export const calculateSymptoms = async (symptoms, scanType = "Bone") => {
    try {
        const response = await fetch(`${API_BASE_URL}/symptoms/calculate?scan_type=${scanType}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(symptoms),
        });

        if (!response.ok) {
            throw new Error('Symptom calculation failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Symptom calculation failed:', error);
        throw error;
    }
};

/**
 * Convert base64 image to blob URL for display
 * @param {string} base64 - Base64 encoded image
 * @returns {string} Blob URL
 */
export const base64ToImageUrl = (base64) => {
    if (!base64) return null;
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/png' });
    return URL.createObjectURL(blob);
};

export default {
    checkHealth,
    analyzeXray,
    analyzeBone,
    analyzeLung,
    routeImage,
    calculateSymptoms,
    base64ToImageUrl,
    API_BASE_URL,
};
