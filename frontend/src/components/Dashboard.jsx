import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import UploadArea from './UploadArea';
import ResultsView from './ResultsView';
import { analyzeXray, analyzeTB, analyzePneumonia, base64ToImageUrl } from '../services/api';

const Dashboard = () => {
    const location = useLocation();
    const [file, setFile] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [error, setError] = useState(null);
    
    // New state for chest condition selection
    const [showChestModal, setShowChestModal] = useState(false);
    const [pendingChestImage, setPendingChestImage] = useState(null);
    const [chestImagePreview, setChestImagePreview] = useState(null);

    useEffect(() => {
        if (location.state?.historyItem) {
            setSelectedHistoryItem(location.state.historyItem);
            setShowResults(true);
            setFile(null);
            setAnalysisResult(null);
        } else {
            setShowResults(false);
            setSelectedHistoryItem(null);
        }
    }, [location.state]);

    const handleUpload = async (uploadedFile) => {
        setFile(uploadedFile);
        setIsAnalyzing(true);
        setSelectedHistoryItem(null);
        setError(null);
        setAnalysisResult(null);

        try {
            // Call the FastAPI backend without symptoms initially
            // Symptoms will be added on the results page
            const result = await analyzeXray(uploadedFile, {}, true);
            
            // Handle invalid image
            if (!result.success || result.scan_type === "Invalid") {
                setError("Invalid image detected. Please upload a valid medical X-ray scan.");
                setIsAnalyzing(false);
                return;
            }
            
            // Check if this is a chest X-ray requiring selection
            if (result.scan_type === "Chest" && result.chest_condition === "SELECTION_REQUIRED") {
                // Show modal for TB/Pneumonia selection
                setPendingChestImage(uploadedFile);
                setChestImagePreview(result.annotated_image_base64 
                    ? base64ToImageUrl(result.annotated_image_base64) 
                    : URL.createObjectURL(uploadedFile));
                setShowChestModal(true);
                setIsAnalyzing(false);
                return;
            }
            
            // Transform API response to match ResultsView expected format
            const transformedResult = {
                success: result.success,
                scanType: result.scan_type,
                chestCondition: result.chest_condition,
                scanTypeConfidence: result.scan_type_confidence,
                disease: result.risk_class,
                detectedCondition: result.detected_condition,
                confidence: result.final_confidence,
                aiConfidence: result.ai_confidence,
                clinicalBoost: result.clinical_boost,
                severity: result.risk_class,
                location: result.ai_location,
                method: result.ai_method,
                statusMessage: result.status_message,
                visualAnalysis: result.visual_analysis_text,
                clinicalContext: result.clinical_context_text,
                clinicalReasons: result.clinical_reasons,
                annotatedImageUrl: result.annotated_image_base64 
                    ? base64ToImageUrl(result.annotated_image_base64) 
                    : null,
                findings: result.ai_confidence > 0 ? [{
                    id: 1,
                    label: result.detected_condition || (result.scan_type === "Bone" ? "Fracture Pattern" : "Lung Abnormality"),
                    region: result.ai_location,
                    confidence: result.ai_confidence
                }] : []
            };
            
            setAnalysisResult(transformedResult);
            setShowResults(true);
        } catch (err) {
            console.error('Analysis error:', err);
            setError(err.message || 'Failed to analyze image. Please ensure the API server is running.');
        } finally {
            setIsAnalyzing(false);
        }
    };

    // Handle chest condition selection (TB or Pneumonia)
    const handleChestConditionSelect = async (condition) => {
        setShowChestModal(false);
        setIsAnalyzing(true);
        setError(null);
        
        try {
            let result;
            if (condition === 'TB') {
                result = await analyzeTB(pendingChestImage, {}, true);
            } else {
                result = await analyzePneumonia(pendingChestImage, {}, true);
            }
            
            // Transform API response
            const transformedResult = {
                success: result.success,
                scanType: result.scan_type,
                chestCondition: result.chest_condition,
                scanTypeConfidence: result.scan_type_confidence,
                disease: result.risk_class,
                detectedCondition: result.detected_condition,
                confidence: result.final_confidence,
                aiConfidence: result.ai_confidence,
                clinicalBoost: result.clinical_boost,
                severity: result.risk_class,
                location: result.ai_location,
                method: result.ai_method,
                statusMessage: result.status_message,
                visualAnalysis: result.visual_analysis_text,
                clinicalContext: result.clinical_context_text,
                clinicalReasons: result.clinical_reasons,
                annotatedImageUrl: result.annotated_image_base64 
                    ? base64ToImageUrl(result.annotated_image_base64) 
                    : null,
                findings: result.ai_confidence > 0 ? [{
                    id: 1,
                    label: result.detected_condition || condition,
                    region: result.ai_location,
                    confidence: result.ai_confidence
                }] : []
            };
            
            setFile(pendingChestImage);
            setAnalysisResult(transformedResult);
            setShowResults(true);
            setPendingChestImage(null);
            setChestImagePreview(null);
        } catch (err) {
            console.error('Chest analysis error:', err);
            setError(err.message || `Failed to analyze for ${condition}. Please try again.`);
            setPendingChestImage(null);
            setChestImagePreview(null);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleCancelChestSelection = () => {
        setShowChestModal(false);
        setPendingChestImage(null);
        setChestImagePreview(null);
        setFile(null);
    };

    const handleReset = () => {
        setFile(null);
        setShowResults(false);
        setSelectedHistoryItem(null);
        setAnalysisResult(null);
        setError(null);
        setShowChestModal(false);
        setPendingChestImage(null);
        setChestImagePreview(null);
    };

    return (
        <>
            {/* Chest Condition Selection Modal */}
            {showChestModal && (
                <div className="modal-overlay" onClick={handleCancelChestSelection}>
                    <div className="chest-selection-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2>🫁 Chest X-Ray Detected</h2>
                            <p>Please select the condition to analyze:</p>
                        </div>
                        
                        {chestImagePreview && (
                            <div className="modal-image-preview">
                                <img src={chestImagePreview} alt="Chest X-Ray" />
                            </div>
                        )}
                        
                        <div className="condition-options">
                            <button 
                                className="condition-btn tb-btn"
                                onClick={() => handleChestConditionSelect('TB')}
                            >
                                <div className="condition-icon">🦠</div>
                                <div className="condition-info">
                                    <h3>Tuberculosis (TB)</h3>
                                    <p>Analyze for TB indicators using YOLO detection model</p>
                                    <ul className="condition-symptoms">
                                        <li>Chronic cough (&gt;2 weeks)</li>
                                        <li>Blood in sputum</li>
                                        <li>Night sweats</li>
                                        <li>Weight loss</li>
                                    </ul>
                                </div>
                            </button>
                            
                            <button 
                                className="condition-btn pneumonia-btn"
                                onClick={() => handleChestConditionSelect('Pneumonia')}
                            >
                                <div className="condition-icon">🔬</div>
                                <div className="condition-info">
                                    <h3>Pneumonia</h3>
                                    <p>Analyze using AI classification + U-Net contour mapping</p>
                                    <ul className="condition-symptoms">
                                        <li>High fever</li>
                                        <li>Productive cough</li>
                                        <li>Shortness of breath</li>
                                        <li>Chest pain</li>
                                    </ul>
                                </div>
                            </button>
                        </div>
                        
                        <button className="modal-cancel-btn" onClick={handleCancelChestSelection}>
                            Cancel and Upload Different Image
                        </button>
                    </div>
                </div>
            )}
            
            {!showResults ? (
                <div className="dashboard-initial">
                    <div className="welcome-banner">
                        <h2>AI-Powered Medical X-Ray Analysis</h2>
                        <p>Upload an X-ray scan for automatic detection of Bone Fractures, TB, or Pneumonia.</p>
                    </div>
                    
                    <UploadArea onUpload={handleUpload} isAnalyzing={isAnalyzing} />
                    
                    {isAnalyzing && (
                        <div className="analysis-status">
                            <p>Analyzing scan with AI...</p>
                            <div className="progress-bar">
                                <div className="progress-fill"></div>
                            </div>
                        </div>
                    )}
                    
                    {error && (
                        <div className="error-message">
                            <p>⚠️ {error}</p>
                            <button onClick={() => setError(null)}>Dismiss</button>
                        </div>
                    )}
                </div>
            ) : (
                <div className="dashboard-results">
                    <ResultsView
                        file={file}
                        data={selectedHistoryItem || analysisResult}
                        onReset={handleReset}
                    />
                </div>
            )}
        </>
    );
};

export default Dashboard;
