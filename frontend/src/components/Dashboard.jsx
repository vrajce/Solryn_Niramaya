import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import UploadArea from './UploadArea';
import ResultsView from './ResultsView';
import { analyzeXray, base64ToImageUrl } from '../services/api';

const Dashboard = () => {
    const location = useLocation();
    const [file, setFile] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [error, setError] = useState(null);

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
            
            // Transform API response to match ResultsView expected format
            const transformedResult = {
                success: result.success,
                scanType: result.scan_type,
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

    const handleReset = () => {
        setFile(null);
        setShowResults(false);
        setSelectedHistoryItem(null);
        setAnalysisResult(null);
        setError(null);
    };

    return (
        <>
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
