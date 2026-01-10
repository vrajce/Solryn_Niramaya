import React, { useState, useRef, useMemo } from 'react';
import { ArrowLeft, AlertCircle, FileText, Share2, Activity, CheckCircle, AlertTriangle, Info, Download, Copy, Check } from 'lucide-react';
import './ResultsView.css';

const ResultsView = ({ file, data, onReset }) => {
    const [showShareModal, setShowShareModal] = useState(false);
    const [copied, setCopied] = useState(false);
    const reportRef = useRef(null);
    
    // Symptom state for clinical boost calculation
    const [symptoms, setSymptoms] = useState({
        // Bone-specific
        deformity: false,
        immobility: false,
        trauma: false,
        swelling: false,
        pain: false,
        // Lung-specific
        cough: false,
        blood_sputum: false,
        night_sweats: false,
        weight_loss: false,
        fever: false
    });
    
    // Determine if this is API data or mock/history data
    const isApiData = data?.visualAnalysis !== undefined;

    // Use passed data or fallback to mock
    const baseAnalysisData = data || {
        disease: "No Data",
        confidence: 0,
        severity: "Unknown",
        findings: []
    };
    
    // Calculate clinical boost based on symptoms (client-side)
    const calculateClinicalBoost = useMemo(() => {
        const scanType = baseAnalysisData.scanType || 'Bone';
        let score = 0;
        const reasons = [];
        
        if (scanType === 'Bone') {
            if (symptoms.deformity) { score += 0.40; reasons.push("Visible Bone Deformity (+40%)"); }
            if (symptoms.immobility) { score += 0.30; reasons.push("Functional Loss/Immobility (+30%)"); }
            if (symptoms.trauma) { score += 0.15; reasons.push("Recent Trauma History (+15%)"); }
            if (symptoms.swelling) { score += 0.10; reasons.push("Soft Tissue Swelling (+10%)"); }
            if (symptoms.pain) { score += 0.05; reasons.push("Localized Pain (+5%)"); }
        } else {
            // Chest/Lung
            if (symptoms.blood_sputum) { score += 0.45; reasons.push("Hemoptysis/Blood in Sputum (+45%)"); }
            if (symptoms.weight_loss) { score += 0.20; reasons.push("Unexplained Weight Loss (+20%)"); }
            if (symptoms.cough) { score += 0.15; reasons.push("Chronic Cough >2 weeks (+15%)"); }
            if (symptoms.night_sweats) { score += 0.10; reasons.push("Night Sweats (+10%)"); }
            if (symptoms.fever) { score += 0.10; reasons.push("High Fever (+10%)"); }
        }
        
        const context = reasons.length > 0 
            ? `${scanType === 'Bone' ? 'Orthopedic' : 'Pulmonary'} risk elevated due to: ${reasons.map(r => r.split(" (+")[0]).join(", ")}.`
            : `No significant ${scanType === 'Bone' ? 'orthopedic' : 'respiratory'} symptoms reported.`;
        
        return { boost: Math.min(score, 0.50), reasons, context };
    }, [symptoms, baseAnalysisData.scanType]);
    
    // Calculate updated analysis data with symptom boost
    const analysisData = useMemo(() => {
        const aiConf = baseAnalysisData.aiConfidence || 0;
        const { boost, reasons, context } = calculateClinicalBoost;
        
        // Calculate final confidence using weighted formula:
        // Final = (AI_Confidence * 0.6) + (Clinical_Boost * 0.4)
        // Normalize boost from 0-0.5 range to 0-1 range for the 40% weight
        const normalizedBoost = Math.min(boost * 2, 1.0);
        let finalConf = (aiConf * 0.6) + (normalizedBoost * 0.4);
        
        // Determine risk class and status
        let riskClass, statusMsg;
        const scanType = baseAnalysisData.scanType || 'Bone';
        
        if (finalConf > 0.60) {
            riskClass = "HIGH RISK";
            statusMsg = scanType === 'Bone' 
                ? "Fracture Confirmed (AI + Clinical)" 
                : "Pulmonary Abnormality Confirmed (AI + Clinical)";
        } else if (finalConf > 0.30) {
            riskClass = "MODERATE RISK";
            statusMsg = scanType === 'Bone'
                ? "Suspected Fracture (Review Required)"
                : "Suspected Lung Pathology (Review Required)";
        } else if (boost > 0.30) {
            riskClass = "CLINICAL WARNING";
            statusMsg = scanType === 'Bone'
                ? "Scan Negative, but Symptoms Critical (Occult Injury?)"
                : "Scan Negative, but Symptoms Critical (Consider CT)";
        } else {
            riskClass = "LOW RISK";
            statusMsg = "No Significant Findings";
        }
        
        return {
            ...baseAnalysisData,
            clinicalBoost: boost,
            clinicalReasons: reasons,
            clinicalContext: context,
            confidence: finalConf,
            disease: riskClass,
            severity: riskClass,
            statusMessage: statusMsg
        };
    }, [baseAnalysisData, calculateClinicalBoost]);

    const handleSymptomChange = (symptom) => {
        setSymptoms(prev => ({ ...prev, [symptom]: !prev[symptom] }));
    };

    // Determine image URL - prioritize annotated image from API
    const imageUrl = data?.annotatedImageUrl 
        || data?.imageUrl 
        || (file ? URL.createObjectURL(file) : "https://via.placeholder.com/500x500?text=Medical+Scan");
    
    const scanName = data?.patientId 
        ? `Scan ${data.date} (${data.patientId})` 
        : file?.name || "X-Ray Analysis";

    // Get risk class styling
    const getRiskClass = (riskClass) => {
        if (!riskClass) return 'unknown';
        const risk = riskClass.toLowerCase();
        if (risk.includes('high')) return 'high-risk';
        if (risk.includes('moderate')) return 'moderate-risk';
        if (risk.includes('warning')) return 'warning-risk';
        if (risk.includes('low')) return 'low-risk';
        return 'unknown';
    };

    // Get risk icon
    const getRiskIcon = (riskClass) => {
        if (!riskClass) return <Info size={32} />;
        const risk = riskClass.toLowerCase();
        if (risk.includes('high')) return <AlertCircle size={32} />;
        if (risk.includes('moderate')) return <AlertTriangle size={32} />;
        if (risk.includes('warning')) return <AlertTriangle size={32} />;
        if (risk.includes('low')) return <CheckCircle size={32} />;
        return <Info size={32} />;
    };

    // Generate Report Function
    const generateReport = () => {
        const reportDate = new Date().toLocaleString();
        const scanType = analysisData.scanType || 'Medical';
        const scanTypeEmoji = scanType === 'Chest' ? '🫁' : scanType === 'Bone' ? '🦴' : '🏥';
        const analysisType = scanType === 'Chest' ? 'Lung/Respiratory Analysis' : scanType === 'Bone' ? 'Bone Fracture Analysis' : 'Medical Analysis';
        
        const reportContent = `
<!DOCTYPE html>
<html>
<head>
    <title>Medical Scan Report - ${scanName}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #1a1a2e; margin-bottom: 5px; }
        .header p { color: #666; margin: 5px 0; }
        .section { margin-bottom: 25px; }
        .section h2 { color: #1a1a2e; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
        .risk-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; }
        .scan-type-badge { display: inline-block; padding: 6px 12px; border-radius: 15px; font-weight: 600; margin-left: 10px; }
        .scan-type-badge.bone { background: #dbeafe; color: #1d4ed8; }
        .scan-type-badge.chest { background: #ede9fe; color: #7c3aed; }
        .high-risk { background: #ffebee; color: #c62828; }
        .moderate-risk { background: #fff3e0; color: #ef6c00; }
        .warning-risk { background: #fff8e1; color: #f9a825; }
        .low-risk { background: #e8f5e9; color: #2e7d32; }
        .metrics { display: flex; gap: 20px; margin: 20px 0; }
        .metric { flex: 1; background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #1a1a2e; }
        .metric-label { font-size: 12px; color: #666; text-transform: uppercase; }
        .findings { background: #f9f9f9; padding: 20px; border-radius: 8px; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666; font-size: 12px; }
        .image-container { text-align: center; margin: 20px 0; }
        .image-container img { max-width: 100%; max-height: 400px; border-radius: 8px; }
        ul { padding-left: 20px; }
        li { margin: 8px 0; }
        @media print { body { padding: 20px; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>${scanTypeEmoji} Dayflow CDSS Report</h1>
        <p>Clinical Decision Support System - ${analysisType}</p>
        <p>Generated: ${reportDate}</p>
    </div>
    
    <div class="section">
        <h2>Patient Information</h2>
        <p><strong>Scan Name:</strong> ${scanName}</p>
        <p><strong>Scan Type:</strong> ${scanType} ${analysisData.scanTypeConfidence ? `(${(analysisData.scanTypeConfidence * 100).toFixed(0)}% confidence)` : ''}</p>
        <p><strong>Analysis Date:</strong> ${data?.date || new Date().toLocaleDateString()}</p>
        ${analysisData.detectedCondition ? `<p><strong>Detected Condition:</strong> ${analysisData.detectedCondition}</p>` : ''}
    </div>
    
    <div class="section">
        <h2>Diagnosis Result</h2>
        <span class="risk-badge ${getRiskClass(analysisData.disease || analysisData.severity)}">
            ${analysisData.disease || analysisData.severity}
        </span>
        <p style="margin-top: 15px;"><strong>Status:</strong> ${analysisData.statusMessage || 'Analysis Complete'}</p>
    </div>
    
    <div class="section">
        <h2>Analysis Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">${((analysisData.aiConfidence || analysisData.confidence || 0) * 100).toFixed(1)}%</div>
                <div class="metric-label">AI Confidence</div>
            </div>
            <div class="metric">
                <div class="metric-value">+${((analysisData.clinicalBoost || 0) * 100).toFixed(0)}%</div>
                <div class="metric-label">Clinical Boost</div>
            </div>
            <div class="metric">
                <div class="metric-value">${((analysisData.confidence || 0) * 100).toFixed(1)}%</div>
                <div class="metric-label">Final Risk</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>AI Reasoning</h2>
        <div class="findings">
            <p><strong>Visual Analysis:</strong> ${analysisData.visualAnalysis || 'N/A'}</p>
            <p><strong>Clinical Context:</strong> ${analysisData.clinicalContext || 'N/A'}</p>
            ${analysisData.location ? `<p><strong>Location:</strong> ${analysisData.location}</p>` : ''}
            ${analysisData.method ? `<p><strong>Detection Method:</strong> ${analysisData.method}</p>` : ''}
            ${analysisData.clinicalReasons?.length > 0 ? `
                <p><strong>Risk Factors:</strong></p>
                <ul>
                    ${analysisData.clinicalReasons.map(r => `<li>${r}</li>`).join('')}
                </ul>
            ` : ''}
        </div>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            ${analysisData.confidence > 0.6 ? `
                <li>Immediate clinical evaluation recommended</li>
                <li>Consider orthopedic consultation</li>
                <li>Immobilization may be required</li>
            ` : analysisData.confidence > 0.3 ? `
                <li>Clinical correlation suggested</li>
                <li>Follow-up imaging may be warranted</li>
                <li>Monitor for symptom progression</li>
            ` : `
                <li>No immediate intervention required</li>
                <li>Continue routine monitoring if symptomatic</li>
            `}
        </ul>
    </div>
    
    <div class="footer">
        <p>This report was generated by Dayflow Clinical Decision Support System</p>
        <p>⚠️ This AI-assisted analysis is for clinical decision support only and should not replace professional medical judgment.</p>
    </div>
</body>
</html>`;
        
        // Create and download the report
        const blob = new Blob([reportContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `Medical_Report_${new Date().toISOString().split('T')[0]}.html`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    // Share Function
    const handleShare = async () => {
        const shareData = {
            title: 'Medical Scan Report',
            text: `Dayflow CDSS Analysis Report\n\nDiagnosis: ${analysisData.disease || analysisData.severity}\nFinal Risk: ${(analysisData.confidence * 100).toFixed(1)}%\nStatus: ${analysisData.statusMessage || 'Analysis Complete'}`,
        };

        // Check if Web Share API is available
        if (navigator.share) {
            try {
                await navigator.share(shareData);
            } catch (err) {
                if (err.name !== 'AbortError') {
                    setShowShareModal(true);
                }
            }
        } else {
            setShowShareModal(true);
        }
    };

    // Copy to clipboard
    const copyToClipboard = () => {
        const text = `Dayflow CDSS Analysis Report
─────────────────────────
Scan: ${scanName}
Date: ${data?.date || new Date().toLocaleDateString()}

DIAGNOSIS: ${analysisData.disease || analysisData.severity}
Final Risk: ${(analysisData.confidence * 100).toFixed(1)}%
Status: ${analysisData.statusMessage || 'Analysis Complete'}

AI Confidence: ${((analysisData.aiConfidence || analysisData.confidence || 0) * 100).toFixed(1)}%
Clinical Boost: +${((analysisData.clinicalBoost || 0) * 100).toFixed(0)}%
Location: ${analysisData.location || 'N/A'}
Method: ${analysisData.method || 'N/A'}

Visual Analysis: ${analysisData.visualAnalysis || 'N/A'}
Clinical Context: ${analysisData.clinicalContext || 'N/A'}
─────────────────────────
Generated by Dayflow CDSS`;

        navigator.clipboard.writeText(text).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    };

    return (
        <div className="results-view">
            <div className="results-header">
                <button onClick={onReset} className="back-button">
                    <ArrowLeft size={18} />
                    Upload New Scan
                </button>
                <div className="results-summary">
                    <span className="scan-name">{scanName}</span>
                    <span className="scan-date">{data?.date || new Date().toLocaleDateString()}</span>
                </div>
            </div>

            <div className="results-content">
                <div className="image-container">
                    <div className="image-wrapper">
                        <img src={imageUrl} alt="Analyzed Scan" className="scan-image" />

                        {isApiData && analysisData.findings?.length > 0 && (
                            <div className="detection-overlay">
                                <span className="detection-label">
                                    {analysisData.location} - {analysisData.method}
                                </span>
                            </div>
                        )}
                    </div>
                </div>

                <div className="analysis-panel">
                    <div className="diagnosis-card">
                        <div className="diagnosis-header">
                            <h3>Diagnosis Result</h3>
                            <span className={`confidence-badge ${analysisData.confidence > 0.6 ? 'high' : analysisData.confidence > 0.3 ? 'medium' : 'low'}`}>
                                <Activity size={14} />
                                {(analysisData.confidence * 100).toFixed(1)}% Final Risk
                            </span>
                        </div>

                        <div className={`disease-highlight ${getRiskClass(analysisData.disease || analysisData.severity)}`}>
                            <div className="disease-icon">
                                {getRiskIcon(analysisData.disease || analysisData.severity)}
                            </div>
                            <div>
                                <h4>{analysisData.disease || analysisData.severity}</h4>
                                <p>{analysisData.statusMessage || `Severity: ${analysisData.severity}`}</p>
                                {analysisData.scanType && (
                                    <span className={`scan-type-badge ${analysisData.scanType?.toLowerCase()}`}>
                                        {analysisData.scanType === 'Bone' ? '🦴' : '🫁'} {analysisData.scanType} Scan
                                        {analysisData.scanTypeConfidence && ` (${(analysisData.scanTypeConfidence * 100).toFixed(0)}%)`}
                                    </span>
                                )}
                                {analysisData.detectedCondition && (
                                    <span className="detected-condition-badge">
                                        {analysisData.detectedCondition}
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* Metrics Dashboard - Only for API data */}
                        {isApiData && (
                            <div className="metrics-dashboard">
                                <div className="metric-card">
                                    <span className="metric-label">AI Confidence</span>
                                    <span className="metric-value">{((analysisData.aiConfidence || 0) * 100).toFixed(1)}%</span>
                                    <span className="metric-sub">{analysisData.location || 'N/A'}</span>
                                </div>
                                <div className="metric-card">
                                    <span className="metric-label">Clinical Boost</span>
                                    <span className="metric-value">+{((analysisData.clinicalBoost || 0) * 100).toFixed(0)}%</span>
                                    <span className="metric-sub">{analysisData.clinicalReasons?.length || 0} Factors</span>
                                </div>
                                <div className="metric-card">
                                    <span className="metric-label">Detection Method</span>
                                    <span className="metric-value-text">{analysisData.method || 'N/A'}</span>
                                </div>
                            </div>
                        )}

                        {/* Interactive Symptoms Panel */}
                        {isApiData && (
                            <div className="symptoms-section">
                                <h4>
                                    {analysisData.scanType === 'Chest' ? '🫁' : '🦴'} Patient Symptoms
                                    <span className="symptoms-hint-inline">Select to update diagnosis</span>
                                </h4>
                                <div className="symptoms-grid-results">
                                    {analysisData.scanType === 'Bone' ? (
                                        <>
                                            <label className={`symptom-checkbox-sm ${symptoms.deformity ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.deformity} onChange={() => handleSymptomChange('deformity')} />
                                                <span>Visible Deformity</span>
                                                <span className="weight">+40%</span>
                                            </label>
                                            <label className={`symptom-checkbox-sm ${symptoms.immobility ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.immobility} onChange={() => handleSymptomChange('immobility')} />
                                                <span>Cannot Move Limb</span>
                                                <span className="weight">+30%</span>
                                            </label>
                                            <label className={`symptom-checkbox-sm ${symptoms.trauma ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.trauma} onChange={() => handleSymptomChange('trauma')} />
                                                <span>Recent Trauma</span>
                                                <span className="weight">+15%</span>
                                            </label>
                                            <label className={`symptom-checkbox-sm ${symptoms.swelling ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.swelling} onChange={() => handleSymptomChange('swelling')} />
                                                <span>Swelling/Bruising</span>
                                                <span className="weight">+10%</span>
                                            </label>
                                            <label className={`symptom-checkbox-sm ${symptoms.pain ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.pain} onChange={() => handleSymptomChange('pain')} />
                                                <span>Localized Pain</span>
                                                <span className="weight">+5%</span>
                                            </label>
                                        </>
                                    ) : (
                                        <>
                                            <label className={`symptom-checkbox-sm ${symptoms.blood_sputum ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.blood_sputum} onChange={() => handleSymptomChange('blood_sputum')} />
                                                <span>Blood in Sputum</span>
                                                <span className="weight">+45%</span>
                                            </label>
                                            <label className={`symptom-checkbox-sm ${symptoms.weight_loss ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.weight_loss} onChange={() => handleSymptomChange('weight_loss')} />
                                                <span>Weight Loss</span>
                                                <span className="weight">+20%</span>
                                            </label>
                                            <label className={`symptom-checkbox-sm ${symptoms.cough ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.cough} onChange={() => handleSymptomChange('cough')} />
                                                <span>Chronic Cough</span>
                                                <span className="weight">+15%</span>
                                            </label>
                                            <label className={`symptom-checkbox-sm ${symptoms.night_sweats ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.night_sweats} onChange={() => handleSymptomChange('night_sweats')} />
                                                <span>Night Sweats</span>
                                                <span className="weight">+10%</span>
                                            </label>
                                            <label className={`symptom-checkbox-sm ${symptoms.fever ? 'checked' : ''}`}>
                                                <input type="checkbox" checked={symptoms.fever} onChange={() => handleSymptomChange('fever')} />
                                                <span>Persistent Fever</span>
                                                <span className="weight">+10%</span>
                                            </label>
                                        </>
                                    )}
                                </div>
                            </div>
                        )}

                        <div className="explanation-section">
                            <h4>AI Reasoning</h4>
                            {isApiData ? (
                                <div className="explanation-text">
                                    <p><strong>Visual Analysis:</strong> {analysisData.visualAnalysis}</p>
                                    <p><strong>Clinical Context:</strong> {analysisData.clinicalContext}</p>
                                    
                                    {analysisData.clinicalReasons?.length > 0 && (
                                        <>
                                            <p><strong>Risk Factors:</strong></p>
                                            <ul>
                                                {analysisData.clinicalReasons.map((reason, idx) => (
                                                    <li key={idx}>{reason}</li>
                                                ))}
                                            </ul>
                                        </>
                                    )}
                                </div>
                            ) : (
                                <p className="explanation-text">
                                    The model identified patterns consistent with the detected condition.
                                    <br /><br />
                                    Key indicators:
                                    <ul>
                                        {analysisData.findings?.map((finding, idx) => (
                                            <li key={idx}>{finding.label} in {finding.region}</li>
                                        )) || <li>No specific findings available</li>}
                                    </ul>
                                </p>
                            )}
                        </div>

                        <div className="recommendation-section">
                            <h4>Recommendations</h4>
                            <ul>
                                {analysisData.scanType === 'Chest' ? (
                                    // Lung/Respiratory recommendations
                                    analysisData.confidence > 0.6 ? (
                                        <>
                                            <li>Immediate pulmonology consultation recommended</li>
                                            <li>Consider sputum test and additional imaging (CT scan)</li>
                                            <li>Isolate patient if TB is suspected</li>
                                            <li>Start empirical treatment if clinically indicated</li>
                                        </>
                                    ) : analysisData.confidence > 0.3 ? (
                                        <>
                                            <li>Clinical correlation with respiratory symptoms suggested</li>
                                            <li>Consider tuberculin skin test or IGRA</li>
                                            <li>Follow-up chest X-ray in 4-6 weeks</li>
                                            <li>Monitor for symptom progression</li>
                                        </>
                                    ) : (
                                        <>
                                            <li>No significant pulmonary findings detected</li>
                                            <li>Continue routine monitoring if symptomatic</li>
                                            <li>Consider other causes of symptoms</li>
                                        </>
                                    )
                                ) : (
                                    // Bone/Fracture recommendations (default)
                                    analysisData.confidence > 0.6 ? (
                                        <>
                                            <li>Immediate clinical evaluation recommended</li>
                                            <li>Consider orthopedic consultation</li>
                                            <li>Immobilization may be required</li>
                                        </>
                                    ) : analysisData.confidence > 0.3 ? (
                                        <>
                                            <li>Clinical correlation suggested</li>
                                            <li>Follow-up imaging may be warranted</li>
                                            <li>Monitor for symptom progression</li>
                                        </>
                                    ) : (
                                        <>
                                            <li>No immediate intervention required</li>
                                            <li>Continue routine monitoring if symptomatic</li>
                                        </>
                                    )
                                )}
                            </ul>
                        </div>

                        <div className="action-buttons">
                            <button className="btn-primary" onClick={generateReport}>
                                <FileText size={18} />
                                Generate Report
                            </button>
                            <button className="btn-secondary" onClick={handleShare}>
                                <Share2 size={18} />
                                Share with Specialist
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Share Modal */}
            {showShareModal && (
                <div className="modal-overlay" onClick={() => setShowShareModal(false)}>
                    <div className="share-modal" onClick={(e) => e.stopPropagation()}>
                        <h3>Share Report</h3>
                        <p>Share this analysis with a specialist</p>
                        
                        <div className="share-options">
                            <button className="share-option" onClick={copyToClipboard}>
                                {copied ? <Check size={20} /> : <Copy size={20} />}
                                {copied ? 'Copied!' : 'Copy to Clipboard'}
                            </button>
                            <button className="share-option" onClick={() => {
                                window.open(`mailto:?subject=Medical Scan Report&body=${encodeURIComponent(`Dayflow CDSS Analysis Report\n\nDiagnosis: ${analysisData.disease || analysisData.severity}\nFinal Risk: ${(analysisData.confidence * 100).toFixed(1)}%`)}`);
                            }}>
                                <Share2 size={20} />
                                Send via Email
                            </button>
                        </div>
                        
                        <button className="modal-close" onClick={() => setShowShareModal(false)}>
                            Close
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ResultsView;
