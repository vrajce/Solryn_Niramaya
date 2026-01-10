import React, { useState, useRef } from 'react';
import { CloudUpload, ScanLine } from 'lucide-react';
import './UploadArea.css';

const UploadArea = ({ onUpload, isAnalyzing }) => {
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (file) => {
        // Validate file type
        if (file.type.startsWith('image/')) {
            onUpload(file);
        } else {
            alert("Please upload a valid image file.");
        }
    };

    const onButtonClick = () => {
        inputRef.current.click();
    };

    return (
        <div
            className={`upload-area ${dragActive ? 'drag-active' : ''} ${isAnalyzing ? 'analyzing' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={onButtonClick}
        >
            <input
                ref={inputRef}
                type="file"
                className="upload-input"
                accept="image/*"
                onChange={handleChange}
            />

            <div className="upload-content">
                <div className="upload-icon-wrapper">
                    <ScanLine size={36} />
                </div>
                <h3 className="upload-title">Upload Medical Scan</h3>
                <p className="upload-subtitle">Drag and drop your X-ray or CT scan here to detect abnormalities with high precision.</p>

                <span className="upload-divider">or</span>

                <button className="upload-button" onClick={(e) => {
                    e.stopPropagation();
                    onButtonClick();
                }}>
                    Browse Files
                </button>

                <p className="upload-hint">Supported formats: JPG, PNG, DICOM (preview only)</p>
            </div>
        </div>
    );
};

export default UploadArea;
