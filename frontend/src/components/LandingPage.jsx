import React, { useState } from 'react';
import { ArrowRight, Search, Menu, Calendar, User, CheckCircle2, Phone, Database, FileText, ScanLine, Brain, ShieldCheck, Zap, Activity, Microscope, Layers } from 'lucide-react';
import './LandingPage.css';

const LandingPage = ({ onGetStarted, onNavigate }) => {
    const [activeServiceTab, setActiveServiceTab] = useState('Detection');

    const handleNavClick = (tab, e) => {
        e.preventDefault();
        onNavigate(tab);
    };

    return (
        <div className="landing-page">
            <nav className="landing-nav">
                <div className="landing-logo" onClick={() => onNavigate('landing')} style={{ cursor: 'pointer' }}>MedScan AI</div>
                <div className="landing-links">
                    <a href="#" onClick={(e) => handleNavClick('history', e)}>History</a>
                    <a href="#" onClick={(e) => handleNavClick('settings', e)}>Settings</a>
                </div>
                <div className="landing-nav-actions">
                    <button className="join-btn" onClick={onGetStarted}>Analyze Scan <ArrowRight size={16} /></button>
                </div>
            </nav>

            <section className="hero-section">
                <div className="hero-content">
                    <h1 className="hero-title">
                        Early disease detection with <span className="highlight">explainable AI</span> & contour mapping
                    </h1>
                    <p className="hero-subtitle">
                        Empowering resource-limited healthcare settings with automated, high-precision diagnostics for TB, Pneumonia, and Fractures.
                    </p>
                    <div className="hero-actions">
                        <button className="btn-black" onClick={onGetStarted}>
                            Analyze Scan Now <ArrowRight size={20} />
                        </button>
                        <button className="btn-beige">
                            View Methodology
                        </button>
                    </div>

                    <div className="social-proof">
                        <div className="avatar-stack">
                            {[1, 2, 3, 4].map((i) => (
                                <div key={i} className="avatar-circle" />
                            ))}
                        </div>
                        <div className="proof-text">
                            <strong>10,000+</strong>
                            <span>scans analyzed</span>
                        </div>
                    </div>

                    <div className="partner-bar">
                        <span>Deploy in your clinic?</span>
                        <button className="partner-arrow"><ArrowRight size={16} /></button>
                    </div>
                </div>

                <div className="hero-image-container">
                    <div className="hero-img-placeholder">
                        <div className="floating-card support-card">
                            <div className="card-avatar">
                                <Activity size={20} />
                            </div>
                            <div className="card-info">
                                <strong>High Sensitivity</strong>
                                <span>Early Warning System</span>
                                <span className="yellow-tag">99.8% Acc.</span>
                            </div>
                        </div>
                        <div className="floating-card convenience-card">
                            <div className="card-icon-circle">
                                <Layers size={20} />
                            </div>
                            <strong>Contour Mapping Overlay</strong>
                        </div>
                    </div>
                </div>
            </section>

            <section className="solutions-section">
                <div className="solutions-header">
                    <h2>Advanced diagnostics to effectively enhance your <span className="icon-text"><div className="icon-box"><Brain size={18} /></div> clinical decisions</span>.</h2>
                    <p className="solutions-sub">Robust AI that works with low-resolution scans and noisy data.</p>
                </div>

                <div className="solutions-tabs">
                    <div className="tab-pill">
                        <button
                            className={`tab-btn ${activeServiceTab === 'Detection' ? 'active' : ''}`}
                            onClick={() => setActiveServiceTab('Detection')}
                        >
                            Detection
                        </button>
                        <button
                            className={`tab-btn ${activeServiceTab === 'Analysis' ? 'active' : ''}`}
                            onClick={() => setActiveServiceTab('Analysis')}
                        >
                            Analysis
                        </button>
                    </div>
                    <div className="right-tabs">
                        <span className="plain-tab">Tuberculosis</span>
                        <span className="plain-tab">Fractures</span>
                    </div>
                </div>

                <div className="solutions-grid">
                    <div className="solution-card image-card">
                        {/* Placeholder for X-Ray image */}
                    </div>
                    <div className="solution-card">
                        <div className="icon-circle"><ScanLine size={24} /></div>
                        <h3>Contour Mapping</h3>
                        <p>Visual localization of abnormalities.</p>
                    </div>
                    <div className="solution-card accent-card">
                        <div className="icon-circle"><Brain size={24} /></div>
                        <h3>Explainable AI</h3>
                        <p>Reasoning for every result.</p>
                    </div>
                    <div className="solution-card blue-card">
                        <div className="icon-circle"><Microscope size={24} /></div>
                        <h3>Low-Res Ready</h3>
                    </div>
                    <div className="solution-card">
                        <div className="icon-circle"><ShieldCheck size={24} /></div>
                        <h3>Privacy First</h3>
                        <p>Offline capable.</p>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default LandingPage;
