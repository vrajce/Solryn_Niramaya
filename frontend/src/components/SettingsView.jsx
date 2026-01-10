import React, { useState } from 'react';
import { Sliders, Shield, Mail, Bell, Wand2 } from 'lucide-react';
import './SettingsView.css';

const SettingsView = () => {
    const [settings, setSettings] = useState({
        autoEnhance: true,
        showConfidence: true,
        darkMode: true,
        emailNotifications: false,
        dataRetention: '30days'
    });

    const handleToggle = (key) => {
        setSettings(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setSettings(prev => ({ ...prev, [name]: value }));
    };

    return (
        <div className="settings-view">
            <div className="settings-header">
                <h2>Settings</h2>
                <p>Manage your preferences and system configuration.</p>
            </div>

            <div className="settings-grid">
                <div className="settings-card">
                    <h3>
                        <Sliders size={20} className="card-icon" />
                        Analysis Preferences
                    </h3>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label><Wand2 size={16} /> Auto-Enhance Images</label>
                            <p>Automatically adjust contrast and brightness for better visibility.</p>
                        </div>
                        <label className="switch">
                            <input
                                type="checkbox"
                                checked={settings.autoEnhance}
                                onChange={() => handleToggle('autoEnhance')}
                            />
                            <span className="slider round"></span>
                        </label>
                    </div>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label><Shield size={16} /> Show Confidence Scores</label>
                            <p>Display AI probability percentages next to diagnoses.</p>
                        </div>
                        <label className="switch">
                            <input
                                type="checkbox"
                                checked={settings.showConfidence}
                                onChange={() => handleToggle('showConfidence')}
                            />
                            <span className="slider round"></span>
                        </label>
                    </div>
                </div>

                <div className="settings-card">
                    <h3>
                        <Shield size={20} className="card-icon" />
                        System & Data
                    </h3>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label><Shield size={16} /> Data Retention Policy</label>
                            <p>How long to keep patient scans in local history.</p>
                        </div>
                        <select
                            name="dataRetention"
                            value={settings.dataRetention}
                            onChange={handleChange}
                            className="setting-select"
                        >
                            <option value="7days">7 Days</option>
                            <option value="30days">30 Days</option>
                            <option value="90days">90 Days</option>
                            <option value="forever">Forever</option>
                        </select>
                    </div>

                    <div className="setting-item">
                        <div className="setting-info">
                            <label><Mail size={16} /> Email Notifications</label>
                            <p>Receive alerts when analysis is complete.</p>
                        </div>
                        <label className="switch">
                            <input
                                type="checkbox"
                                checked={settings.emailNotifications}
                                onChange={() => handleToggle('emailNotifications')}
                            />
                            <span className="slider round"></span>
                        </label>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SettingsView;
