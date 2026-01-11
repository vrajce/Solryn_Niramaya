import React from 'react';
import { History, Activity } from 'lucide-react';
import './Header.css';

const Header = ({ activeTab, onNavigate }) => {
    return (
        <header className="header">
            <div className="header-container">
                <div className="header-logo" onClick={() => onNavigate('landing')}>
                    <div className="logo-icon">
                        <Activity size={28} strokeWidth={2.5} />
                    </div>
                    <span className="logo-text">Niramaya</span>
                </div>
                <nav className="header-nav">
                    <button
                        className={`nav-link ${activeTab === 'history' ? 'active' : ''}`}
                        onClick={() => onNavigate('history')}
                    >
                        <History size={18} style={{ marginRight: '8px' }} />
                        History
                    </button>
                </nav>
            </div>
        </header>
    );
};

export default Header;
