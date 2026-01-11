import React, { useState, useEffect } from 'react';
import { Search, Filter, Eye, AlertTriangle, CheckCircle, Clock, Trash2, RefreshCw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './HistoryView.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const HistoryView = () => {
    const navigate = useNavigate();
    const [historyData, setHistoryData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');

    // Fetch history on mount
    useEffect(() => {
        fetchHistory();
    }, []);

    const fetchHistory = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/history?limit=50`);
            if (response.ok) {
                const data = await response.json();
                setHistoryData(data);
            }
        } catch (error) {
            console.error('Failed to fetch history:', error);
        } finally {
            setLoading(false);
        }
    };

    const deleteItem = async (id, e) => {
        e.stopPropagation();
        try {
            await fetch(`${API_BASE_URL}/history/${id}`, { method: 'DELETE' });
            setHistoryData(prev => prev.filter(item => item.id !== id));
        } catch (error) {
            console.error('Failed to delete:', error);
        }
    };

    const filteredData = historyData.filter(item => 
        item.patientId?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.result?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.scanType?.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const getStatusIcon = (status) => {
        switch (status) {
            case 'Reviewed': return <CheckCircle size={14} />;
            case 'Flagged': return <AlertTriangle size={14} />;
            default: return <Clock size={14} />;
        }
    };

    const handleViewDetails = (item) => {
        navigate('/dashboard', { state: { historyItem: item } });
    };

    return (
        <div className="history-view">
            <div className="history-header">
                <h2>Scan History</h2>
                <div className="history-filters">
                    <div className="search-wrapper">
                        <Search size={16} className="search-icon" />
                        <input 
                            type="text" 
                            placeholder="Search Patient ID..." 
                            className="search-input"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>
                    <button className="filter-btn" onClick={fetchHistory} title="Refresh">
                        <RefreshCw size={16} className={loading ? 'spinning' : ''} />
                        Refresh
                    </button>
                </div>
            </div>

            <div className="table-container">
                {loading ? (
                    <div className="loading-state">Loading history...</div>
                ) : filteredData.length === 0 ? (
                    <div className="empty-state">
                        <p>No scan history found</p>
                        <span>Analyzed scans will appear here</span>
                    </div>
                ) : (
                    <table className="history-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Patient ID</th>
                                <th>Scan Type</th>
                                <th>Result</th>
                                <th>Confidence</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredData.map((item) => (
                                <tr key={item.id}>
                                    <td>{item.date} {item.time}</td>
                                    <td>{item.patientId}</td>
                                    <td>
                                        <span className={`scan-type-tag ${item.scanType?.toLowerCase()}`}>
                                            {item.scanType === 'Bone' ? '🦴' : '🫁'} {item.scanType}
                                        </span>
                                    </td>
                                    <td>
                                        <span className={`result-tag ${item.result?.includes('LOW') ? 'normal' : 'abnormal'}`}>
                                            {item.result}
                                        </span>
                                    </td>
                                    <td>{((item.confidence || 0) * 100).toFixed(0)}%</td>
                                    <td>
                                        <span className={`status-badge ${item.status?.toLowerCase().replace(' ', '-')}`}>
                                            {getStatusIcon(item.status)}
                                            {item.status}
                                        </span>
                                    </td>
                                    <td className="actions-cell">
                                        <button className="action-link" onClick={() => handleViewDetails(item)}>
                                            <Eye size={16} />
                                            View
                                        </button>
                                        <button className="action-link delete" onClick={(e) => deleteItem(item.id, e)}>
                                            <Trash2 size={16} />
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
};

export default HistoryView;
