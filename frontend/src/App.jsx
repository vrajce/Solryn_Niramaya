import { Routes, Route, useLocation, useNavigate } from 'react-router-dom'
import Header from './components/Header'
import Dashboard from './components/Dashboard'
import HistoryView from './components/HistoryView'
import LandingPage from './components/LandingPage'
import './App.css'

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const isLanding = location.pathname === '/';

  // We can pass navigate to components that need to change routes programmatically
  // Or simpler, refactor them to use useNavigate() hook internally.
  // For now, passing onNavigate to minimize prop drill refactor friction in children,
  // but they will just call navigate.

  const handleNavigate = (path) => {
    // Map 'landing' to '/' for router
    if (path === 'landing') navigate('/');
    else navigate(`/${path}`);
  };

  return (
    <div className="app-container">
      {/* Hide Header on Landing Page */}
      {!isLanding && <Header activeTab={location.pathname.substring(1)} onNavigate={handleNavigate} />}
      <main className="app-main">
        <Routes>
          <Route path="/" element={<LandingPage onGetStarted={() => navigate('/dashboard')} onNavigate={handleNavigate} />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<HistoryView />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
