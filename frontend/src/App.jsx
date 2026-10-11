import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import DatasetExplorer from './pages/DatasetExplorer';
import PreprocessingLab from './pages/PreprocessingLab';
import ModelLab from './pages/ModelLab';
import ExperimentMatrix from './pages/ExperimentMatrix';
import ComparativeAnalysis from './pages/ComparativeAnalysis';
import PredictionDemo from './pages/PredictionDemo';
import ResearchDocs from './pages/ResearchDocs';
import api from './services/api';

export default function App() {
  const [activePage, setActivePage] = useState('home');
  const [health, setHealth] = useState(null);
  const [loadingHealth, setLoadingHealth] = useState(false);

  const fetchHealthData = async () => {
    setLoadingHealth(true);
    try {
      const data = await api.getHealth();
      setHealth(data);
    } catch (err) {
      console.warn('Backend currently offline or starting up:', err.message);
      setHealth({ status: 'offline', hardware: { device: 'CPU' } });
    } finally {
      setLoadingHealth(false);
    }
  };

  useEffect(() => {
    fetchHealthData();
    const timer = setInterval(fetchHealthData, 15000); // Poll health every 15s
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col font-sans text-slate-100">
      <Navbar
        health={health}
        loading={loadingHealth}
        onRefresh={fetchHealthData}
      />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar activePage={activePage} onSelectPage={setActivePage} />

        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          {activePage === 'home' && (
            <Home health={health} onNavigate={setActivePage} />
          )}
          {activePage === 'dataset' && <DatasetExplorer />}
          {activePage === 'preprocessing' && <PreprocessingLab />}
          {activePage === 'models' && <ModelLab />}
          {activePage === 'matrix' && (
            <ExperimentMatrix onNavigate={setActivePage} />
          )}
          {activePage === 'analysis' && <ComparativeAnalysis />}
          {activePage === 'predict' && <PredictionDemo />}
          {activePage === 'docs' && <ResearchDocs />}
        </main>
      </div>
    </div>
  );
}
