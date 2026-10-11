import React, { useState, useEffect } from 'react';
import { Grid, Play, AlertCircle, CheckCircle2, Clock, RotateCcw, Flame } from 'lucide-react';
import api from '../services/api';
import AcademicDisclaimer from '../components/AcademicDisclaimer';

export default function ExperimentMatrix({ onNavigate }) {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actionMessage, setActionMessage] = useState(null);

  const loadExperiments = async () => {
    setLoading(true);
    try {
      const res = await api.getExperiments();
      setExperiments(res.data);
    } catch (err) {
      setError(err.response?.data?.message || err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadExperiments();
  }, []);

  const handleTrain = async (id, isSmoke = false) => {
  setError(null);
  setActionMessage(null);

  try {
    if (!id) {
      setError('Experiment ID not found.');
      return;
    }

    // Check the current experiment status from the backend
    const experiment = await api.getExperiment(id);

    // If results already exist, fetch them instead of training again
    if (experiment.status === 'COMPLETED') {
      await api.getExperiment(id);
      setActionMessage(
        `Loaded stored results for experiment '${id}'. No new training was started.`
      );
      await loadExperiments();
      return;
    }

    // Never start another training process while one is running
    if (experiment.status === 'RUNNING') {
      setActionMessage(
        `Experiment '${id}' is currently running. Existing backend status loaded.`
      );
      await loadExperiments();
      return;
    }

    // Only NOT_STARTED or FAILED experiments can start training
    const res = await api.trainExperiment(id, { smoke_test: isSmoke });
    setActionMessage(res.message);
    await loadExperiments();

  } catch (err) {
    setError(err.response?.data?.message || err.message);
  }
};

  const getExperiment = (prepId, modelId) => {
    return experiments.find(
      (e) => e.preprocessing_id === prepId && e.model_name === modelId
    );
  };

  const PREP_ROWS = [
    { id: 'P0_baseline', name: 'P0 Baseline', desc: 'Standard Resizing & Normalization' },
    { id: 'P1_clahe', name: 'P1 CLAHE', desc: 'Contrast Limited Adaptive Equalization' },
    { id: 'P2_denoising', name: 'P2 Denoising', desc: 'Edge-Preserving / Gaussian Filtering' },
    { id: 'P3_sharpening', name: 'P3 Sharpening', desc: 'Unsharp Masking Trabeculae Enhancement' },
  ];

  const MODEL_COLS = [
    { id: 'resnet18', name: 'ResNet-18', params: '11.2M' },
    { id: 'densenet121', name: 'DenseNet-121', params: '7.0M' },
    { id: 'efficientnet_b0', name: 'EfficientNet-B0', params: '4.0M' },
  ];

  return (
    <div className="space-y-6">
      

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            12-Experiment Research Matrix
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-emerald-300 border border-emerald-500/20">
              4 Preprocessing × 3 CNNs
            </span>
          </h2>
          <p className="text-xs text-slate-400">
            Systematic benchmark across all permutations. Every cell shares identical splits and seeds.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={loadExperiments}
            disabled={loading}
            className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 flex items-center space-x-1.5 transition-colors"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-red-300 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {actionMessage && (
        <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 text-xs flex items-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          <span>{actionMessage}</span>
        </div>
      )}

      {/* The 4x3 Matrix Grid */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-950/60 text-xs">
                <th className="p-4 font-bold text-slate-400 uppercase tracking-wider w-1/4">
                  Preprocessing Pipeline
                </th>
                {MODEL_COLS.map((col) => (
                  <th key={col.id} className="p-4 font-bold text-white tracking-wide">
                    <div className="flex items-center justify-between">
                      <span>{col.name}</span>
                      <span className="text-[10px] font-mono font-normal px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                        {col.params}
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 text-xs">
              {PREP_ROWS.map((row) => (
                <tr key={row.id} className="hover:bg-slate-800/30 transition-colors">
                  <td className="p-4 bg-slate-950/40 border-r border-slate-800/80">
                    <div className="font-bold text-brand-400 mb-0.5">{row.name}</div>
                    <div className="text-[11px] text-slate-500 leading-snug">{row.desc}</div>
                  </td>
                  {MODEL_COLS.map((col) => {
                    const exp = getExperiment(row.id, col.id);
                    const status = exp?.status || 'NOT_STARTED';
                    const hasMetrics = exp?.test_accuracy !== null && exp?.test_accuracy !== undefined;

                    return (
                      <td key={col.id} className="p-4 border-r border-slate-800/80 last:border-r-0">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span
                              className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full border ${
                                status === 'COMPLETED'
                                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                                  : status === 'RUNNING'
                                  ? 'bg-blue-500/10 text-blue-400 border-blue-500/20 animate-pulse'
                                  : status === 'FAILED'
                                  ? 'bg-red-500/10 text-red-400 border-red-500/20'
                                  : 'bg-slate-800 text-slate-400 border-slate-700'
                              }`}
                            >
                              {status === 'NOT_STARTED' ? 'Awaiting Data / Run' : status}
                            </span>
                            {exp?.is_smoke_test && (
                              <span className="text-[9px] px-1 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
                                Smoke Test
                              </span>
                            )}
                          </div>

                          {hasMetrics ? (
                            <div className="space-y-1 font-mono text-[11px] pt-1">
                              <div className="flex justify-between text-slate-400">
                                <span>Accuracy:</span>
                                <span className="font-bold text-white">
                                  {(exp.test_accuracy * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="flex justify-between text-slate-400">
                                <span>Macro F1:</span>
                                <span className="font-bold text-emerald-400">
                                  {exp.macro_f1.toFixed(3)}
                                </span>
                              </div>
                            </div>
                          ) : (
                            <p className="text-[11px] text-slate-500">
                              Experiment unrun. No fabricated results present.
                            </p>
                          )}

                          <div className="flex items-center space-x-2 pt-1">
                            <button
                              onClick={() => handleTrain(exp?.id, true)}
                              disabled={status === 'RUNNING'}
                              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-medium border border-slate-700 flex items-center space-x-1"
                              title="Smoke test using 20 images to verify pipeline without long training"
                            >
                              <Flame className="w-3 h-3 text-amber-400" />
                              <span>Smoke Test</span>
                            </button>
                            <button
                              onClick={() => handleTrain(exp?.id, false)}
                              disabled={status === 'RUNNING'}
                              className="px-2 py-1 rounded bg-brand-600/80 hover:bg-brand-500 text-white text-[10px] font-medium flex items-center space-x-1"
                              title="Full training run"
                            >
                              <Play className="w-3 h-3" />
                              <span>Full Train</span>
                            </button>
                          </div>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
