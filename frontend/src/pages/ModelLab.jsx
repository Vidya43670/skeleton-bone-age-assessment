import React, { useState, useEffect } from 'react';
import { Layers, Cpu, Sliders, CheckCircle2, Box, Zap } from 'lucide-react';
import api from '../services/api';
import AcademicDisclaimer from '../components/AcademicDisclaimer';

export default function ModelLab() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await api.getModels();
        setModels(res.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchModels();
  }, []);

  return (
    <div className="space-y-6">
      

      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          CNN Model Architecture Laboratory
          <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-purple-300 border border-purple-500/20">
            PyTorch & torchvision
          </span>
        </h2>
        <p className="text-xs text-slate-400">
          Comparative deep learning architectures adapted for 4-class pediatric bone age classification.
        </p>
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {models.map((m) => (
          <div
            key={m.id}
            className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col justify-between hover:border-slate-700 transition-all shadow-sm"
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">
                  CNN Backbone
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/10 text-purple-300 border border-purple-500/20 font-mono">
                  {m.approx_params}
                </span>
              </div>

              <div>
                <h3 className="text-lg font-bold text-white mb-1">{m.name}</h3>
                <span className="text-xs text-slate-500 block">{m.family}</span>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed">
                {m.description}
              </p>

              <div className="space-y-2 pt-2 border-t border-slate-800/80 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-800/40">
                  <span className="text-slate-400">Target Head</span>
                  <span className="font-semibold text-white">{m.adapted_num_classes} Discrete Classes</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/40">
                  <span className="text-slate-400">Input Resolution</span>
                  <span className="font-mono text-slate-300">{m.input_resolution}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/40">
                  <span className="text-slate-400">Transfer Weights</span>
                  <span className="font-mono text-emerald-400">{m.weights}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-400">Fine-Tuning Mode</span>
                  <span className="text-slate-300">End-to-End Backprop</span>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between text-xs">
              <div className="flex items-center space-x-1.5 text-emerald-400">
                <CheckCircle2 className="w-4 h-4" />
                <span>Ready for Matrix Benchmarking</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Transfer Learning Methodology Details */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h3 className="text-sm font-bold text-white mb-2 flex items-center space-x-2">
          <Zap className="w-4 h-4 text-amber-400" />
          <span>Academic Transfer Learning Implementation Protocol</span>
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed mb-4">
          To ensure strict experimental fairness across all 12 combinations in the research matrix:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
            <span className="font-bold text-brand-400 block mb-1">1. Uniform Seed Control</span>
            <p className="text-slate-400 leading-relaxed">
              Deterministic random seed (42) applied across PyTorch, NumPy, and data split generators to eliminate sampling bias.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
            <span className="font-bold text-purple-400 block mb-1">2. Layer Adaptation</span>
            <p className="text-slate-400 leading-relaxed">
              Final fully connected linear projection layer replaced with a linear layer mapped to <code className="text-purple-300">num_classes=4</code> with Cross-Entropy Loss.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
            <span className="font-bold text-emerald-400 block mb-1">3. Early Stopping & Checkpoints</span>
            <p className="text-slate-400 leading-relaxed">
              Validation macro F1-score tracked every epoch. The best model checkpoint is automatically preserved in <code className="text-emerald-300">models/</code>.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
