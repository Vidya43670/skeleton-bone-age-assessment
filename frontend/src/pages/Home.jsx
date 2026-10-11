import React from 'react';
import { Sliders, Layers, Grid, Database, ArrowRight, ShieldCheck, Cpu } from 'lucide-react';
import StatCard from '../components/StatCard';


export default function Home({ health, onNavigate }) {
  const stats = [
    {
      title: 'Preprocessing Methods',
      value: health?.research_matrix?.preprocessing_methods_count || 4,
      subtitle: 'P0 Baseline, P1 CLAHE, P2 Denoise, P3 Sharpen',
      icon: Sliders,
      color: 'brand',
    },
    {
      title: 'CNN Architectures',
      value: health?.research_matrix?.cnn_architectures_count || 3,
      subtitle: 'ResNet-18, DenseNet-121, EfficientNet-B0',
      icon: Layers,
      color: 'purple',
    },
    {
      title: 'Research Matrix',
      value: '12 Combinations',
      subtitle: '4 Preprocessing × 3 CNN Models',
      icon: Grid,
      color: 'emerald',
    },
    {
      title: 'Target Classes',
      value: '4 Age Groups',
      subtitle: '0–48, 49–96, 97–144, 145–216 mos',
      icon: Database,
      color: 'amber',
    },
  ];

  return (
    <div className="space-y-6">
      {/*  */}

      {/* Hero Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-brand-950/40 border border-slate-800 rounded-2xl p-8 relative overflow-hidden">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-300 text-xs font-semibold">
            <span>Academic Research Project</span>
            <span>•</span>
            <span>RSNA Pediatric Bone Age Challenge</span>
          </div>

          <h1 className="text-3xl font-extrabold text-white tracking-tight leading-tight">
            Skeleton Bone Age Assessment Using Advanced Image Preprocessing and CNN-Based Classification
          </h1>

          <p className="text-slate-300 text-sm leading-relaxed">
            A comprehensive empirical study analyzing how different radiologic image preprocessing techniques
            (Baseline, CLAHE, Gaussian Denoising, and Unsharp Sharpening) influence multi-class bone age classification
            performance across three foundational deep learning architectures: <strong>ResNet-18</strong>, <strong>DenseNet-121</strong>, and <strong>EfficientNet-B0</strong>.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={() => onNavigate('matrix')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold transition-all shadow-md shadow-brand-500/20"
            >
              <span>View 12-Experiment Matrix</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('preprocessing')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all"
            >
              <span>Explore Preprocessing Lab</span>
            </button>
            <button
              onClick={() => onNavigate('dataset')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all"
            >
              <span>Dataset Explorer</span>
            </button>
          </div>
        </div>
      </div>

      {/* Key Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s, idx) => (
          <StatCard key={idx} {...s} />
        ))}
      </div>

      {/* Research Methodology Workflow */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h2 className="text-base font-bold text-white mb-1">System Architecture & Research Workflow</h2>
        <p className="text-xs text-slate-400 mb-6">
          Rigorous pipeline design ensuring strict data leakage prevention and identical splits across all 12 experimental combinations.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 relative">
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2">
            <div className="text-xs font-bold text-brand-400 uppercase tracking-wide">Step 1 • Ingestion</div>
            <h3 className="text-sm font-semibold text-white">RSNA Pediatric Dataset</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Continuous bone age labels (in months) mapped into 4 discrete research categories (Classes 0–3) with train/val/test stratification.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2">
            <div className="text-xs font-bold text-purple-400 uppercase tracking-wide">Step 2 • Preprocessing</div>
            <h3 className="text-sm font-semibold text-white">4 Image Pipelines</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              P0 Baseline, P1 CLAHE contrast enhancement, P2 radiologic noise reduction, and P3 unsharp bone contour sharpening.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2">
            <div className="text-xs font-bold text-emerald-400 uppercase tracking-wide">Step 3 • Modeling</div>
            <h3 className="text-sm font-semibold text-white">3 Transfer CNNs</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              ResNet-18, DenseNet-121, and EfficientNet-B0 adapted with 4-class output heads, trained under identical seeds & hyperparams.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800/80 space-y-2">
            <div className="text-xs font-bold text-amber-400 uppercase tracking-wide">Step 4 • Evaluation</div>
            <h3 className="text-sm font-semibold text-white">12-Cell Comparative Analysis</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Accuracy, Macro F1-score, Per-Class Precision/Recall, 4x4 Confusion Matrices, and training/inference latency comparison.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
