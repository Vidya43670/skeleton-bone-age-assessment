import React, { useState, useEffect } from 'react';
import {
  Database,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  FolderSearch,
  Info,
  ShieldCheck,
  Split,
  Image as ImageIcon,
  Sparkles
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import api from '../services/api';
import StatCard from '../components/StatCard';
 
export default function DatasetExplorer() {
  const [summary, setSummary] = useState(null);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState(null);
  const [smallSampleMode, setSmallSampleMode] = useState(false);
  const [showUnmatchedDetails, setShowUnmatchedDetails] = useState(false);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const summaryRes = await api.getDatasetSummary();
      setSummary(summaryRes.data);

      if (summaryRes.data?.valid_paired_records > 0) {
        const sampleRes = await api.getDatasetSamples(8);
        setSamples(sampleRes.data || []);
      }
    } catch (err) {
      setError(err.response?.data?.message || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleValidate = async () => {
    setValidating(true);
    setError(null);
    try {
      const payload = smallSampleMode ? { sample_size: 50 } : {};
      const res = await api.validateDataset(payload);
      setSummary(res.data);

      const sampleRes = await api.getDatasetSamples(8);
      setSamples(sampleRes.data || []);
    } catch (err) {
      setError(err.response?.data?.message || err.message);
    } finally {
      setValidating(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const isDetected = summary?.is_detected && summary?.valid_paired_records > 0;

  // Prepare Recharts data for Class Distribution
  const classChartData = summary?.class_distribution
    ? Object.entries(summary.class_distribution).map(([name, data]) => ({
        name: `Class ${data.class_id}`,
        label: data.label,
        count: data.count,
        percentage: data.percentage,
        color:
          data.class_id === 0
            ? '#3b82f6'
            : data.class_id === 1
            ? '#10b981'
            : data.class_id === 2
            ? '#a855f7'
            : '#f59e0b',
      }))
    : [];

  // Prepare Recharts data for Age Histogram
  const histogramData = summary?.age_stats?.detailed?.histogram || [];

  return (
    <div className="space-y-6">
      

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            Dataset Explorer & Analysis
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              RSNA Pediatric Cohort
            </span>
          </h2>
          <p className="text-xs text-slate-400">
            Path validation, continuous age distribution, 4-class conversion, and zero-leakage splits.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Small sample mode toggle */}
          <label className="flex items-center space-x-2 text-xs text-slate-300 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg cursor-pointer hover:bg-slate-800/60">
            <input
              type="checkbox"
              checked={smallSampleMode}
              onChange={(e) => setSmallSampleMode(e.target.checked)}
              className="rounded border-slate-700 text-brand-600 focus:ring-0 bg-slate-950"
            />
            <span>Small-Sample Test (50)</span>
          </label>

          <button
            onClick={loadData}
            disabled={loading}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 flex items-center space-x-1.5 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          <button
            onClick={handleValidate}
            disabled={validating}
            className="px-4 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold flex items-center space-x-1.5 shadow-md shadow-brand-500/20 transition-colors"
          >
            <FolderSearch className={`w-3.5 h-3.5 ${validating ? 'animate-spin' : ''}`} />
            <span>{validating ? 'Validating Dataset...' : 'Run Dataset Validation'}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-red-300 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* High-Level Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Images Found"
          value={summary?.total_images_found ?? 0}
          subtitle="Files on disk (.png, .jpg)"
          icon={Database}
          color="brand"
        />
        <StatCard
          title="Paired Valid Records"
          value={summary?.valid_paired_records ?? 0}
          subtitle="Verified CSV row + uncorrupted image"
          icon={CheckCircle2}
          color="emerald"
        />
        <StatCard
          title="Unmatched Records"
          value={(summary?.missing_images_count || 0) + (summary?.unmatched_image_files?.count || 0)}
          subtitle={`${summary?.missing_images_count || 0} missing img | ${summary?.unmatched_image_files?.count || 0} orphan img`}
          icon={AlertCircle}
          color="amber"
        />
        <StatCard
          title="Class Imbalance Ratio"
          value={summary?.class_imbalance_ratio ? `${summary.class_imbalance_ratio}x` : '—'}
          subtitle={summary?.class_imbalance_ratio ? 'N_max / N_min ratio' : 'Pending validation'}
          icon={Split}
          color="purple"
        />
      </div>

      {/* Dataset Status Banner */}
      {!isDetected ? (
        <div className="bg-slate-900 border border-amber-500/30 rounded-xl p-6">
          <div className="flex items-start space-x-3">
            <Info className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-3">
              <div>
                <h3 className="text-sm font-bold text-amber-300">
                  RSNA Pediatric Bone Age Dataset Setup Required
                </h3>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                  In strict compliance with academic research integrity (Rule #1: <em>Do not fabricate dataset statistics</em>), actual dataset files are required to perform validation.
                </p>
              </div>

              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-xs text-brand-300">
                d:\MiniProject\data\raw\
              </div>

              <div className="text-xs text-slate-400 space-y-1">
                <p>1. Copy <code className="text-slate-200">boneage-training-dataset.csv</code> into <code className="text-slate-200">data/raw/</code></p>
                <p>2. Copy your X-ray image files into <code className="text-slate-200">data/raw/</code> or <code className="text-slate-200">data/raw/boneage-training-dataset/</code></p>
                <p>3. Click <strong>"Run Dataset Validation"</strong> above to compute statistical metrics and generate the leakage-verified train/val/test split.</p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-slate-900 border border-emerald-500/30 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-emerald-300">
          <div className="flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <span>{summary?.status_message}</span>
          </div>
          {summary?.validated_at && (
            <span className="font-mono text-slate-400 text-[11px]">
              Validated: {new Date(summary.validated_at).toLocaleString()}
            </span>
          )}
        </div>
      )}

      {/* Stratified Split & Leakage Verification Panel */}
      {summary?.split_stats && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Standardized 70 / 15 / 15 Split (Data Leakage Verification)</span>
            </h3>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold flex items-center space-x-1">
              <span>Zero Overlap Guaranteed</span>
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-xs text-slate-400 block mb-1">Training Set (70%)</span>
              <span className="text-lg font-bold text-white font-mono">
                {summary.split_stats.train_count} samples
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">Augmentations applied here only</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-xs text-slate-400 block mb-1">Validation Set (15%)</span>
              <span className="text-lg font-bold text-purple-400 font-mono">
                {summary.split_stats.val_count} samples
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">Early stopping & checkpointing</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-xs text-slate-400 block mb-1">Test Set (15%)</span>
              <span className="text-lg font-bold text-emerald-400 font-mono">
                {summary.split_stats.test_count} samples
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">Final 12-cell matrix benchmark</span>
            </div>
          </div>
        </div>
      )}

      {/* Actual Research Experiment Configuration */}
      <div className="bg-slate-900 border border-brand-500/30 rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-brand-400" />
              Research Experiment Training Configuration
            </h3>
            <p className="text-[11px] text-slate-400 mt-1">
              Configuration used for the 12 CNN × preprocessing experiments
            </p>
          </div>

          <span className="text-xs px-2.5 py-1 rounded-full bg-brand-500/10 text-brand-300 border border-brand-500/20 font-semibold">
            75% Training Subset
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block mb-1">
              Valid Records
            </span>
            <span className="text-lg font-bold text-white font-mono">
              12,590
            </span>
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block mb-1">
              Training Split
            </span>
            <span className="text-lg font-bold text-white font-mono">
              8,812
            </span>
            <span className="text-[10px] text-slate-500 block">
              70%
            </span>
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-brand-500/30">
            <span className="text-[10px] text-slate-400 block mb-1">
              Training Used
            </span>
            <span className="text-lg font-bold text-brand-400 font-mono">
              6,609
            </span>
            <span className="text-[10px] text-slate-500 block">
              75% of training split
            </span>
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block mb-1">
              Validation
            </span>
            <span className="text-lg font-bold text-purple-400 font-mono">
              1,889
            </span>
            <span className="text-[10px] text-slate-500 block">
              15%
            </span>
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block mb-1">
              Test
            </span>
            <span className="text-lg font-bold text-emerald-400 font-mono">
              1,889
            </span>
            <span className="text-[10px] text-slate-500 block">
              15%
            </span>
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block mb-1">
              Experiments
            </span>
            <span className="text-lg font-bold text-white font-mono">
              12
            </span>
            <span className="text-[10px] text-slate-500 block">
              3 CNNs × 4 methods
            </span>
          </div>

        </div>

        <div className="text-[11px] text-slate-500 border-t border-slate-800 pt-3">
          The complete dataset is validated above. For the research experiments,
          75% of the 8,812-image training split was used for model training,
          resulting in 6,609 training samples. The validation and test sets
          remained unchanged.
        </div>
      </div>

      {/* Visual Analytics Charts (Recharts) */}
      {isDetected && classChartData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Class Distribution Bar Chart */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white">4-Class Distribution</h3>
                <p className="text-[11px] text-slate-400">Sample count per discrete research age class</p>
              </div>
              <span className="text-xs font-mono text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">
                Imbalance: {summary.class_imbalance_ratio}x
              </span>
            </div>

            <div className="h-60 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={classChartData}>
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.5rem',
                      fontSize: '12px',
                    }}
                    formatter={(value, name, item) => [
                      `${value} samples (${item.payload.percentage}%)`,
                      item.payload.label,
                    ]}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {classChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Continuous Bone Age Histogram */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white">Continuous Bone Age Histogram</h3>
                <p className="text-[11px] text-slate-400">10-bin frequency distribution (months)</p>
              </div>
              <div className="text-[11px] text-slate-400 font-mono">
                Mean: {summary.age_stats?.mean}m | Med: {summary.age_stats?.median}m
              </div>
            </div>

            <div className="h-60 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={histogramData}>
                  <XAxis dataKey="range" stroke="#64748b" fontSize={10} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.5rem',
                      fontSize: '12px',
                    }}
                    formatter={(value) => [`${value} images`, 'Count']}
                  />
                  <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Unmatched and Corrupted File Diagnostics */}
      {isDetected && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white">Record Matching & Integrity Audit</h3>
            <button
              onClick={() => setShowUnmatchedDetails(!showUnmatchedDetails)}
              className="text-xs text-brand-400 hover:text-brand-300 transition-colors"
            >
              {showUnmatchedDetails ? 'Hide Diagnostics' : 'Show Detailed Diagnostics'}
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-400 block mb-0.5">Duplicate CSV IDs:</span>
              <span className="font-bold text-white font-mono">{summary.duplicate_records_count}</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-400 block mb-0.5">Missing Images (CSV):</span>
              <span className="font-bold text-amber-400 font-mono">{summary.missing_images_count}</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-400 block mb-0.5">Unmatched Disk Files:</span>
              <span className="font-bold text-amber-400 font-mono">{summary.unmatched_image_files?.count || 0}</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-400 block mb-0.5">Out of Range (0–216m):</span>
              <span className="font-bold text-white font-mono">{summary.out_of_range_count}</span>
            </div>
          </div>

          {showUnmatchedDetails && (
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800/80 space-y-2 text-xs">
              {summary.unmatched_csv_records?.sample_unmatched_ids?.length > 0 && (
                <div>
                  <span className="text-slate-400 font-semibold block mb-1">
                    Sample CSV IDs lacking image files:
                  </span>
                  <div className="flex flex-wrap gap-1 font-mono text-[11px] text-amber-300">
                    {summary.unmatched_csv_records.sample_unmatched_ids.map((id) => (
                      <span key={id} className="px-1.5 py-0.5 bg-slate-900 rounded border border-slate-800">
                        {id}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {summary.unmatched_image_files?.sample_unmatched_files?.length > 0 && (
                <div>
                  <span className="text-slate-400 font-semibold block mb-1">
                    Sample disk images lacking CSV record:
                  </span>
                  <div className="flex flex-wrap gap-1 font-mono text-[11px] text-amber-300">
                    {summary.unmatched_image_files.sample_unmatched_files.map((f) => (
                      <span key={f} className="px-1.5 py-0.5 bg-slate-900 rounded border border-slate-800">
                        {f}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Sample Hand Radiograph Gallery */}
      {samples.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <ImageIcon className="w-4 h-4 text-brand-400" />
              <span>Sample Verified Radiograph Gallery</span>
            </h3>
            <span className="text-xs text-slate-400">Ground truth labels & mapped classes</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {samples.map((s) => (
              <div
                key={s.id}
                className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col items-center space-y-2"
              >
                <div className="w-full h-36 bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center border border-slate-800">
                  <img
                    src={`/api/dataset/image/${s.id}`}
                    alt={`X-ray ${s.id}`}
                    className="max-h-full object-contain"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.style.display = 'none';
                    }}
                  />
                </div>
                <div className="w-full space-y-1 text-xs">
                  <div className="flex items-center justify-between font-mono">
                    <span className="text-slate-400 text-[11px]">ID: {s.id}</span>
                    <span className="font-semibold text-white">{s.boneage_months}m</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-brand-500/10 text-brand-300 border border-brand-500/20 font-semibold">
                      Class {s.class_id}
                    </span>
                    <span className="text-[10px] text-slate-400">
                      {s.male === true ? 'Male' : s.male === false ? 'Female' : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
