import React, { useState, useEffect } from 'react';
import { Sliders, Upload, Play, CheckCircle2, AlertCircle, Eye } from 'lucide-react';
import api from '../services/api';
import AcademicDisclaimer from '../components/AcademicDisclaimer';

export default function PreprocessingLab() {
  const [methods, setMethods] = useState([]);
  const [selectedMethod, setSelectedMethod] = useState('P1_clahe');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [processedResult, setProcessedResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMethods = async () => {
      try {
        const res = await api.getPreprocessingMethods();
        setMethods(res.data);
      } catch (err) {
        console.error(err);
      }
    };
    fetchMethods();
  }, []);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setProcessedResult(null);
      setError(null);
    }
  };

  const handleRunPreview = async () => {
    if (!imageFile) {
      setError('Please upload an X-ray image to preview the preprocessing pipeline.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('method_id', selectedMethod);

      const res = await api.previewPreprocessing(formData);
      setProcessedResult(res);
    } catch (err) {
      setError(err.response?.data?.message || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      

      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          Image Preprocessing Laboratory
          <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-brand-300 border border-brand-500/20">
            P0 – P3 Pipelines
          </span>
        </h2>
        <p className="text-xs text-slate-400">
          Side-by-side visual analysis and parametric tuning of radiologic contrast, denoising, and edge enhancement.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-red-300 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Preprocessing Selection Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        {methods.map((m) => {
          const isSelected = selectedMethod === m.id;
          return (
            <button
              key={m.id}
              onClick={() => {
                setSelectedMethod(m.id);
                setProcessedResult(null);
              }}
              className={`p-4 rounded-xl text-left border transition-all ${
                isSelected
                  ? 'bg-brand-950/40 border-brand-500 text-white ring-1 ring-brand-500 shadow-sm'
                  : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-800/50'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold uppercase tracking-wider text-brand-400">{m.code}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                  {m.id}
                </span>
              </div>
              <div className="text-sm font-bold text-white mb-1">{m.name}</div>
              <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">{m.description}</p>
            </button>
          );
        })}
      </div>

      {/* Main Interactive Studio */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls & Upload Panel */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-brand-400" />
            <span>Interactive Controls</span>
          </h3>

          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300 block">
              1. Select Test Hand X-Ray Image
            </label>
            <div className="border-2 border-dashed border-slate-800 rounded-xl p-6 text-center hover:border-slate-700 transition-colors">
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleImageChange}
                className="hidden"
                id="xray-upload"
              />
              <label htmlFor="xray-upload" className="cursor-pointer flex flex-col items-center space-y-2">
                <Upload className="w-6 h-6 text-slate-400" />
                <span className="text-xs font-medium text-slate-300">
                  {imageFile ? imageFile.name : 'Click to upload pediatric X-ray'}
                </span>
                <span className="text-[10px] text-slate-500">Supports PNG, JPG (RSNA format)</span>
              </label>
            </div>
          </div>

          <div className="space-y-2 pt-2">
            <label className="text-xs font-medium text-slate-300 block">
              2. Pipeline Execution
            </label>
            <button
              onClick={handleRunPreview}
              disabled={loading || !imageFile}
              className="w-full py-2.5 px-4 rounded-lg bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold flex items-center justify-center space-x-2 shadow-md shadow-brand-500/20 transition-all"
            >
              <Play className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              <span>{loading ? 'Processing Filter...' : `Apply ${selectedMethod.toUpperCase()}`}</span>
            </button>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800/80 text-xs text-slate-400 space-y-1.5">
            <span className="font-semibold text-slate-300 block text-[11px] uppercase tracking-wider">Method Parameters:</span>
            {selectedMethod === 'P0_baseline' && (
              <p>• Output Dimension: 224×224<br/>• Normalization: Min-Max (0–255)</p>
            )}
            {selectedMethod === 'P1_clahe' && (
              <p>• Clip Limit: 2.0<br/>• Tile Grid Size: 8×8 contextual regions</p>
            )}
            {selectedMethod === 'P2_denoising' && (
              <p>• Kernel Size: 3×3<br/>• Filter: Gaussian Filtering (σ = 0.8)</p>
            )}
            {selectedMethod === 'P3_sharpening' && (
              <p>• Method: Unsharp Masking<br/>• Amount: 1.5×, Radius: 1.0</p>
            )}
          </div>
        </div>

        {/* Visual Comparison Stage */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <Eye className="w-4 h-4 text-emerald-400" />
              <span>Side-by-Side Radiologic Comparison</span>
            </h3>
            {processedResult && (
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                Filter Applied
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 min-h-[340px]">
            {/* Original Box */}
            <div className="border border-slate-800 rounded-xl bg-slate-950 p-3 flex flex-col items-center justify-center relative overflow-hidden">
              <span className="text-xs font-semibold text-slate-400 mb-2">Original Input X-ray</span>
              {imagePreview ? (
                <img
                  src={imagePreview}
                  alt="Original X-ray"
                  className="max-h-72 object-contain rounded border border-slate-800"
                />
              ) : (
                <div className="text-center text-slate-600 text-xs py-16">
                  No image selected yet.<br />Upload a hand radiograph on the left.
                </div>
              )}
            </div>

            {/* Processed Box */}
            <div className="border border-slate-800 rounded-xl bg-slate-950 p-3 flex flex-col items-center justify-center relative overflow-hidden">
              <span className="text-xs font-semibold text-brand-400 mb-2">
                Processed ({selectedMethod})
              </span>
              {processedResult?.processed_image ? (
                <img
                  src={processedResult.processed_image}
                  alt="Processed Radiograph"
                  className="max-h-72 object-contain rounded border border-slate-800"
                />
              ) : (
                <div className="text-center text-slate-600 text-xs py-16">
                  {imageFile
                    ? 'Ready to run filter. Click "Apply" to view.'
                    : 'Awaiting image selection.'}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
