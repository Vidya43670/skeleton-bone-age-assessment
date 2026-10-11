import React, { useState } from 'react';
import { Stethoscope, Upload, Play, AlertCircle, CheckCircle2, ShieldAlert, Sparkles } from 'lucide-react';
import api from '../services/api';
import AcademicDisclaimer from '../components/AcademicDisclaimer';

export default function PredictionDemo() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [selectedModel, setSelectedModel] = useState('resnet18');
  const [selectedPrep, setSelectedPrep] = useState('P0_baseline');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [notReadyMessage, setNotReadyMessage] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
      setNotReadyMessage(null);
    }
  };

  const handlePredict = async () => {
    if (!imageFile) {
      setError('Please select an X-ray image for inference.');
      return;
    }
    setLoading(true);
    setError(null);
    setNotReadyMessage(null);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('model_name', selectedModel);
      formData.append('preprocessing_id', selectedPrep);

      const res = await api.predict(formData);
      setResult(res);
    } catch (err) {
      if (err.response?.data?.status === 'not_ready') {
        setNotReadyMessage(err.response.data.message);
      } else {
        setError(err.response?.data?.message || err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      

      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          Bone Age Prediction Demonstration
          <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-amber-300 border border-amber-500/20">
            Interactive Testbed
          </span>
        </h2>
        <p className="text-xs text-slate-400">
          Upload a pediatric radiograph and test trained model inference with confidence distributions.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-red-300 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {notReadyMessage && (
        <div className="p-5 rounded-xl bg-slate-900 border border-amber-500/40 text-xs space-y-2">
          <div className="flex items-center space-x-2 text-amber-400 font-bold">
            <ShieldAlert className="w-4 h-4" />
            <span>Model Checkpoint Required</span>
          </div>
          <p className="text-slate-300 leading-relaxed">{notReadyMessage}</p>
          <div className="text-[11px] text-slate-500">
            In accordance with Rule #1, inference requires an actual trained checkpoint saved on disk.
          </div>
        </div>
      )}

      {/* Inference Configuration Form */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span>Inference Configuration</span>
          </h3>

          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300 block">1. Target Hand X-Ray</label>
            <div className="border-2 border-dashed border-slate-800 rounded-xl p-6 text-center hover:border-slate-700 transition-colors">
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={handleImageChange}
                className="hidden"
                id="predict-upload"
              />
              <label htmlFor="predict-upload" className="cursor-pointer flex flex-col items-center space-y-2">
                <Upload className="w-6 h-6 text-slate-400" />
                <span className="text-xs font-medium text-slate-300">
                  {imageFile ? imageFile.name : 'Select Pediatric Hand X-Ray'}
                </span>
                <span className="text-[10px] text-slate-500">DICOM export or PNG/JPG</span>
              </label>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300 block">2. Select CNN Architecture</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-brand-500"
            >
              <option value="resnet18">ResNet-18 (Residual Network)</option>
              <option value="densenet121">DenseNet-121 (Dense Network)</option>
              <option value="efficientnet_b0">EfficientNet-B0 (Compound Scaling)</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-300 block">3. Select Preprocessing Pipeline</label>
            <select
              value={selectedPrep}
              onChange={(e) => setSelectedPrep(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-brand-500"
            >
              <option value="P0_baseline">P0: Baseline (Standard Resize & Norm)</option>
              <option value="P1_clahe">P1: CLAHE (Contrast Enhancement)</option>
              <option value="P2_denoising">P2: Denoising (Gaussian Filter)</option>
              <option value="P3_sharpening">P3: Sharpening (Unsharp Masking)</option>
            </select>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading || !imageFile}
            className="w-full py-2.5 px-4 rounded-lg bg-brand-600 hover:bg-brand-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold flex items-center justify-center space-x-2 shadow-md shadow-brand-500/20 transition-all"
          >
            <Play className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>{loading ? 'Evaluating Model...' : 'Run Bone Age Assessment'}</span>
          </button>
        </div>

        {/* Preview and Results */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-bold text-white">Assessment Results</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 min-h-[300px]">
            <div className="border border-slate-800 rounded-xl bg-slate-950 p-4 flex flex-col items-center justify-center">
              <span className="text-xs font-semibold text-slate-400 mb-2">Radiograph Input</span>
              {imagePreview ? (
                <img
                  src={imagePreview}
                  alt="X-ray radiograph"
                  className="max-h-64 object-contain rounded border border-slate-800"
                />
              ) : (
                <div className="text-center text-slate-600 text-xs py-16">
                  Select an X-ray image to preview
                </div>
              )}
            </div>

            <div className="border border-slate-800 rounded-xl bg-slate-950 p-5 flex flex-col justify-center space-y-4">
             <span className="text-xs font-semibold text-slate-400">
  Class Probabilities
</span>

{result?.prediction ? (
  <div className="space-y-5">

    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4">
      <div className="text-xs text-slate-400 mb-1">
        Predicted Age Group
      </div>

      <div className="text-xl font-bold text-white">
        {result.prediction.age_range}
      </div>

      <div className="text-sm text-slate-400 mt-2">
        Confidence:{' '}
        <span className="text-white font-semibold">
          {result.prediction.confidence}%
        </span>
      </div>
    </div>

    <div className="space-y-3">
      {result.class_probabilities?.map((item) => (
        <div key={item.class_id}>

          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-300">
              {item.label}
            </span>

            <span className="text-slate-400">
              {item.percentage}%
            </span>
          </div>

          <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-cyan-500 transition-all duration-500"
              style={{ width: `${item.percentage}%` }}
            />
          </div>

        </div>
      ))}
    </div>

    <div className="text-xs text-slate-500 border-t border-slate-800 pt-3">
      Model: {result.model?.display_name}
      <br />
      Preprocessing: {result.preprocessing?.name}
    </div>

  </div>
) : (
  <div className="text-center text-slate-600 text-xs py-8">
    Upload an X-ray and click{' '}
    <span className="text-slate-400">
      Run Prediction
    </span>{' '}
    to view the results.
  </div>
)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
