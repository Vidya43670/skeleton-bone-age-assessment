import React from 'react';
import { BookOpen, FileText, CheckCircle2, ShieldAlert } from 'lucide-react';
import AcademicDisclaimer from '../components/AcademicDisclaimer';

export default function ResearchDocs() {
  return (
    <div className="space-y-6 max-w-5xl">
      

      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          Academic Research Documentation & Methodology
        </h2>
        <p className="text-xs text-slate-400">
          Theoretical foundation, mathematical formulations, and experimental design protocols.
        </p>
      </div>

      {/* Section 1: Problem Statement */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
        <h3 className="text-sm font-bold text-brand-400 uppercase tracking-wide">
          1. Research Background & Problem Formulation
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          Bone age assessment (BAA) is a key clinical radiologic procedure used to diagnose pediatric endocrine disorders, growth anomalies, and genetic syndromes. Historically, radiologists manually compare left-hand radiographs against reference atlases such as the <strong>Greulich-Pyle (GP)</strong> or <strong>Tanner-Whitehouse (TW)</strong> standards. However, manual estimation suffers from significant intra- and inter-observer variability (often ±1 to ±2 years).
        </p>
        <p className="text-xs text-slate-300 leading-relaxed">
          While modern deep learning models achieve high accuracy on raw images, pediatric radiographs exhibit non-standardized brightness, variable radiologic tube voltages, and detector noise. This study investigates the systematic impact of <strong>radiologic image preprocessing</strong> (contrast adaptation, denoising, and trabecular sharpening) across diverse convolutional architectures for multi-class skeletal maturity classification.
        </p>
      </div>

      {/* Section 2: Mathematical Formulations */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
        <h3 className="text-sm font-bold text-purple-400 uppercase tracking-wide">
          2. Radiologic Preprocessing Mathematical Formulations
        </h3>

        <div className="space-y-3 text-xs">
          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
            <span className="font-bold text-white">P1: Contrast Limited Adaptive Histogram Equalization (CLAHE)</span>
            <p className="text-slate-400 leading-relaxed">
              Standard histogram equalization over-amplifies noise in homogeneous soft-tissue regions. CLAHE operates on localized contextual tiles (e.g. 8×8), clipping the histogram at threshold &beta; before redistributing excess pixels uniformly across bins:
            </p>
            <div className="font-mono text-brand-300 bg-slate-900/80 p-2 rounded text-[11px]">
              {'clip_limit = 2.0, tile_grid = (8, 8)'}
            </div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
            <span className="font-bold text-white">P2: Radiologic Gaussian Noise Attenuation</span>
            <p className="text-slate-400 leading-relaxed">
              High-frequency detector quantum mottle is attenuated via isotropic 2D Gaussian convolution:
            </p>
            <div className="font-mono text-purple-300 bg-slate-900/80 p-2 rounded text-[11px]">
              {'G(x, y) = (1 / (2*π*σ²)) * exp(-(x² + y²) / (2*σ²)),  σ = 0.8,  k = 3×3'}
            </div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
            <span className="font-bold text-white">P3: Epiphyseal Boundary Unsharp Masking</span>
            <p className="text-slate-400 leading-relaxed">
              Enhances edge transitions between the epiphysis and metaphysis without increasing noise in uniform areas:
            </p>
            <div className="font-mono text-emerald-300 bg-slate-900/80 p-2 rounded text-[11px]">
              {'I_sharp(x, y) = I(x, y) + α * (I(x, y) - G_σ(x, y) * I(x, y)),  α = 1.5'}
            </div>
          </div>
        </div>
      </div>

      {/* Section 3: Evaluation Metrics */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
        <h3 className="text-sm font-bold text-emerald-400 uppercase tracking-wide">
          3. Experimental Protocol & Evaluation Metrics
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          The 4 classes are evaluated using both global and per-class metrics:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-slate-400">
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
            <strong className="text-slate-200 block mb-1">Macro-Averaged F1-Score</strong>
            F1 evaluates the harmonic mean of precision and recall averaged equally across all 4 classes to penalize class-imbalance bias.
          </div>
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
            <strong className="text-slate-200 block mb-1">Standardized Train/Val/Test Split</strong>
            70% Training, 15% Validation, and 15% Testing with identical seed (42) guarantees fair comparison across all 12 combinations.
          </div>
        </div>
      </div>

      {/* Section 4: Limitations and Scope */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
        <h3 className="text-sm font-bold text-amber-400 uppercase tracking-wide">
          4. Academic Limitations & Future Scope
        </h3>
        <ul className="text-xs text-slate-300 space-y-2 list-disc list-inside leading-relaxed">
          <li><strong>Discretization Loss:</strong> Mapping continuous monthly ages into 4 broad bins simplifies classification benchmark comparisons but loses fine-grained monthly regression resolution.</li>
          <li><strong>Demographic Variance:</strong> The RSNA dataset primarily represents North American pediatric cohorts; cross-ethnicity generalization requires further validation.</li>
          <li><strong>Future Scope:</strong> Extending the 12-cell matrix with bone segmentation (e.g. U-Net carpal isolation) and dual-branch networks combining radiograph features with sex/gender tokens.</li>
        </ul>
      </div>
    </div>
  );
}
