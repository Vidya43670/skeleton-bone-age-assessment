import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function AcademicDisclaimer() {
  return (
    <div className="bg-amber-950/40 border border-amber-500/40 rounded-xl p-4 flex items-start space-x-3 text-amber-200 text-sm mb-6 shadow-sm">
      <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
      <div>
        <span className="font-semibold text-amber-300 uppercase tracking-wide text-xs block mb-0.5">
          Academic Research Prototype Only — Not for Clinical Use
        </span>
        <p className="text-amber-200/90 leading-relaxed text-xs">
          This system is developed strictly for university academic research and empirical comparison under the RSNA Pediatric Bone Age Challenge dataset. It does NOT constitute medical advice, is NOT clinically validated, and MUST NOT be used for diagnostic or treatment decisions.
        </p>
      </div>
    </div>
  );
}
