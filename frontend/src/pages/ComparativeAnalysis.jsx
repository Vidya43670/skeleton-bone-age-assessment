import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  AlertCircle,
  RefreshCw,
  CheckCircle2,
} from 'lucide-react';
import api from '../services/api';
import AcademicDisclaimer from '../components/AcademicDisclaimer';

const CLASS_LABELS = [
  'Class 0 (0–48 mo)',
  'Class 1 (49–96 mo)',
  'Class 2 (97–144 mo)',
  'Class 3 (145–216 mo)',
];

function ConfusionMatrix({ matrix, model, preprocessing }) {
  if (!matrix || matrix.length !== 4) {
    return null;
  }

  const maxValue = Math.max(...matrix.flat());

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
      <div className="mb-4">
        <h4 className="text-sm font-bold text-white">
          {model} + {preprocessing}
        </h4>

        <p className="text-[11px] text-slate-500 mt-1">
          Rows = Actual Class | Columns = Predicted Class
        </p>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[520px]">
          <div className="text-center text-[10px] text-slate-500 mb-2">
            Predicted Class
          </div>

          <div className="grid grid-cols-[170px_repeat(4,1fr)] gap-1">
            <div></div>

            {CLASS_LABELS.map((label, index) => (
              <div
                key={index}
                className="text-[9px] text-slate-400 text-center p-2"
              >
                C{index}
              </div>
            ))}

            {matrix.map((row, actualIndex) => (
              <React.Fragment key={actualIndex}>
                <div className="text-[9px] text-slate-400 flex items-center justify-end pr-3">
                  {CLASS_LABELS[actualIndex]}
                </div>

                {row.map((value, predictedIndex) => {
                  const intensity =
                    maxValue > 0 ? value / maxValue : 0;

                  return (
                    <div
                      key={predictedIndex}
                      className="h-14 rounded-md flex items-center justify-center border border-slate-800"
                      style={{
                        backgroundColor: `rgba(45, 212, 191, ${
                          0.08 + intensity * 0.65
                        })`,
                      }}
                    >
                      <span className="text-xs font-bold text-white">
                        {value}
                      </span>
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>

          <div className="text-center text-[10px] text-slate-500 mt-3">
            Actual Class
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ComparativeAnalysis() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadResults = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await api.getComparisonResults();
      setResults(res.data);
    } catch (err) {
      setError(err.response?.data?.message || err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadResults();
  }, []);

  const completedExperiments = results.filter(
    (r) => r.status === 'COMPLETED'
  );

  return (
    <div className="space-y-6">
      

      {/* HEADER */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            Comparative Experimental Analysis

            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-brand-300 border border-brand-500/20">
              Empirical Benchmarks
            </span>
          </h2>

          <p className="text-xs text-slate-400">
            Statistical comparison of Accuracy, Macro F1-scores,
            and preprocessing impact.
          </p>
        </div>

        <button
          onClick={loadResults}
          disabled={loading}
          className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 flex items-center space-x-1.5 transition-colors"
        >
          <RefreshCw
            className={`w-3.5 h-3.5 ${
              loading ? 'animate-spin' : ''
            }`}
          />

          <span>Refresh Results</span>
        </button>
      </div>

      {/* ERROR */}
      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-red-300 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* LOADING */}
      {loading && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center">
          <RefreshCw className="w-6 h-6 animate-spin text-brand-400 mx-auto mb-3" />

          <p className="text-xs text-slate-400">
            Loading experimental results...
          </p>
        </div>
      )}

      {/* NO RESULTS */}
      {!loading && completedExperiments.length === 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center space-y-4">
          <BarChart3 className="w-8 h-8 text-brand-400 mx-auto" />

          <div>
            <h3 className="text-base font-bold text-white">
              No Experimental Runs Completed Yet
            </h3>

            <p className="text-xs text-slate-400 mt-2">
              No completed experimental results are currently available.
            </p>
          </div>
        </div>
      )}

      {/* RESULTS */}
      {!loading && completedExperiments.length > 0 && (
        <>
          {/* COMPLETION SUMMARY */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />

              <div>
                <p className="text-sm font-bold text-white">
                  {completedExperiments.length} / 12 Experiments Completed
                </p>

                <p className="text-[11px] text-slate-500">
                  Results are loaded from the project SQLite database.
                </p>
              </div>
            </div>
          </div>

          {/* COMPARISON TABLE */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-slate-800">
              <h3 className="text-sm font-bold text-white">
                Comparative Performance
              </h3>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950/60 border-b border-slate-800 text-slate-400">
                  <tr>
                    <th className="p-3 font-semibold">
                      Preprocessing
                    </th>

                    <th className="p-3 font-semibold">
                      CNN Model
                    </th>

                    <th className="p-3 font-semibold">
                      Mode
                    </th>

                    <th className="p-3 font-semibold">
                      Val Acc (%)
                    </th>

                    <th className="p-3 font-semibold">
                      Test Acc (%)
                    </th>

                    <th className="p-3 font-semibold">
                      Macro F1 (%)
                    </th>

                    <th className="p-3 font-semibold">
                      Macro Precision (%)
                    </th>

                    <th className="p-3 font-semibold">
                      Macro Recall (%)
                    </th>
                  </tr>
                </thead>

                <tbody className="divide-y divide-slate-800 text-slate-300">
                  {completedExperiments.map((row) => (
                    <tr
                      key={row.id}
                      className="hover:bg-slate-800/40"
                    >
                      <td className="p-3 font-bold text-brand-400">
                        {row.preprocessing}
                      </td>

                      <td className="p-3 font-semibold text-white">
                        {row.model}
                      </td>

                      <td className="p-3">
                        <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px]">
                          {row.is_smoke_test
                            ? 'Smoke Test'
                            : 'Full'}
                        </span>
                      </td>

                      <td className="p-3 font-mono">
                        {row.val_accuracy != null
                          ? `${row.val_accuracy.toFixed(2)}`
                          : '—'}
                      </td>

                      <td className="p-3 font-mono font-bold text-emerald-400">
                        {row.test_accuracy != null
                          ? `${row.test_accuracy.toFixed(2)}`
                          : '—'}
                      </td>

                      <td className="p-3 font-mono font-bold text-brand-400">
                        {row.macro_f1 != null
                          ? `${(
                              row.macro_f1 * 100
                            ).toFixed(2)}`
                          : '—'}
                      </td>

                      <td className="p-3 font-mono">
                        {row.macro_precision != null
                          ? `${(
                              row.macro_precision * 100
                            ).toFixed(2)}`
                          : '—'}
                      </td>

                      <td className="p-3 font-mono">
                        {row.macro_recall != null
                          ? `${(
                              row.macro_recall * 100
                            ).toFixed(2)}`
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* CONFUSION MATRICES */}
          <div>
            <div className="mb-4">
              <h3 className="text-base font-bold text-white">
                Confusion Matrices
              </h3>

              <p className="text-xs text-slate-500 mt-1">
                Actual test-set predictions for each completed experiment.
              </p>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
              {completedExperiments.map((row) => (
                <ConfusionMatrix
                  key={row.id}
                  matrix={row.confusion_matrix}
                  model={row.model}
                  preprocessing={row.preprocessing}
                />
              ))}
            </div>
          </div>
        </>
      )}

      {/* METRICS GUIDE */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <h3 className="text-sm font-bold text-white mb-2">
          Evaluation Metrics Standard
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-400">
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="font-semibold text-slate-200 block mb-1">
              Macro F1-Score
            </span>

            <span>
              Unweighted mean of F1-scores across all 4 classes,
              giving each age class equal importance.
            </span>
          </div>

          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="font-semibold text-slate-200 block mb-1">
              4×4 Confusion Matrix
            </span>

            <span>
              Shows actual classes against predicted classes and
              makes class-wise errors visible.
            </span>
          </div>

          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
            <span className="font-semibold text-slate-200 block mb-1">
              Test Evaluation
            </span>

            <span>
              Metrics shown here come from the held-out test set
              containing 1,889 samples.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}