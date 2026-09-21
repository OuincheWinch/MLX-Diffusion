import { useState } from "react";
import { api } from "../../api";
import ModelInstaller from "../ModelInstaller";
import { formatBytes } from "../../utils/formatBytes";

export default function ModelsSection({ models, loading, onModelsChanged, onFeedback }) {
  const [removing, setRemoving] = useState(null);
  const [err, setErr] = useState(null);

  async function removeModel(m) {
    if (!window.confirm(`Remove the weights of "${m.label}" from disk?\n\nYou can re-download it later from the Generate tab.`)) {
      return;
    }
    setRemoving(m.id);
    setErr(null);
    try {
      await api(`/api/models/${m.id}`, { method: "DELETE" });
      await onModelsChanged?.();
      onFeedback?.({ type: "success", text: `${m.label} removed from disk` });
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setRemoving(null);
    }
  }

  if (loading) {
    return <div className="settings-row"><span className="hint">Loading model registry…</span></div>;
  }

  return (
    <div className="settings-row">
      {err && <p className="error" role="alert">{err}</p>}
      <div className="models-table">
        <div className="models-table-head">
          <span>Model</span>
          <span>State</span>
          <span>On disk</span>
          <span>Actions</span>
        </div>
        {models.map((m) => {
          const installed = !!m.installed;
          const override = !!m.has_model_override;
          return (
            <div className="models-table-row" key={m.id}>
              <span className="models-name">
                <strong>{m.label}</strong>
                <span className="models-id">{m.id}</span>
              </span>
              <span className="models-state">
                <span className={`settings-badge ${installed ? "ok" : "missing"}`}>
                  {installed ? "✓ installed" : "not installed"}
                </span>
                {override && <span className="settings-badge">custom defaults</span>}
              </span>
              <span className="models-size">{formatBytes(m.disk_usage_bytes || 0)}</span>
              <span className="models-actions">
                {installed ? (
                  <button
                    type="button"
                    className="btn-mini"
                    disabled={removing === m.id}
                    onClick={() => removeModel(m)}
                    title="Delete weights from disk (guard: refuses while generating/downloading)"
                  >
                    {removing === m.id ? "Deleting…" : "🗑 Remove"}
                  </button>
                ) : (
                  <ModelInstaller modelInfo={m} onInstalled={onModelsChanged} />
                )}
              </span>
            </div>
          );
        })}
      </div>
      <p className="params-hint">
        Models are stored under <code>backend/data/models/</code> (SDXL),{" "}
        <code>backend/data/models/krea2-turbo-q4/</code> and the local Hugging Face hub cache
        (~/.cache/huggingface). Removing a model frees disk space; it is re-downloaded lazily on
        first use.
      </p>
    </div>
  );
}