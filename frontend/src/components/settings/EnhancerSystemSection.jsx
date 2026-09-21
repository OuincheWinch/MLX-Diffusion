import { useEffect, useMemo, useState } from "react";
import { api } from "../../api";

const API = "/api/prompt/enhancer/system-prompts";

// The editable box is pre-filled with the user's override if present, otherwise
// with the built-in instructions — so the model's system prompt is always
// visible and editable rather than hidden behind a "view built-in" toggle.
// Text mode shows the engine guidance; JSON mode shows the full JSON contract
// (engine guidance + schema structure + fill-in guidelines).
function draftFor(engine, mode) {
  if (mode === "json") {
    return (
      engine.custom_json_instructions || engine.default_json_instructions || ""
    );
  }
  return engine.custom_instructions || engine.default_instructions || "";
}

function isCustom(engine, mode) {
  return mode === "json" ? !!engine.is_json_custom : !!engine.is_custom;
}

export default function EnhancerSystemSection({ onFeedback, onSaved }) {
  const [engines, setEngines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeKey, setActiveKey] = useState(null);
  const [mode, setMode] = useState("text");
  const [drafts, setDrafts] = useState({});
  const [saving, setSaving] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [showDefault, setShowDefault] = useState(false);

  async function load() {
    setLoading(true);
    try {
      const data = await api(API);
      const list = data.engines || [];
      setEngines(list);
      const next = {};
      for (const e of list) {
        next[`text:${e.key}`] = draftFor(e, "text");
        next[`json:${e.key}`] = draftFor(e, "json");
      }
      setDrafts(next);
      setActiveKey((prev) => prev || list[0]?.key || null);
      setError(null);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  const active = useMemo(
    () => engines.find((e) => e.key === activeKey) || null,
    [engines, activeKey]
  );

  const draftKey = active ? `${mode}:${active.key}` : "";
  const draft = draftKey ? (drafts[draftKey] ?? "") : "";
  const baseline = active ? draftFor(active, mode) : "";
  const dirty = active ? draft.trim() !== baseline.trim() : false;
  const activeCustom = active ? isCustom(active, mode) : false;
  const modeLabel = mode === "json" ? "JSON" : "text";

  async function save(instructions) {
    if (!active) return;
    setSaving(true);
    try {
      const data = await api(API, {
        method: "POST",
        body: JSON.stringify({ engine_key: active.key, mode, instructions }),
      });
      const list = data.engines || [];
      setEngines(list);
      const updated = list.find((e) => e.key === active.key);
      setDrafts((prev) => ({
        ...prev,
        [`${mode}:${active.key}`]: updated ? draftFor(updated, mode) : "",
      }));
      onFeedback?.({ type: "success", text: "Enhancer system prompt saved" });
      onSaved?.();
    } catch (e) {
      onFeedback?.({ type: "error", text: e.message || String(e) });
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return <p className="hint">Loading prompt-enhancer profiles…</p>;
  }
  if (error) {
    return (
      <p className="error" role="alert">
        {error}
      </p>
    );
  }
  if (!active) {
    return <p className="hint">No engines available.</p>;
  }

  return (
    <div className="enhancer-section">
      <div className="enhancer-subtabs" role="tablist">
        {engines.map((e) => (
          <button
            key={e.key}
            type="button"
            role="tab"
            aria-selected={e.key === activeKey}
            className={`enhancer-subtab${e.key === activeKey ? " active" : ""}`}
            onClick={() => {
              setActiveKey(e.key);
              setShowPreview(false);
              setShowDefault(false);
            }}
          >
            {e.label.split(" (")[0]}
            {(e.is_custom || e.is_json_custom) && (
              <span className="enhancer-dot" title="Custom prompt active" />
            )}
          </button>
        ))}
      </div>

      <div className="enhancer-meta">
        <span className="settings-badge">{active.label}</span>
        <span className="settings-hint">target length: {active.length}</span>
        {activeCustom ? (
          <span className="settings-badge ok">custom {modeLabel}</span>
        ) : (
          <span className="settings-badge">built-in {modeLabel}</span>
        )}
      </div>

      <div className="enhancer-mode-toggle" role="tablist" aria-label="Enhancer output mode">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "text"}
          className={`btn-mini${mode === "text" ? " active" : ""}`}
          onClick={() => {
            setMode("text");
            setShowPreview(false);
          }}
        >
          📝 Text mode
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "json"}
          className={`btn-mini${mode === "json" ? " active" : ""}`}
          onClick={() => {
            setMode("json");
            setShowPreview(false);
          }}
        >
          {"{ }"} JSON mode
        </button>
      </div>

      <label className="settings-field">
        <span className="settings-field-title">
          {mode === "json"
            ? "JSON system prompt — structure + fill-in guidelines"
            : "System prompt — engine guidance (text mode)"}
        </span>
        <span className="settings-hint">
          {mode === "json"
            ? "This is the complete instruction sent to the local LLM in JSON mode: the engine guidance, the exact JSON structure (keys) to output, and the rules for filling each field. Edit it and Save to store a custom override, or ↺ Reset to built-in to restore the default."
            : `This text is injected into the local LLM's system prompt when you click ✨ Enhance with an ${active.label.split(" (")[0]} model in text mode. Edit it and Save to store a custom override, or ↺ Reset to built-in to restore the default.`}{" "}
          Text and JSON modes are stored independently.
        </span>
        <textarea
          className="enhancer-textarea"
          rows={12}
          value={draft}
          onChange={(ev) =>
            setDrafts((prev) => ({ ...prev, [draftKey]: ev.target.value }))
          }
          spellCheck={false}
        />
      </label>

      <div className="settings-actions">
        <button
          type="button"
          className={`btn-mini enhancer-save-btn${dirty ? " dirty" : ""}`}
          disabled={saving || !dirty}
          onClick={() => save(draft)}
          title={dirty ? "Save this custom guidance" : "Edit the text to enable saving"}
        >
          {saving ? "Saving…" : "💾 Save"}
        </button>
        <button
          type="button"
          className="btn-mini"
          disabled={saving || !activeCustom}
          onClick={() => save("")}
          title="Restore the built-in engine guidance"
        >
          ↺ Reset to built-in
        </button>
        <button
          type="button"
          className={`btn-mini${showDefault ? " active" : ""}`}
          onClick={() => setShowDefault((v) => !v)}
        >
          {showDefault ? "Hide built-in" : "View built-in"}
        </button>
        <button
          type="button"
          className={`btn-mini${showPreview ? " active" : ""}`}
          onClick={() => setShowPreview((v) => !v)}
        >
          {showPreview ? "Hide preview" : "Preview full prompt"}
        </button>
      </div>

      {showDefault && (
        <div className="enhancer-readonly">
          <span className="settings-field-title">
            {mode === "json"
              ? "Built-in JSON instructions (structure + guidelines)"
              : "Built-in engine guidance (text mode)"}
          </span>
          <pre>
            {mode === "json"
              ? active.default_json_instructions
              : active.default_instructions}
          </pre>
        </div>
      )}

      {showPreview && (
        <div className="enhancer-readonly">
          <span className="settings-field-title">
            Full system prompt sent to the LLM ({modeLabel} mode, no active LoRA triggers)
          </span>
          <pre>{mode === "json" ? active.json_system_prompt : active.system_prompt}</pre>
          <span className="settings-hint">
            {mode === "json"
              ? "JSON mode appends the mandatory LoRA-trigger block (when triggers are active) after your instructions."
              : "Text mode appends the prose TASK and general prompt-engineering rules after your guidance."}
          </span>
        </div>
      )}

      <p className="params-hint">
        These prompts drive the local prompt enhancer (Qwen2.5-0.5B-Instruct, MLX). Each generated
        model is mapped to one of these four engines: FLUX.2 Klein / SDXL / Krea 2 / Z-Image Turbo.
      </p>
    </div>
  );
}