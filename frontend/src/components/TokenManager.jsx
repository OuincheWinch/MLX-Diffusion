import { useEffect, useState } from "react";
import { api } from "../api";

const CIVITAI_LABEL = "Civitai API Key";
const HF_LABEL = "Hugging Face Token";

function TokenRow({
  provider,
  label,
  placeholder,
  configured,
  showInput,
  value,
  setValue,
  onToggle,
  onSave,
  saving,
  feedback,
}) {
  return (
    <div className={`token-manager-row token-manager-${provider}`}>
      <div className="token-status-group">
        <span className="civitai-token-status">
          {configured ? "🔑 " : "🔓 "}{label}: {configured ? "Configured" : "Not set"}
        </span>
        <button type="button" className="btn-token-toggle" onClick={onToggle}>
          {showInput ? "Cancel" : configured ? "Edit" : `+ Set ${label}`}
        </button>
      </div>
      {showInput && (
        <div className="civitai-token-input-row">
          <input
            type="password"
            placeholder={placeholder}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            autoComplete="off"
          />
          <button
            type="button"
            className="btn-token-save"
            onClick={onSave}
            disabled={saving}
          >
            {saving ? "Saving…" : value.trim() ? `Save ${label}` : "Clear"}
          </button>
        </div>
      )}
      {feedback && (
        <p className={`civitai-feedback ${feedback.type}`}>{feedback.text}</p>
      )}
    </div>
  );
}

export default function TokenManager({
  autofocus = null,
  onTokenSaved = null,
  className = "",
}) {
  const [civitaiToken, setCivitaiToken] = useState("");
  const [civitaiConfigured, setCivitaiConfigured] = useState(false);
  const [showCivitai, setShowCivitai] = useState(false);
  const [hfToken, setHfToken] = useState("");
  const [hfConfigured, setHfConfigured] = useState(false);
  const [showHf, setShowHf] = useState(false);
  const [saving, setSaving] = useState(null);
  const [feedback, setFeedback] = useState(null);

  useEffect(() => {
    api("/api/civitai/token").then((r) => setCivitaiConfigured(Boolean(r.configured))).catch(() => {});
    api("/api/hf/token").then((r) => setHfConfigured(Boolean(r.configured))).catch(() => {});
  }, []);

  useEffect(() => {
    if (autofocus) {
      const timer = setTimeout(() => {
        if (autofocus === "civitai") setShowCivitai(true);
        if (autofocus === "hf") setShowHf(true);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [autofocus]);

  async function save(provider, label) {
    setSaving(provider);
    setFeedback(null);
    try {
      const value = provider === "civitai" ? civitaiToken : hfToken;
      const res = await api(provider === "civitai" ? "/api/civitai/token" : "/api/hf/token", {
        method: "POST",
        body: JSON.stringify({ token: value.trim() }),
      });
      if (provider === "civitai") {
        setCivitaiConfigured(Boolean(res.configured));
        setShowCivitai(false);
        setCivitaiToken("");
      } else {
        setHfConfigured(Boolean(res.configured));
        setShowHf(false);
        setHfToken("");
      }
      setFeedback({
        type: "success",
        text: res.configured ? `${label} saved securely (local file, never sent to clients).` : `${label} cleared.`,
      });
      onTokenSaved?.({ provider, configured: Boolean(res.configured) });
      setTimeout(() => setFeedback(null), 3500);
    } catch (e) {
      setFeedback({ type: "error", text: `Failed to save ${label}: ${e.message}` });
    } finally {
      setSaving(null);
    }
  }

  return (
    <div className={`token-manager ${className}`}>
      <TokenRow
        provider="civitai"
        label="Civitai"
        placeholder="Civitai API Key (from civitai.red/?ref_code=88C8VEBA)"
        configured={civitaiConfigured}
        showInput={showCivitai}
        value={civitaiToken}
        setValue={setCivitaiToken}
        onToggle={() => setShowCivitai((v) => !v)}
        onSave={() => save("civitai", CIVITAI_LABEL)}
        saving={saving === "civitai"}
        feedback={null}
      />
      <TokenRow
        provider="hf"
        label="Hugging Face"
        placeholder="Hugging Face Token (from huggingface.co/settings/tokens)"
        configured={hfConfigured}
        showInput={showHf}
        value={hfToken}
        setValue={setHfToken}
        onToggle={() => setShowHf((v) => !v)}
        onSave={() => save("hf", HF_LABEL)}
        saving={saving === "hf"}
        feedback={null}
      />
      {feedback && <p className={`civitai-feedback ${feedback.type}`}>{feedback.text}</p>}
    </div>
  );
}