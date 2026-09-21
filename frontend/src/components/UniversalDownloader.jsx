import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import TokenManager from "./TokenManager";

export default function UniversalDownloader({
  engineBase,
  onLoraDownloaded,
  onSwitchModel,
}) {
  const [downloadInput, setDownloadInput] = useState("");
  const [downloadName, setDownloadName] = useState("");
  const [downloadTriggers, setDownloadTriggers] = useState("");
  const [downloadBaseModel, setDownloadBaseModel] = useState("");
  const [downloading, setDownloading] = useState(false);
  const [downloadFeedback, setDownloadFeedback] = useState(null);

  const [activeDownloads, setActiveDownloads] = useState([]);
  const completedTaskIdsRef = useRef(new Set());

  const [tokenAutofocus, setTokenAutofocus] = useState(null);

  useEffect(() => {
    api("/api/loras/downloads")
      .then((list) => {
        const arr = Array.isArray(list) ? list : [];
        setActiveDownloads(arr);
        for (const d of arr) {
          if (["done", "error", "cancelled"].includes(d.status)) {
            completedTaskIdsRef.current.add(d.id);
          }
        }
      })
      .catch(() => {});
  }, []);

  // Poll active downloads
  useEffect(() => {
    let interval = null;
    const hasPending = activeDownloads.some(
      (d) => d.status === "downloading" || d.status === "pending"
    );

    if (hasPending) {
      interval = setInterval(async () => {
        try {
          const list = await api("/api/loras/downloads");
          setActiveDownloads(list);

          let justFinished = false;
          for (const item of list) {
            if (item.status === "done" && !completedTaskIdsRef.current.has(item.id)) {
              completedTaskIdsRef.current.add(item.id);
              justFinished = true;
            }
          }

          if (justFinished) {
            onLoraDownloaded?.();
          }
        } catch {
          /* ignore transient polling errors */
        }
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [activeDownloads, onLoraDownloaded]);

  const detectedSource = useMemo(() => {
    const v = downloadInput.trim().toLowerCase();
    if (!v) return null;
    if (
      v.includes("huggingface.co") ||
      v.includes("hf.co") ||
      (/^[a-z0-9_-]+\/[a-z0-9._-]+$/i.test(v) && !v.includes(" ") && isNaN(Number(v)))
    ) {
      return "hf";
    }
    if (
      v.includes("civitai.com") ||
      v.includes("civitai.red") ||
      (/^\d+$/.test(v) && v.length <= 10)
    ) {
      return "civitai";
    }
    if (v.startsWith("http://") || v.startsWith("https://")) {
      return "direct";
    }
    return null;
  }, [downloadInput]);

  async function handleDownload() {
    if (!downloadInput.trim()) return;
    setDownloading(true);
    setDownloadFeedback(null);
    try {
      const payload = {
        url_or_id: downloadInput.trim(),
      };
      if (downloadName.trim()) payload.name = downloadName.trim();
      if (downloadTriggers.trim()) {
        payload.triggers = downloadTriggers.split(",").map((t) => t.trim()).filter(Boolean);
      }
      if (downloadBaseModel.trim()) {
        payload.base_model = downloadBaseModel.trim();
      }

      const res = await api("/api/loras/download", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      setDownloadInput("");
      setDownloadName("");
      setDownloadTriggers("");
      setDownloadFeedback({
        type: "info",
        text: `Starting download of "${res.model_name}" (${res.base_model?.toUpperCase() || "LoRA"})…`,
      });
      const list = await api("/api/loras/downloads");
      setActiveDownloads(list);

      const doneNow = list.filter(
        (d) => d.status === "done" && !completedTaskIdsRef.current.has(d.id)
      );
      for (const d of doneNow) {
        completedTaskIdsRef.current.add(d.id);
      }
      if (doneNow.length > 0) {
        onLoraDownloaded?.();
      }
    } catch (e) {
      setDownloadFeedback({ type: "error", text: `Download failed: ${e.message}` });
      const msg = String(e.message || "");
      if (
        msg.includes("authentication") ||
        msg.includes("download-auth") ||
        msg.includes("401") ||
        msg.includes("AUTH_REQUIRED") ||
        msg.includes("HF_TOKEN_REQUIRED")
      ) {
        if (downloadInput.includes("huggingface.co") || downloadInput.includes("hf.co")) {
          setTokenAutofocus("hf");
        } else {
          setTokenAutofocus("civitai");
        }
      }
    } finally {
      setDownloading(false);
    }
  }

  async function handleCancelDownload(taskId) {
    try {
      await api(`/api/loras/downloads/${taskId}`, { method: "DELETE" });
      const updated = await api("/api/loras/downloads");
      setActiveDownloads(updated);
    } catch (e) {
      setDownloadFeedback({ type: "error", text: `Failed to cancel: ${e.message}` });
    }
  }

  return (
    <div className="universal-downloader-wrapper">
      <div className="universal-download-card">
        <div className="universal-download-header">
          <div className="universal-badges">
            <span className={`civitai-tag ${detectedSource === "civitai" ? "active-source" : detectedSource ? "dimmed" : ""}`}>
              CIVITAI
            </span>
            <span className={`hf-tag ${detectedSource === "hf" ? "active-source" : detectedSource ? "dimmed" : ""}`}>
              HUGGING FACE
            </span>
            <span className={`direct-tag ${detectedSource === "direct" ? "active-source" : detectedSource ? "dimmed" : ""}`}>
              DIRECT URL
            </span>
          </div>
          <span className="universal-card-title">Universal LoRA Downloader</span>
          {detectedSource === "hf" && (
            <span className="hub-detected-badge hf-detected">🤗 Hugging Face detected</span>
          )}
          {detectedSource === "civitai" && (
            <span className="hub-detected-badge civitai-detected">⚡ Civitai detected</span>
          )}
          {detectedSource === "direct" && (
            <span className="hub-detected-badge direct-detected">🔗 Direct Link detected</span>
          )}
        </div>

        <p className="universal-download-hint">
          Paste a Civitai model URL or ID, a Hugging Face repo ID or URL, or any direct link to a <code>.safetensors</code> file.
        </p>

        <div className="universal-input-row">
          <input
            placeholder="Civitai URL or ID, Hugging Face repo or URL, or direct .safetensors link..."
            value={downloadInput}
            onChange={(e) => setDownloadInput(e.target.value)}
            disabled={downloading}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleDownload();
              }
            }}
          />
          <button
            type="button"
            className="btn-universal-download"
            onClick={handleDownload}
            disabled={downloading || !downloadInput.trim()}
          >
            {downloading ? "Starting…" : "⚡ Download & Register"}
          </button>
        </div>

        <div className="universal-options-row">
          <input
            placeholder="Custom name (optional)"
            value={downloadName}
            onChange={(e) => setDownloadName(e.target.value)}
            disabled={downloading}
          />
          <input
            placeholder="Trigger words: comma, separated (optional)"
            value={downloadTriggers}
            onChange={(e) => setDownloadTriggers(e.target.value)}
            disabled={downloading}
          />
          <select
            value={downloadBaseModel}
            onChange={(e) => setDownloadBaseModel(e.target.value)}
            disabled={downloading}
          >
            <option value="">Auto-Detect Architecture</option>
            <option value="sdxl">SDXL</option>
            <option value="flux2">FLUX.2</option>
            <option value="krea2">Krea-2</option>
            <option value="z-image">Z-Image</option>
          </select>
        </div>

        <TokenManager autofocus={tokenAutofocus} className="universal-tokens-bar" />

        {downloadFeedback && (
          <p className={`civitai-feedback ${downloadFeedback.type}`}>
            {downloadFeedback.text}
          </p>
        )}
      </div>

      {/* ACTIVE DOWNLOADS CONTAINER (Shared across Civitai, HF & Direct URL) */}
      {activeDownloads.length > 0 && (
        <div className="active-downloads-container">
          {activeDownloads.map((dl) => (
            <div key={dl.id} className={`civitai-download-card ${dl.status}`}>
              <div className="civitai-download-header">
                <span className="civitai-download-name">
                  {(dl.source === "hf" || dl.source === "huggingface") && <span className="hf-tag" style={{ marginRight: 6 }}>HF</span>}
                  {dl.source === "direct_url" && <span className="direct-tag" style={{ marginRight: 6 }}>DIRECT</span>}
                  {(!dl.source || dl.source === "civitai") && <span className="civitai-tag" style={{ marginRight: 6 }}>CIVITAI</span>}
                  <strong>{dl.model_name}</strong>
                  <span className={`lora-badge lora-badge-${dl.base_model}`}>{dl.base_model?.toUpperCase()}</span>
                </span>
                <div className="civitai-download-actions">
                  {dl.status === "downloading" && (
                    <button
                      type="button"
                      className="btn-cancel-download"
                      onClick={() => handleCancelDownload(dl.id)}
                      title="Cancel download"
                    >
                      ✕ Cancel
                    </button>
                  )}
                  {dl.status === "done" && <span className="dl-tag done">✓ Ready</span>}
                  {dl.status === "error" && <span className="dl-tag error">✖ Error</span>}
                  {dl.status === "cancelled" && <span className="dl-tag cancelled">Cancelled</span>}
                </div>
              </div>
              <div className="civitai-progress-track">
                <div
                  className={`civitai-progress-fill ${dl.status}`}
                  style={{ width: `${Math.round((dl.progress || 0) * 100)}%` }}
                />
              </div>
              <div className="civitai-download-footer">
                <span className="civitai-dl-status-text">{dl.status_text || dl.error}</span>
                {dl.status === "downloading" && dl.speed_mb_s > 0 && (
                  <span className="civitai-dl-speed">{dl.speed_mb_s} MB/s</span>
                )}
              </div>
              {dl.status === "error" && dl.error && dl.error.includes("AUTH_REQUIRED") && (
                <div className="civitai-auth-prompt">
                  <span>This model requires authentication on Civitai.</span>
                  <button
                    type="button"
                    className="btn-open-token"
                    onClick={() => setTokenAutofocus("civitai")}
                  >
                    🔑 Configure API Key
                  </button>
                </div>
              )}
              {dl.status === "error" && dl.error && dl.error.includes("HF_TOKEN_REQUIRED") && (
                <div className="civitai-auth-prompt">
                  <span>This model requires authentication on Hugging Face.</span>
                  <button
                    type="button"
                    className="btn-open-token"
                    onClick={() => setTokenAutofocus("hf")}
                  >
                    🔑 Configure HF Token
                  </button>
                </div>
              )}
              {dl.status === "done" && engineBase && dl.base_model !== engineBase && (
                <div className="civitai-switch-prompt">
                  <span>Installed for {dl.base_model?.toUpperCase()}.</span>
                  <button
                    type="button"
                    className="btn-switch-prompt"
                    onClick={() => onSwitchModel?.(dl.base_model)}
                  >
                    ⚡ Switch to {dl.base_model?.toUpperCase()}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
