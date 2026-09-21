import { useState, memo } from "react";
import { api, fetchImageBlob, imageUrl } from "../api";
import { bindFullImageDrag } from "../utils/dragDrop";

function CanvasProgressOverlay({ busy, progress, standalone = false, generatingPrompt = null, phase = null, phaseDetail = null }) {
  if (!busy) return null;
  const isProgress = progress && progress.steps > 0;
  const batchTotal = progress?.batch || 1;
  const batchIdx = (progress?.image_index != null ? progress.image_index : progress?.saved_index ? progress.saved_index - 1 : 0) + 1;
  const isBatch = batchTotal > 1;

  let mainText = "Preparing generation…";
  if (isProgress) {
    if (phase === "saving" || progress.step >= progress.steps) {
      mainText = `🎨 Decoding VAE (${Math.round(progress.elapsed || 0)}s)…`;
    } else {
      mainText = `Step ${progress.step}/${progress.steps} (${Math.round(progress.elapsed || 0)}s)`;
    }
  } else {
    if (phase === "downloading") {
      mainText = "📥 Downloading model weights…";
    } else if (phase === "loading_model") {
      mainText = "🧠 Loading model into unified memory…";
    } else if (phase === "compiling") {
      mainText = "⚡ Compiling Metal shaders & encoding prompt…";
    } else if (phase === "saving") {
      mainText = "🎨 Finalizing image & saving…";
    } else if (standalone) {
      mainText = "Starting engine & loading model…";
    }
  }

  return (
    <div className={standalone ? "canvas-busy-standalone" : "canvas-busy-overlay"}>
      <div className="spinner" />
      {isBatch && (
        <span className="canvas-batch-pill">
          Image {batchIdx} / {batchTotal}
        </span>
      )}
      <p>{mainText}</p>
      {phaseDetail && !isProgress && (
        <p className="canvas-phase-detail">{phaseDetail}</p>
      )}
      {isProgress && (
        <div className="canvas-progress-track">
          <div
            className="progress-fill"
            style={{
              width: `${Math.min(100, (progress.step / progress.steps) * 100)}%`,
            }}
          />
        </div>
      )}
      {generatingPrompt && (
        <p className="canvas-busy-prompt" title={generatingPrompt}>
          &ldquo;{generatingPrompt}&rdquo;
        </p>
      )}
    </div>
  );
}

function ResultCanvas({
  currentImage,
  onSetCurrentImage,
  busy,
  progress,
  phase = null,
  phaseDetail = null,
  batchImages = [],
  generatingPrompt = null,
  onSetReferenceImage,
  onVariation,
}) {
  const [upscaling, setUpscaling] = useState(null); // null | "lanczos-2" | "lanczos-4"
  const [copiedField, setCopiedField] = useState(null); // null | "image" | "prompt" | "seed"

  async function handleUpscale(scale = 2) {
    if (!currentImage?.id) return;
    const key = `lanczos-${scale}`;
    setUpscaling(key);
    try {
      const upscaled = await api(`/api/images/${currentImage.id}/upscale`, {
        method: "POST",
        body: JSON.stringify({ scale }),
      });
      onSetCurrentImage?.(upscaled);
    } catch (e) {
      console.error("Upscale failed:", e);
    } finally {
      setUpscaling(null);
    }
  }

  async function copyImageToClipboard() {
    if (!currentImage?.id) return;
    try {
      const blob = await fetchImageBlob(currentImage.id);
      await navigator.clipboard.write([
        new ClipboardItem({ [blob.type || "image/png"]: blob }),
      ]);
      setCopiedField("image");
      setTimeout(() => setCopiedField(null), 2000);
    } catch {
      window.open(imageUrl(currentImage.id), "_blank");
    }
  }

  async function copyPrompt() {
    if (!currentImage?.prompt) return;
    try {
      await navigator.clipboard.writeText(currentImage.prompt);
      setCopiedField("prompt");
      setTimeout(() => setCopiedField(null), 1500);
    } catch {}
  }

  async function copySeed() {
    if (currentImage?.seed == null) return;
    try {
      await navigator.clipboard.writeText(String(currentImage.seed));
      setCopiedField("seed");
      setTimeout(() => setCopiedField(null), 1500);
    } catch {}
  }

  return (
    <div className="result-canvas-panel">
      <div className="canvas-header">
        <h3>Studio Canvas</h3>
        {busy ? (
          <span className={`canvas-meta-pill busy-pill ${phase ? `phase-${phase}` : ""}`}>
            {phase === "downloading"
              ? "📥 Downloading Model"
              : phase === "loading_model"
                ? "🧠 Loading into Memory"
                : phase === "compiling"
                  ? "⚡ Compiling Shaders"
                  : phase === "saving"
                    ? "🎨 Finalizing Image"
                    : progress?.steps > 0
                      ? `⚙ Step ${progress.step}/${progress.steps}`
                      : "⚙ Working…"}
          </span>
        ) : currentImage ? (
          <span className="canvas-meta-pill">
            {currentImage.width}×{currentImage.height} · {currentImage.generation_time}s
          </span>
        ) : null}
      </div>

      <div className="canvas-viewport">
        {currentImage ? (
          <div className="canvas-image-container">
            <img
              key={currentImage.id}
              src={imageUrl(currentImage.id)}
              alt={currentImage.prompt || "Generated image"}
              className="canvas-image"
              title="Drag for full-resolution image"
              {...bindFullImageDrag(currentImage)}
            />
            <CanvasProgressOverlay
              busy={busy}
              progress={progress}
              phase={phase}
              phaseDetail={phaseDetail}
              generatingPrompt={generatingPrompt}
            />
          </div>
        ) : (
          <div className="canvas-empty">
            {busy ? (
              <CanvasProgressOverlay
                busy={busy}
                progress={progress}
                standalone
                phase={phase}
                phaseDetail={phaseDetail}
                generatingPrompt={generatingPrompt}
              />
            ) : (
              <>
                <div className="empty-icon">🎨</div>
                <p className="empty-title">Your creation will appear here</p>
                <p className="empty-subtitle">
                  Enter a prompt and click Generate to start rendering
                </p>
              </>
            )}
          </div>
        )}
      </div>

      {/* Batch Filmstrip / Gallery */}
      {batchImages && batchImages.length > 0 && (
        <div className="canvas-batch-strip">
          <div className="batch-strip-header">
            <span className="batch-strip-title">
              Batch Gallery ({batchImages.length}{progress?.batch && progress.batch > batchImages.length ? `/${progress.batch}` : ""})
            </span>
            {busy && progress?.batch && batchImages.length < progress.batch && (
              <span className="batch-strip-status">
                Rendering #{batchImages.length + 1}...
              </span>
            )}
          </div>
          <div className="batch-strip-items">
            {batchImages.map((img, idx) => (
              <div
                key={img.id || idx}
                role="button"
                tabIndex={0}
                className={`batch-strip-thumb ${currentImage?.id === img.id ? "active" : ""}`}
                onClick={() => onSetCurrentImage?.(img)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSetCurrentImage?.(img);
                  }
                }}
                title={`Image #${idx + 1} (Seed: ${img.seed}) - Click to preview · Drag for full-resolution image`}
                {...bindFullImageDrag(img)}
              >
                <img
                  src={imageUrl(img.id, true)}
                  alt={`Batch #${idx + 1}`}
                  {...bindFullImageDrag(img)}
                />
                <span className="batch-thumb-badge">#{idx + 1}</span>
              </div>
            ))}
            {busy && progress?.batch && batchImages.length < progress.batch && (
              <div className="batch-strip-thumb placeholder" title={`Rendering image #${batchImages.length + 1}...`}>
                <div className="mini-spinner" />
                <span className="batch-thumb-badge">#{batchImages.length + 1}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {currentImage && (
        <div className={`canvas-details ${busy ? "canvas-details-prev" : ""}`}>
          {busy && (
            <div className="canvas-prev-note">
              <span className="canvas-prev-tag">Previous Result</span>
              <span className="canvas-prev-hint">Showing last completed creation while new image renders</span>
            </div>
          )}
          <p className="canvas-prompt" title={currentImage.prompt}>
            &ldquo;{currentImage.prompt}&rdquo;
          </p>

          <div className="canvas-badges">
            <span className="badge">
              <strong>Model:</strong> {currentImage.model?.split("/").pop()}
            </span>
            <span className="badge">
              <strong>Seed:</strong> {currentImage.seed}
            </span>
            <span className="badge">
              <strong>Steps:</strong> {currentImage.steps}
            </span>
            {currentImage.sampler && (
              <span className="badge">
                <strong>Sampler:</strong> {currentImage.sampler}
              </span>
            )}
            {currentImage.guidance != null && (
              <span className="badge">
                <strong>CFG:</strong> {currentImage.guidance}
              </span>
            )}
            {currentImage.fast_vae && (
              <span className="badge">
                <strong>VAE:</strong> SOTA ⚡
              </span>
            )}
          </div>

          <div className="canvas-actions">
            <button
              className="action-btn"
              onClick={() => handleUpscale(2)}
              disabled={upscaling != null || busy}
              title="Fast Resampling 2x (Lanczos + Unsharp)"
            >
              {upscaling === "lanczos-2" ? "Fast 2x…" : "⚡ Fast 2x"}
            </button>
            <button
              className="action-btn"
              onClick={() => handleUpscale(4)}
              disabled={upscaling != null || busy}
              title="Fast Resampling 4x (Lanczos + Unsharp)"
            >
              {upscaling === "lanczos-4" ? "Fast 4x…" : "⚡ Fast 4x"}
            </button>
            <button
              className="action-btn"
              onClick={() => {
                const filename = currentImage.file || (currentImage.format ? `${currentImage.id}.${currentImage.format.toLowerCase()}` : `${currentImage.id}.png`);
                onSetReferenceImage?.({
                  path: filename,
                  preview: imageUrl(currentImage.id),
                });
              }}
              title="Use this image as Img2Img reference"
            >
              🖼️ Use as Reference
            </button>
            <button
              className="action-btn"
              onClick={() => onVariation?.(currentImage)}
              disabled={busy}
              title="Generate a variation with Seed + 1"
            >
              🔄 Variation
            </button>
            <button
              className="action-btn"
              onClick={copyImageToClipboard}
              title="Copy PNG pixels to system clipboard"
            >
              {copiedField === "image" ? "Image Copied ✓" : "📋 Copy Image"}
            </button>
            <button className="action-btn" onClick={copyPrompt} title="Copy Prompt">
              {copiedField === "prompt" ? "Prompt Copied ✓" : "📋 Copy Prompt"}
            </button>
            <button className="action-btn" onClick={copySeed} title="Copy Seed">
              {copiedField === "seed" ? "Seed Copied ✓" : "📋 Copy Seed"}
            </button>
            <a
              className="action-btn download-link"
              href={imageUrl(currentImage.id)}
              download={currentImage.file || `${currentImage.id}.${currentImage.format || 'png'}`}
              title={`Download full resolution ${currentImage.file?.endsWith('.jpeg') || currentImage.file?.endsWith('.jpg') || currentImage.format === 'jpeg' ? 'JPEG' : 'PNG'}`}
            >
              ⬇ Download
            </a>

          </div>
        </div>
      )}
    </div>
  );
}

export default memo(ResultCanvas);
