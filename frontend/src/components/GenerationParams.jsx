import { memo } from "react";
import { STANDARD_SIZES } from "../constants/sizes";

function GenerationParams({
  modelInfo,
  width,
  setWidth,
  height,
  setHeight,
  steps,
  setSteps,
  guidance,
  setGuidance,
  seed,
  setSeed,
  batch,
  setBatch,
  negativePrompt,
  setNegativePrompt,
  sampler,
  setSampler,
  cacheInterval,
  setCacheInterval,
  supportsRef,
  supportsMultiRef,
  refImages,
  refStrength,
  setRefStrength,
  pickRefImages,
  removeRefImage,
  insertIntoPrompt,
}) {
  const maxRefImages = supportsMultiRef ? 10 : 1;

  function randomizeSeed() {
    setSeed("");
  }

  function incrementSeed(delta) {
    setSeed((prev) => {
      const cur = prev === "" || isNaN(Number(prev)) ? 0 : Number(prev);
      return Math.max(0, cur + delta);
    });
  }

  return (
    <div className="generation-params-container">
      <div className="param-grid">
        <label>
          Size
          <div className="size-row">
            <select
              value={`${width}x${height}`}
              onChange={(e) => {
                const [w, h] = e.target.value.split("x").map(Number);
                setWidth(w);
                setHeight(h);
              }}
            >
              {!STANDARD_SIZES.some((s) => s.value === `${width}x${height}`) && (
                <option value={`${width}x${height}`}>
                  ✦ {width} × {height} (Preset)
                </option>
              )}
              {STANDARD_SIZES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
            <span className="shape-preview" title={`${width} × ${height}`}>
              <span
                className="shape"
                style={(() => {
                  const max = 22;
                  const r = width / height;
                  return r >= 1
                    ? { width: `${max}px`, height: `${Math.max(4, Math.round(max / r))}px` }
                    : { width: `${Math.max(4, Math.round(max * r))}px`, height: `${max}px` };
                })()}
              />
            </span>
          </div>
        </label>

        <label>
          <span className="step-label-header">
            Steps ({steps})
            {modelInfo?.id === "krea2-turbo" && steps <= 4 && (
              <span className="distill-subtle-tag" title="4-step distillation LoRA auto-activated in parameters">
                ⚡ 4-step distill
              </span>
            )}
          </span>
          <input
            type="range"
            min="1"
            max="50"
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
          />
        </label>

        {modelInfo.supports_guidance && (
          <label>
            Guidance
            <input
              type="number"
              min="0"
              max="10"
              step="0.1"
              value={guidance}
              onChange={(e) => setGuidance(e.target.value)}
            />
          </label>
        )}

        <label>
          <div className="seed-label-row">
            <span>Seed</span>
            <div className="seed-quick-actions">
              <button
                type="button"
                className="btn-tiny"
                onClick={randomizeSeed}
                title="Randomize (empty seed)"
              >
                🎲
              </button>
              <button
                type="button"
                className="btn-tiny"
                onClick={() => incrementSeed(1)}
                title="Next Seed (+1)"
              >
                +1
              </button>
              <button
                type="button"
                className="btn-tiny"
                onClick={() => incrementSeed(1024)}
                title="Next Batch Seed (+1024)"
              >
                +1024
              </button>
            </div>
          </div>
          <input
            type="number"
            placeholder="random"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          />
        </label>

        <label>
          Batch
          <select
            value={batch}
            onChange={(e) => setBatch(Number(e.target.value))}
          >
            {Array.from({ length: 16 }, (_, i) => i + 1).map((n) => (
              <option key={n} value={n}>
                {n} image{n > 1 ? "s" : ""}
                {n > 1 ? ` (+1024 seed)` : ""}
              </option>
            ))}
          </select>
        </label>
      </div>

      {modelInfo.supports_negative && (
        <label>
          Negative prompt
          <textarea
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder="What to avoid: blurry, low quality..."
            rows={2}
          />
        </label>
      )}

      {modelInfo.samplers && (
        <label>
          Sampler
          <select value={sampler} onChange={(e) => setSampler(e.target.value)}>
            {modelInfo.samplers.map((sm) => (
              <option key={sm} value={sm}>
                {sm}
              </option>
            ))}
          </select>
        </label>
      )}

      {modelInfo.engine === "sdxl" && (
        <label>
          DeepCache
          <select value={cacheInterval} onChange={(e) => setCacheInterval(Number(e.target.value))}>
            <option value={1}>Off (Exact UNet)</option>
            <option value={2}>⚡ DeepCache 2 (~1.6x faster)</option>
            <option value={3}>⚡⚡ DeepCache 3 (~2x faster)</option>
          </select>
        </label>
      )}

      {supportsRef && (
        <fieldset className="ref-section">
          <legend>
            Reference images ({refImages.length}/{maxRefImages})
            {supportsMultiRef && <span className="ref-badge-pill">FLUX.2 In-Context</span>}
          </legend>

          <div className="ref-gallery-row">
            {refImages.map((img, idx) => (
              <div className="ref-card" key={img.id || img.path}>
                <div className="ref-thumb-wrapper">
                  <img src={img.preview} alt={`Reference ${idx + 1}`} />
                  <span className="ref-index-badge">Image {idx + 1}</span>
                  <button
                    type="button"
                    className="ref-remove-btn"
                    onClick={() => removeRefImage(idx)}
                    title="Remove reference image"
                  >
                    ✕
                  </button>
                </div>
                <button
                  type="button"
                  className="ref-insert-chip"
                  onClick={() => insertIntoPrompt(`Image ${idx + 1}`)}
                  title={`Insert "Image ${idx + 1}" into prompt`}
                >
                  + Prompt tag
                </button>
              </div>
            ))}

            {refImages.length < maxRefImages && (
              <label className="ref-add-card" title="Add reference image (up to 10 for FLUX.2)">
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp,.heic,.HEIC,image/heic,image/heif"
                  multiple={supportsMultiRef}
                  onChange={pickRefImages}
                  style={{ display: "none" }}
                />
                <span className="ref-add-plus">＋</span>
                <span className="ref-add-text">
                  {refImages.length === 0 ? "Add Image" : `Add (${refImages.length + 1})`}
                </span>
              </label>
            )}
          </div>

          {refImages.length > 0 && !supportsMultiRef && (
            <label className="ref-strength">
              Strength {Number(refStrength).toFixed(2)}
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={refStrength}
                onChange={(e) => setRefStrength(e.target.value)}
              />
            </label>
          )}
          {supportsMultiRef && (
            <p className="hint">
              💡 <strong>FLUX.2 In-Context Conditioning:</strong> Up to 10 reference images. The transformer injects image tokens directly into cross-attention. Refer to them naturally in your prompt as <code>Image 1</code>, <code>Image 2</code>, etc. (e.g. <em>&quot;A portrait of the character from Image 1 in the artistic style of Image 2&quot;</em>).
            </p>
          )}
        </fieldset>
      )}
    </div>
  );
}

export default memo(GenerationParams);
