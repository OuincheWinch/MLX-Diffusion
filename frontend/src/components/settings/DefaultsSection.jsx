import { useEffect, useMemo, useState } from "react";

const SAMPLER_NAMES = [
  "euler_trailing",
  "dpmpp_2m_karras",
  "euler_a_substep",
  "euler_a",
  "euler",
  "ddim",
];

export default function DefaultsSection({ settings, models, update, onFeedback }) {
  const [artistName, setArtistName] = useState("");
  const [outputFormat, setOutputFormat] = useState("png");
  const [stealth, setStealth] = useState(false);
  const [sampler, setSampler] = useState("");
  const [cacheInterval, setCacheInterval] = useState(1);
  const [fastVae, setFastVae] = useState(true);
  const [saving, setSaving] = useState(false);

  const [selModel, setSelModel] = useState("");
  const [draft, setDraft] = useState({});

  useEffect(() => {
    if (!settings) return;
    setArtistName(settings.artist_name || "");
    setOutputFormat(settings.default_output_format || "png");
    setStealth(!!settings.default_stealth);
    setSampler(settings.default_sampler || "");
    setCacheInterval(settings.default_cache_interval ?? 1);
    setFastVae(settings.default_fast_vae == null ? true : !!settings.default_fast_vae);
  }, [settings]);

  const availableModels = useMemo(
    () => (models.length ? models : []),
    [models]
  );

  useEffect(() => {
    if (!selModel && availableModels.length) setSelModel(availableModels[0].id);
  }, [availableModels, selModel]);

  const currentDefaults = useMemo(() => {
    if (!settings?.model_defaults) return {};
    return settings.model_defaults[selModel] || {};
  }, [settings, selModel]);

  useEffect(() => {
    if (!currentDefaults) return;
    const m = availableModels.find((x) => x.id === selModel);
    setDraft({
      steps: currentDefaults.steps ?? m?.default_steps ?? 4,
      guidance: currentDefaults.guidance ?? m?.default_guidance ?? 1.0,
      sampler: currentDefaults.sampler ?? m?.default_sampler ?? "",
      cache_interval: currentDefaults.cache_interval ?? m?.default_cache_interval ?? 1,
      fast_vae: currentDefaults.fast_vae ?? (m?.default_fast_vae ?? true),
      width: currentDefaults.width ?? 0,
      height: currentDefaults.height ?? 0,
    });
  }, [currentDefaults, selModel, availableModels]);

  async function save(updates) {
    setSaving(true);
    try {
      await update(updates);
      onFeedback?.({ type: "success", text: "Preferences saved" });
    } catch (e) {
      onFeedback?.({ type: "error", text: e.message || String(e) });
    } finally {
      setSaving(false);
    }
  }

  const selModelMeta = useMemo(
    () => availableModels.find((m) => m.id === selModel),
    [availableModels, selModel]
  );

  return (
    <>
      <div className="settings-row">
        <label className="settings-label">
          <span className="settings-field-title">🖋 Artist credit</span>
          <span className="settings-hint">
            Embedded in EXIF / Civitai metadata of every generated image. Empty
            uses &quot;MLX-DIFFUSION&quot;.
          </span>
          <div className="settings-inline">
            <input
              type="text"
              value={artistName}
              onChange={(e) => setArtistName(e.target.value)}
              placeholder="MLX-DIFFUSION"
              maxLength={200}
            />
            <button
              type="button"
              className="btn-mini"
              disabled={saving}
              onClick={() => save({ artist_name: artistName.trim() })}
            >
              Save artist
            </button>
          </div>
        </label>

        <div className="settings-grid">
          <label className="settings-field">
            <span>Default output format</span>
            <select
              value={outputFormat}
              onChange={(e) => setOutputFormat(e.target.value)}
            >
              <option value="png">PNG (lossless)</option>
              <option value="jpeg">JPEG (smaller)</option>
            </select>
          </label>
          <label className="settings-field settings-check">
            <input
              type="checkbox"
              checked={stealth}
              onChange={(e) => setStealth(e.target.checked)}
            />
            <span>Default 🥷 Stealth (no metadata)</span>
          </label>
        </div>
        <div className="settings-actions">
          <button
            type="button"
            className="btn-mini"
            disabled={saving}
            onClick={() =>
              save({
                default_output_format: outputFormat,
                default_stealth: stealth,
              })
            }
          >
            Save output preferences
          </button>
        </div>
      </div>

      <div className="settings-row">
        <span className="settings-field-title">◇ Global fallbacks</span>
        <span className="settings-hint">
          Applied only when a model has no explicit per-model override. Fast-VAE
          (TAESD/TAEF) trades a little fidelity for much faster VAE decoding.
        </span>
        <div className="settings-grid">
          <label className="settings-field">
            <span>Default sampler</span>
            <select value={sampler} onChange={(e) => setSampler(e.target.value)}>
              <option value="">(model default)</option>
              {SAMPLER_NAMES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="settings-field">
            <span>SDXL DeepCache interval</span>
            <input
              type="number"
              min={1}
              max={10}
              value={cacheInterval}
              onChange={(e) => setCacheInterval(Number(e.target.value))}
            />
          </label>
          <label className="settings-field settings-check">
            <input
              type="checkbox"
              checked={fastVae}
              onChange={(e) => setFastVae(e.target.checked)}
            />
            <span>⚡ Fast VAE (all engines)</span>
          </label>
        </div>
        <div className="settings-actions">
          <button
            type="button"
            className="btn-mini"
            disabled={saving}
            onClick={() =>
              save({
                default_sampler: sampler,
                default_cache_interval: cacheInterval,
                default_fast_vae: fastVae,
              })
            }
          >
            Save global defaults
          </button>
        </div>
      </div>

      <div className="settings-row">
        <span className="settings-field-title">🎛 Per-model defaults</span>
        <span className="settings-hint">
          Overrides loaded automatically when you switch models in the Generate
          form. Leave a field blank / 0 to keep the model&apos;s built-in value.
        </span>
        <div className="settings-inline">
          <select value={selModel} onChange={(e) => setSelModel(e.target.value)}>
            {availableModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          {currentDefaults && Object.keys(currentDefaults).length > 0 && (
            <span className="settings-badge">custom ⚙</span>
          )}
        </div>
        {selModelMeta && (
          <>
            <div className="settings-grid settings-grid-3">
              <label className="settings-field">
                <span>Steps</span>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={draft.steps ?? ""}
                  placeholder={String(selModelMeta.default_steps ?? 4)}
                  onChange={(e) => setDraft({ ...draft, steps: Number(e.target.value) })}
                />
              </label>
              <label className="settings-field">
                <span>Guidance</span>
                <input
                  type="number"
                  step={0.5}
                  min={0}
                  max={30}
                  value={draft.guidance ?? ""}
                  placeholder={String(selModelMeta.default_guidance ?? 1.0)}
                  onChange={(e) => setDraft({ ...draft, guidance: Number(e.target.value) })}
                />
              </label>
              <label className="settings-field">
                <span>Sampler</span>
                <select
                  value={draft.sampler || ""}
                  onChange={(e) => setDraft({ ...draft, sampler: e.target.value })}
                >
                  <option value="">(default)</option>
                  {(selModelMeta.samplers || SAMPLER_NAMES).map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <label className="settings-field">
                <span>DeepCache (SDXL)</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={draft.cache_interval ?? ""}
                  placeholder="1"
                  onChange={(e) =>
                    setDraft({ ...draft, cache_interval: Number(e.target.value) })
                  }
                />
              </label>
              <label className="settings-field settings-check">
                <input
                  type="checkbox"
                  checked={!!draft.fast_vae}
                  onChange={(e) => setDraft({ ...draft, fast_vae: e.target.checked })}
                />
                <span>Fast-VAE</span>
              </label>
              <label className="settings-field">
                <span>Width</span>
                <input
                  type="number"
                  min={128}
                  max={4096}
                  step={64}
                  value={draft.width || ""}
                  placeholder="default"
                  onChange={(e) => setDraft({ ...draft, width: Number(e.target.value) })}
                />
              </label>
              <label className="settings-field">
                <span>Height</span>
                <input
                  type="number"
                  min={128}
                  max={4096}
                  step={64}
                  value={draft.height || ""}
                  placeholder="default"
                  onChange={(e) => setDraft({ ...draft, height: Number(e.target.value) })}
                />
              </label>
            </div>
            <div className="settings-actions">
              <button
                type="button"
                className="btn-mini"
                disabled={saving}
                onClick={() => {
                  const overrides = {};
                  if (draft.steps) overrides.steps = Number(draft.steps);
                  if (draft.guidance) overrides.guidance = Number(draft.guidance);
                  if (draft.sampler) overrides.sampler = draft.sampler;
                  if (draft.cache_interval) overrides.cache_interval = Number(draft.cache_interval);
                  overrides.fast_vae = !!draft.fast_vae;
                  if (draft.width) overrides.width = Number(draft.width);
                  if (draft.height) overrides.height = Number(draft.height);
                  save({ model_defaults: { [selModel]: overrides } });
                }}
              >
                Save defaults for this model
              </button>
              {currentDefaults && Object.keys(currentDefaults).length > 0 && (
                <button
                  type="button"
                  className="btn-mini"
                  disabled={saving}
                  onClick={async () => {
                    try {
                      await update({ model_defaults: { [selModel]: {} } });
                      onFeedback?.({ type: "success", text: "Model defaults reset" });
                    } catch (e) {
                      onFeedback?.({ type: "error", text: e.message || String(e) });
                    }
                  }}
                >
                  Reset to built-in
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </>
  );
}