import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatBytes } from "../../utils/formatBytes";

function fmtSeconds(s) {
  if (s == null || isNaN(s)) return "—";
  if (s <= 0) return "now";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function gauge(label, value, title) {
  return (
    <span className="engine-chip" title={title}>
      <em>{label}</em> {value}
    </span>
  );
}

export default function EngineSection() {
  const [status, setStatus] = useState(null);
  const [err, setErr] = useState(null);
  const [showStderr, setShowStderr] = useState(false);

  useEffect(() => {
    let alive = true;
    let refreshTimer = null;
    async function poll() {
      try {
        const s = await api("/api/engine/status");
        if (alive) {
          setStatus(s);
          setErr(null);
        }
      } catch (e) {
        if (alive) setErr(e.message || String(e));
      }
      if (alive) refreshTimer = setTimeout(poll, 5000);
    }
    refreshTimer = setTimeout(poll, 0);
    return () => {
      alive = false;
      clearTimeout(refreshTimer);
    };
  }, []);

  if (err) {
    return (
      <div className="settings-row">
        <span className="error">{err}</span>
      </div>
    );
  }
  if (!status) {
    return <div className="settings-row"><span className="hint">Loading engine status…</span></div>;
  }

  const metal = status.metal || {};
  const wired = status.wired || {};
  const mflux = status.mflux || {};
  const sdxl = status.sdxl || {};
  const storage = status.storage || {};

  return (
    <div className="settings-row">
      <div className="engine-grid">
        <div className="engine-card">
          <header>
            <strong>🖥 Metal / GPU</strong>
          </header>
          <div className="engine-chips">
            {metal.name && gauge("GPU", metal.name)}
            {gauge("Memory", metal.memory_size ? formatBytes(metal.memory_size) : "—")}
            {metal.recommended_max_working_set_size
              ? gauge("Max working set", formatBytes(metal.recommended_max_working_set_size))
              : null}
          </div>
          <div className="engine-chips">
            {gauge("Peak memory", metal.peak_memory != null ? formatBytes(metal.peak_memory) : "—", "Highest wired memory reached this session")}
            {gauge("Cache", metal.cache_memory != null ? formatBytes(metal.cache_memory) : "—", "mlx allocator cache")}
          </div>
          <div className="engine-chips">
            {gauge("Wired limit", wired.generic_budget_bytes ? formatBytes(wired.generic_budget_bytes) : "unbounded", `MLX_WIRED_LIMIT_GB=${wired.generic_limit_gb || 0}`)}
            {gauge("krea2 wired", wired.krea_budget_bytes ? formatBytes(wired.krea_budget_bytes) : "unbounded", `MLX_KREA_WIRED_LIMIT_GB=${wired.krea_limit_gb || 0}`)}
          </div>
        </div>

        <div className="engine-card">
          <header>
            <strong>⚡ mflux pipelines</strong>
            <span className={`engine-dot ${mflux.resident ? "on" : "off"}`} />
          </header>
          <p className="engine-resident">
            {mflux.resident
              ? <>
                  <strong>{mflux.model || "model"}</strong> resident in unified memory
                  {mflux.watchdog_armed && mflux.seconds_until_release != null && (
                    <span className="engine-watchdog">
                      auto-release in {fmtSeconds(mflux.seconds_until_release)}
                    </span>
                  )}
                </>
              : "No pipeline resident (cold reload on next generation)"}
          </p>
          <div className="engine-chips">
            {gauge("Idle policy", `${mflux.idle_kill_s || 300}s`)}
            {gauge("Prompt cache", String(mflux.prompt_cache_size ?? 0))}
            {gauge("Watchdog", mflux.watchdog_armed ? "armed" : "cold")}
          </div>
        </div>

        <div className="engine-card">
          <header>
            <strong>🧵 SDXL daemon</strong>
            <span className={`engine-dot ${sdxl.resident ? "on" : "off"}`} />
          </header>
          <p className="engine-resident">
            {sdxl.resident
              ? <>
                  <strong>{sdxl.model || "engine"}</strong> alive
                  {sdxl.watchdog_armed && sdxl.seconds_until_release != null && (
                    <span className="engine-watchdog">
                      auto-release in {fmtSeconds(sdxl.seconds_until_release)}
                    </span>
                  )}
                </>
              : "Daemon not running (spawned lazily)"}
          </p>
          <div className="engine-chips">
            {gauge("Idle policy", `${sdxl.idle_kill_s || 300}s`)}
            {gauge("Watchdog", sdxl.watchdog_armed ? "armed" : "cold")}
          </div>
          {sdxl.stderr_tail && (
            <>
              <button
                type="button"
                className="btn-mini"
                onClick={() => setShowStderr((v) => !v)}
              >
                {showStderr ? "Hide daemon log" : "Daemon log tail"}
              </button>
              {showStderr && <pre className="engine-stderr">{sdxl.stderr_tail}</pre>}
            </>
          )}
        </div>

        <div className="engine-card">
          <header>
            <strong>💾 Storage</strong>
          </header>
          <div className="engine-chips">
            {gauge("Generated", `${storage.image_count ?? 0} images`)}
            {gauge("Size", storage.image_bytes != null ? formatBytes(storage.image_bytes) : "—")}
            {gauge("TAEF", (status.taef || []).length ? status.taef.join(", ") : "none loaded")}
          </div>
          <div className="engine-chips">
            {gauge("Settings file", storage.settings_exists ? "present" : "absent", storage.settings_file)}
          </div>
        </div>
      </div>
    </div>
  );
}