import { useCallback, useEffect, useState, memo } from "react";
import { api } from "../api";
import QueueRecoveryDrawer from "./QueueRecoveryDrawer";

const STATUS_LABELS = {
  queued: "⏳ Queued",
  generating: "⚙ Generating",
  done: "✓ Done",
  error: "✗ Error",
  cancelled: "Cancelled",
};

function GenerationStack() {
  const [jobs, setJobs] = useState([]);
  const [recoverableItems, setRecoverableItems] = useState([]);
  const [showRecovery, setShowRecovery] = useState(false);

  const fetchRecovery = useCallback(async () => {
    try {
      const res = await api("/api/queue/recovery");
      setRecoverableItems(res.items || []);
    } catch {}
  }, []);

  useEffect(() => {
    let alive = true;
    let timerId = null;

    async function poll() {
      if (timerId) clearTimeout(timerId);
      try {
        const list = await api("/api/jobs?limit=25");
        if (!alive) return;
        const cutoff = Date.now() / 1000 - 120;
        const relevant = list.filter(
          (j) =>
            ["queued", "generating"].includes(j.status) ||
            (j.finished_at ?? 0) > cutoff
        );
        setJobs(relevant);
        fetchRecovery();
        const hasActive = relevant.some((j) =>
          ["queued", "generating"].includes(j.status)
        );
        const isHidden = typeof document !== "undefined" && document.hidden;
        const delay = hasActive ? (isHidden ? 8000 : 4000) : (isHidden ? 30000 : 10000);
        timerId = setTimeout(poll, delay);
      } catch {
        if (alive) {
          const isHidden = typeof document !== "undefined" && document.hidden;
          timerId = setTimeout(poll, isHidden ? 30000 : 10000);
        }
      }
    }

    function handleVisibilityChange() {
      if (typeof document !== "undefined" && !document.hidden && alive) {
        poll();
      }
    }

    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }
    poll();
    fetchRecovery();

    return () => {
      alive = false;
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      }
      if (timerId) clearTimeout(timerId);
    };
  }, [fetchRecovery]);

  const active = jobs.filter((j) => ["queued", "generating"].includes(j.status));

  // Count only active workload (current and next/queued generations), adjusted by batch number
  const activeImageCount = active.reduce((sum, j) => {
    const totalBatch = j.batch || j.progress?.batch || 1;
    if (j.status === "generating") {
      const currentIndex = j.progress?.image_index ?? 0;
      const remainingInBatch = Math.max(1, totalBatch - currentIndex);
      return sum + remainingInBatch;
    }
    return sum + totalBatch;
  }, 0);

  async function cancelJob(id) {
    window.dispatchEvent(new CustomEvent("mlx:cancel-job", { detail: { id } }));
    try {
      await api(`/api/jobs/${id}/cancel`, { method: "POST" });
      setJobs((prev) =>
        prev.map((j) => (j.id === id ? { ...j, status: "cancelled" } : j))
      );
      fetchRecovery();
    } catch {}
  }

  async function emptyQueue() {
    window.dispatchEvent(new CustomEvent("mlx:cancel-all"));
    try {
      await api("/api/jobs/cancel-all", { method: "POST" });
      setJobs((prev) =>
        prev.map((j) =>
          ["queued", "generating"].includes(j.status)
            ? { ...j, status: "cancelled" }
            : j
        )
      );
      fetchRecovery();
    } catch {}
  }

  const interruptedCount = recoverableItems.filter((x) => x.reason === "interrupted").length;

  if (jobs.length === 0 && recoverableItems.length === 0) return null;

  return (
    <>
      <div className="gen-stack">
        <h3>
          Generation stack ({activeImageCount})
          <span className="gen-stack-actions">
            {active.some((j) => j.status === "generating") && (
              <button
                className="btn-mini"
                onClick={() => {
                  const run = active.find((j) => j.status === "generating");
                  if (run) cancelJob(run.id);
                }}
              >
                ✕ Kill current
              </button>
            )}
            {active.length > 1 && (
              <button className="btn-mini" onClick={emptyQueue}>
                ⌫ Empty queue
              </button>
            )}
            {recoverableItems.length > 0 && (
              <button
                className="btn-mini btn-recovery-trigger"
                onClick={() => setShowRecovery(true)}
                title="Consulter et réinsérer les prompts annulés ou interrompus"
              >
                ↺ {recoverableItems.length} récupérable{recoverableItems.length > 1 ? "s" : ""}
              </button>
            )}
          </span>
        </h3>

        {interruptedCount > 0 && (
          <div className="queue-crash-alert">
            <span>
              ⚡ {interruptedCount} génération{interruptedCount > 1 ? "s" : ""} interrompue{interruptedCount > 1 ? "s" : ""} lors de la dernière session
            </span>
            <button
              className="btn-mini btn-crash-restore"
              onClick={() => setShowRecovery(true)}
            >
              Voir & Restaurer
            </button>
          </div>
        )}

        {jobs.map((j) => {
          const p = j.progress;
          const pct =
            j.status === "done"
              ? 100
              : p && p.steps > 0
                ? Math.min(100, (p.step / p.steps) * 100)
                : null;
          return (
            <div key={j.id} className={`gen-stack-item ${j.status}`}>
              <div className="gen-stack-head">
                <span className={`gen-stack-status ${j.phase ? `phase-${j.phase}` : ""}`}>
                  {j.status === "generating" && j.phase === "downloading"
                    ? "📥 Downloading"
                    : j.status === "generating" && j.phase === "loading_model"
                      ? "🧠 Loading Memory"
                      : j.status === "generating" && j.phase === "compiling"
                        ? "⚡ Compiling"
                        : j.status === "generating" && j.phase === "saving"
                          ? "🎨 Finalizing"
                          : STATUS_LABELS[j.status] || j.status}
                </span>
                <span className="gen-stack-model">{j.model}</span>
              </div>
              {["queued", "generating"].includes(j.status) && (
                <button
                  className="btn-mini gen-stack-kill"
                  title="Cancel this job"
                  onClick={() => cancelJob(j.id)}
                >
                  ✕
                </button>
              )}
              <div className="gen-stack-prompt" title={j.prompt}>
                {j.prompt}
              </div>
              {pct != null && (
                <div className="progress-bar-hairline">
                  <div className="progress-fill" style={{ width: `${pct}%` }} />
                </div>
              )}
              {p ? (
                <div className="hint">
                  {j.status === "done" ? (
                    <>
                      Completed in {j.result?.generation_time ?? Math.round(p.elapsed)}s ({p.steps} steps
                      {(p.batch > 1 || j.batch > 1) ? ` · ${p.batch || j.batch} images` : ""})
                    </>
                  ) : (
                    <>
                      {p.step >= p.steps ? (
                        <>Decoding image (VAE) · elapsed {Math.round(p.elapsed)}s</>
                      ) : (
                        <>Step {p.step}/{p.steps} · elapsed {Math.round(p.elapsed)}s</>
                      )}
                      {p.eta_seconds != null && p.eta_seconds >= 0 && (
                        <> · ETA ~{Math.round(p.eta_seconds)}s</>
                      )}
                      {(p.batch > 1 || j.batch > 1) && (
                        <> · image {(p.image_index ?? 0) + 1}/{p.batch || j.batch}</>
                      )}
                    </>
                  )}
                </div>
              ) : j.status === "queued" ? (
                <div className="hint">
                  Waiting in queue{(j.batch > 1) ? ` · batch of ${j.batch} images` : ""}
                </div>
              ) : j.status === "generating" && !p ? (
                <div className="hint">
                  {j.phase_detail || (j.phase === "downloading" ? "Downloading weights…" : "Loading model weights into unified memory…")}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <QueueRecoveryDrawer
        isOpen={showRecovery}
        onClose={() => setShowRecovery(false)}
        items={recoverableItems}
        onRefresh={fetchRecovery}
      />
    </>
  );
}

export default memo(GenerationStack);
