import { useState } from "react";
import { api } from "../api";

export default function QueueRecoveryDrawer({
  isOpen,
  onClose,
  items = [],
  onRefresh,
}) {
  const [restoring, setRestoring] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [copiedAll, setCopiedAll] = useState(false);

  if (!isOpen) return null;

  async function handleRestoreAll() {
    setRestoring(true);
    try {
      await api("/api/queue/recovery/restore", {
        method: "POST",
        body: JSON.stringify({ job_ids: [] }),
      });
      onRefresh?.();
      onClose();
    } catch (err) {
      alert("Erreur lors de la réinsertion : " + err);
    } finally {
      setRestoring(false);
    }
  }

  async function handleRestoreSingle(id) {
    setRestoring(true);
    try {
      await api("/api/queue/recovery/restore", {
        method: "POST",
        body: JSON.stringify({ job_ids: [id] }),
      });
      onRefresh?.();
    } catch (err) {
      alert("Erreur lors de la réinsertion : " + err);
    } finally {
      setRestoring(false);
    }
  }

  async function handleDeleteSingle(id) {
    try {
      await api(`/api/queue/recovery?job_id=${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      onRefresh?.();
    } catch {}
  }

  async function handleClearAll() {
    if (!window.confirm("Effacer définitivement l'historique de récupération ?")) return;
    try {
      await api("/api/queue/recovery", { method: "DELETE" });
      onRefresh?.();
      onClose();
    } catch {}
  }

  async function handleClearAndForget() {
    if (
      !window.confirm(
        "Effacer ET oublier ?\n\nCela supprime aussi les jobs encore en attente pour qu'aucun ne soit restauré au prochain démarrage. Action définitive."
      )
    ) {
      return;
    }
    try {
      await api("/api/queue/recovery?forget=true", { method: "DELETE" });
      onRefresh?.();
      onClose();
    } catch {}
  }

  function handleLoadInForm(item) {
    const req = item.request || {};
    window.dispatchEvent(
      new CustomEvent("mlx:load-prompt", {
        detail: { ...req },
      })
    );
    onClose();
  }

  async function handleCopyPrompt(item) {
    const text = item.request?.prompt || "";
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(item.id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch {}
  }

  async function handleCopyAllPrompts() {
    const all = items
      .map((it, idx) => `${idx + 1}. [${it.request?.model || "model"}] ${it.request?.prompt || ""}`)
      .join("\n\n");
    if (!all) return;
    try {
      await navigator.clipboard.writeText(all);
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 2000);
    } catch {}
  }

  function formatTime(ts) {
    if (!ts) return "";
    try {
      const d = new Date(ts * 1000);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch {
      return "";
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="queue-recovery-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="queue-recovery-header">
          <div className="queue-recovery-title">
            <span className="queue-recovery-icon">↺</span>
            <h3>File d'attente récupérable</h3>
            <span className="queue-recovery-count-badge">
              {items.length} {items.length > 1 ? "prompts" : "prompt"}
            </span>
          </div>
          <button type="button" className="btn-close" onClick={onClose} title="Fermer">
            ✕
          </button>
        </div>

        <p className="queue-recovery-subtitle">
          Prompts sauvegardés suite à une annulation de queue ou un arrêt imprévu de la machine.
        </p>

        {items.length > 0 && (
          <div className="queue-recovery-bulk-actions">
            <button
              type="button"
              className="btn-primary btn-restore-all"
              onClick={handleRestoreAll}
              disabled={restoring}
            >
              ↺ Tout réinsérer dans la file ({items.length})
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleCopyAllPrompts}
            >
              {copiedAll ? "✓ Copiés !" : "⎘ Copier tous les prompts"}
            </button>
            <button
              type="button"
              className="btn-danger-ghost"
              onClick={handleClearAll}
            >
              🗑 Tout effacer
            </button>
            <button
              type="button"
              className="btn-danger"
              onClick={handleClearAndForget}
              title="Efface aussi la file en attente : rien ne sera restauré au prochain démarrage"
            >
              🚫 Effacer et oublier
            </button>
          </div>
        )}

        <div className="queue-recovery-list">
          {items.length === 0 ? (
            <div className="queue-recovery-empty">
              <span className="empty-sparkle">✨</span>
              <p>Aucun prompt en attente de récupération.</p>
              <span className="hint">
                Vos générations annulées ou interrompues s'archiveront automatiquement ici.
              </span>
            </div>
          ) : (
            items.map((it) => {
              const req = it.request || {};
              const isInterrupted = it.reason === "interrupted";
              const loraCount = req.loras?.length || 0;

              return (
                <div
                  key={it.id}
                  className={`queue-recovery-item ${isInterrupted ? "interrupted" : "cancelled"}`}
                >
                  <div className="queue-recovery-item-head">
                    <span
                      className={`badge-status ${isInterrupted ? "badge-interrupted" : "badge-cancelled"}`}
                    >
                      {isInterrupted ? "⚡ Interrompu (Crash/Arrêt)" : "⌫ Annulé"}
                    </span>
                    <span className="queue-recovery-item-time">
                      {formatTime(it.timestamp)}
                    </span>
                    <span className="queue-recovery-model-tag">
                      {req.model || "modèle standard"}
                    </span>
                    {req.width && req.height && (
                      <span className="queue-recovery-spec-pill">
                        {req.width}×{req.height}
                      </span>
                    )}
                    {req.steps && (
                      <span className="queue-recovery-spec-pill">
                        {req.steps} steps
                      </span>
                    )}
                    {loraCount > 0 && (
                      <span className="queue-recovery-spec-pill">
                        {loraCount} LoRA{loraCount > 1 ? "s" : ""}
                      </span>
                    )}
                  </div>

                  <div className="queue-recovery-prompt" title={req.prompt}>
                    &ldquo;{req.prompt}&rdquo;
                  </div>

                  <div className="queue-recovery-item-actions">
                    <button
                      type="button"
                      className="btn-mini btn-action-requeue"
                      onClick={() => handleRestoreSingle(it.id)}
                      disabled={restoring}
                      title="Réinsérer ce prompt directement dans la file"
                    >
                      ↺ Réinsérer
                    </button>
                    <button
                      type="button"
                      className="btn-mini"
                      onClick={() => handleLoadInForm(it)}
                      title="Charger ce prompt et ses réglages dans le formulaire"
                    >
                      ✎ Charger
                    </button>
                    <button
                      type="button"
                      className="btn-mini"
                      onClick={() => handleCopyPrompt(it)}
                      title="Copier le texte du prompt"
                    >
                      {copiedId === it.id ? "✓ Copié" : "⎘ Copier"}
                    </button>
                    <button
                      type="button"
                      className="btn-mini btn-item-delete"
                      onClick={() => handleDeleteSingle(it.id)}
                      title="Supprimer définitivement ce prompt de la liste de récupération"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
