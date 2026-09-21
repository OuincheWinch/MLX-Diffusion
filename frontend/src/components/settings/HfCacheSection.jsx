import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatBytes } from "../../utils/formatBytes";

export default function HfCacheSection({ onFeedback }) {
  const [cache, setCache] = useState(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(null);
  const [err, setErr] = useState(null);

  async function refresh() {
    setLoading(true);
    try {
      const data = await api("/api/hf/cache");
      setCache(data);
      setErr(data.error || null);
    } catch (e) {
      setErr(e.message || String(e));
      setCache(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function clearRepo(repo) {
    if (!window.confirm(`Remove "${repo.repo_id}" (${formatBytes(repo.size_bytes)}) from the Hugging Face cache?`)) {
      return;
    }
    setClearing(repo.repo_id);
    try {
      const res = await api("/api/hf/cache/clear", {
        method: "POST",
        body: JSON.stringify({ repo_id: repo.repo_id }),
      });
      onFeedback?.({ type: "success", text: `Freed ${formatBytes(res.freed_bytes)}` });
      await refresh();
    } catch (e) {
      onFeedback?.({ type: "error", text: e.message || String(e) });
    } finally {
      setClearing(null);
    }
  }

  if (loading && !cache) {
    return <div className="settings-row"><span className="hint">Scanning Hugging Face cache…</span></div>;
  }
  if (err && !cache) {
    return <div className="settings-row"><span className="error">{err}</span></div>;
  }

  const repos = cache?.repos || [];
  return (
    <div className="settings-row">
      <div className="settings-inline">
        <span className="settings-badge">{formatBytes(cache?.total_bytes || 0)} total</span>
        <span className="settings-hint">
          {cache?.exists ? cache.root : `${cache?.root} does not exist yet`}
        </span>
        <button type="button" className="btn-mini" onClick={refresh} disabled={loading}>
          ↺ Refresh
        </button>
      </div>
      {repos.length === 0 ? (
        <p className="params-hint">No cached repositories.</p>
      ) : (
        <div className="models-table">
          <div className="models-table-head">
            <span>Repository</span>
            <span>Size</span>
            <span>Details</span>
            <span>Actions</span>
          </div>
          {repos.map((r) => (
            <div className="models-table-row" key={r.repo_id}>
              <span className="models-name">
                <strong>{r.repo_id}</strong>
                <span className="models-id">{r.repo_type}</span>
              </span>
              <span className="models-size">{formatBytes(r.size_bytes)}</span>
              <span className="models-state">
                {r.n_revisions} revision{r.n_revisions !== 1 ? "s" : ""} · {r.n_files} files
              </span>
              <span className="models-actions">
                <button
                  type="button"
                  className="btn-mini"
                  disabled={clearing === r.repo_id}
                  onClick={() => clearRepo(r)}
                >
                  {clearing === r.repo_id ? "Removing…" : "🗑 Clear"}
                </button>
              </span>
            </div>
          ))}
        </div>
      )}
      <p className="params-hint">
        Model weights live in the shared Hugging Face hub cache (~/.cache/huggingface). Clearing a
        repo removes snapshots + blobs; generation re-downloads it lazily. Repos currently
        downloading are protected.
      </p>
    </div>
  );
}