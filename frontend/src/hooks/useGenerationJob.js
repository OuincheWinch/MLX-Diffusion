import { useEffect, useRef, useState } from "react";
import { api } from "../api";

export function useGenerationJob({ onGenerated, onImageSaved }) {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [jobPhase, setJobPhase] = useState(null);
  const [jobPhaseDetail, setJobPhaseDetail] = useState(null);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [generatingPrompt, setGeneratingPrompt] = useState(null);
  const [currentResult, setCurrentResult] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [submittedParams, setSubmittedParams] = useState(null);
  const [showSwitchDialog, setShowSwitchDialog] = useState(false);

  const lastSavedRef = useRef(null);
  const lastAutoShownIdRef = useRef(null);
  const onGeneratedRef = useRef(onGenerated);
  const onImageSavedRef = useRef(onImageSaved);

  useEffect(() => {
    onGeneratedRef.current = onGenerated;
    onImageSavedRef.current = onImageSaved;
  }, [onGenerated, onImageSaved]);

  // Initial setup: load latest image for canvas and resume any active running job
  useEffect(() => {
    api("/api/gallery?limit=1")
      .then((res) => {
        if (res.items?.length > 0) {
          setCurrentResult((prev) => prev ?? res.items[0]);
        }
      })
      .catch(() => {});

    api("/api/jobs?limit=10")
      .then((list) => {
        const active = list.find((j) => ["generating", "queued"].includes(j.status));
        if (active) {
          setJobId(active.id);
          setStatus(active.status);
          setGeneratingPrompt(active.prompt || null);
          if (active.progress) setProgress(active.progress);
        }
      })
      .catch(() => {});
  }, []);

  // Polling loop for active job
  useEffect(() => {
    if (!jobId) return;
    const t = setInterval(async () => {
      try {
        const job = await api(`/api/jobs/${jobId}`);
        setStatus(job.status);
        setJobPhase(job.phase || null);
        setJobPhaseDetail(job.phase_detail || null);
        const actualPrompt = job.request?.prompt || job.prompt || null;
        if (actualPrompt) {
          setGeneratingPrompt(actualPrompt);
        }
        if (job.progress) {
          setProgress(job.progress);
          if (
            job.progress.saved_id &&
            job.progress.saved_id !== lastSavedRef.current
          ) {
            lastSavedRef.current = job.progress.saved_id;
            onImageSavedRef.current?.(job.progress.saved_id, job.progress.saved_index, job.progress.batch);
          }
        }

        // Live Batch Updates: sync partial results and update canvas immediately
        if (job.partial_results && job.partial_results.length > 0) {
          setBatchResults(job.partial_results);
        }
        if (job.last_result && job.last_result.id !== lastAutoShownIdRef.current) {
          lastAutoShownIdRef.current = job.last_result.id;
          setCurrentResult(job.last_result);
          onGeneratedRef.current?.(job.last_result);
        } else if (
          job.progress?.saved_id &&
          job.progress.saved_id !== lastAutoShownIdRef.current &&
          !job.last_result
        ) {
          lastAutoShownIdRef.current = job.progress.saved_id;
          api(`/api/images/${job.progress.saved_id}`)
            .then((imgMeta) => {
              if (imgMeta) {
                setCurrentResult(imgMeta);
                onGeneratedRef.current?.(imgMeta);
                setBatchResults((prev) => {
                  if (prev.some((x) => x.id === imgMeta.id)) return prev;
                  return [...prev, imgMeta];
                });
              }
            })
            .catch(() => {});
        }

        if (job.status === "done") {
          clearInterval(t);
          const finalResults = job.results || job.partial_results || [];
          if (finalResults.length > 0) {
            setBatchResults(finalResults);
            for (const r of finalResults) {
              if (r.id !== lastSavedRef.current) {
                onImageSavedRef.current?.(r.id, r.seed, finalResults.length);
              }
            }
            const last = finalResults[finalResults.length - 1];
            if (last) {
              lastAutoShownIdRef.current = last.id;
              setCurrentResult(last);
              onGeneratedRef.current?.(last);
            }
          } else if (job.result) {
            lastAutoShownIdRef.current = job.result.id;
            setCurrentResult(job.result);
            onGeneratedRef.current?.(job.result);
            setBatchResults([job.result]);
          } else {
            setError(job.error || "Generation completed without producing an image.");
          }

          // Automatically chain to next queued/generating job in stack if present
          try {
            const activeList = await api("/api/jobs?limit=10");
            const nextJob = activeList.find(
              (j) => j.id !== jobId && ["queued", "generating"].includes(j.status)
            );
            if (nextJob) {
              setJobId(nextJob.id);
              setStatus(nextJob.status);
              setJobPhase(nextJob.phase || null);
              setJobPhaseDetail(nextJob.phase_detail || null);
              setProgress(nextJob.progress || null);
              setGeneratingPrompt(nextJob.prompt || null);
              setBatchResults(nextJob.partial_results || []);
              lastAutoShownIdRef.current = null;
              lastSavedRef.current = null;
              return;
            }
          } catch {}

          setJobId(null);
          setStatus(null);
          setJobPhase(null);
          setJobPhaseDetail(null);
          setProgress(null);
          setGeneratingPrompt(null);
        } else if (["error", "cancelled"].includes(job.status)) {
          clearInterval(t);
          setError(job.error || (job.status === "cancelled" ? "Generation cancelled" : "Generation failed"));
          const partials = job.partial_results || [];
          if (partials.length > 0) {
            setBatchResults(partials);
            const last = partials[partials.length - 1];
            lastAutoShownIdRef.current = last.id;
            setCurrentResult(last);
            onGeneratedRef.current?.(last);
          }

          // Check if another job is waiting in the queue
          try {
            const activeList = await api("/api/jobs?limit=10");
            const nextJob = activeList.find(
              (j) => j.id !== jobId && ["queued", "generating"].includes(j.status)
            );
            if (nextJob) {
              setJobId(nextJob.id);
              setStatus(nextJob.status);
              setJobPhase(nextJob.phase || null);
              setJobPhaseDetail(nextJob.phase_detail || null);
              setProgress(nextJob.progress || null);
              setGeneratingPrompt(nextJob.prompt || null);
              setBatchResults(nextJob.partial_results || []);
              lastAutoShownIdRef.current = null;
              lastSavedRef.current = null;
              return;
            }
          } catch {}

          setJobId(null);
          setStatus(null);
          setJobPhase(null);
          setJobPhaseDetail(null);
          setProgress(null);
          setGeneratingPrompt(null);
        }
      } catch (e) {
        clearInterval(t);
        setError(String(e));
        setJobId(null);
        setStatus(null);
        setJobPhase(null);
        setJobPhaseDetail(null);
        setGeneratingPrompt(null);
      }
    }, 500);
    return () => clearInterval(t);
  }, [jobId]);

  // Cross-component event listeners for immediate cancel sync
  useEffect(() => {
    function handleCancelAll() {
      setJobId(null);
      setStatus(null);
      setJobPhase(null);
      setJobPhaseDetail(null);
      setProgress(null);
      setGeneratingPrompt(null);
    }

    function handleCancelJob(e) {
      if (e.detail?.id && e.detail.id === jobId) {
        setJobId(null);
        setStatus(null);
        setJobPhase(null);
        setJobPhaseDetail(null);
        setProgress(null);
        setGeneratingPrompt(null);
      }
    }

    window.addEventListener("mlx:cancel-all", handleCancelAll);
    window.addEventListener("mlx:cancel-job", handleCancelJob);
    return () => {
      window.removeEventListener("mlx:cancel-all", handleCancelAll);
      window.removeEventListener("mlx:cancel-job", handleCancelJob);
    };
  }, [jobId]);

  // Idle watcher: when not tracking any job, check periodically if an active job appeared
  useEffect(() => {
    if (jobId) return;
    const idleInterval = setInterval(async () => {
      try {
        const list = await api("/api/jobs?limit=10");
        const active = list.find((j) => ["generating", "queued"].includes(j.status));
        if (active) {
          setJobId(active.id);
          setStatus(active.status);
          setJobPhase(active.phase || null);
          setJobPhaseDetail(active.phase_detail || null);
          setProgress(active.progress || null);
          setGeneratingPrompt(active.prompt || null);
          setBatchResults(active.partial_results || []);
          lastAutoShownIdRef.current = null;
          lastSavedRef.current = null;
        }
      } catch {}
    }, 1500);
    return () => clearInterval(idleInterval);
  }, [jobId]);

  function activateJob(newJobId, promptText) {
    lastSavedRef.current = null;
    lastAutoShownIdRef.current = null;
    setBatchResults([]);
    setStatus("queued");
    setJobPhase("loading_model");
    setJobPhaseDetail("Queued · Waiting for engine...");
    setJobId(newJobId);
    setGeneratingPrompt(promptText);
    setProgress(null);
    setError(null);
  }

  async function cancelJob() {
    const currentId = jobId;
    setJobId(null);
    setStatus(null);
    setJobPhase(null);
    setJobPhaseDetail(null);
    setProgress(null);
    setGeneratingPrompt(null);
    if (!currentId) return;
    try {
      await api(`/api/jobs/${currentId}/cancel`, { method: "POST" });
    } catch {}
  }

  return {
    jobId,
    status,
    jobPhase,
    jobPhaseDetail,
    progress,
    error,
    setError,
    generatingPrompt,
    currentResult,
    setCurrentResult,
    batchResults,
    setBatchResults,
    submittedParams,
    setSubmittedParams,
    showSwitchDialog,
    setShowSwitchDialog,
    activateJob,
    cancelJob,
    lastSavedRef,
    lastAutoShownIdRef,
  };
}
