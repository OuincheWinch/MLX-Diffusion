import { useEffect, useMemo, useState } from "react";
import { api, API_BASE } from "../api";
import { findLoraEntry } from "../utils/loraUtils";

export function useLoraPanel({ onError }) {
  const [loras, setLoras] = useState([]);
  const [loraRegistry, setLoraRegistry] = useState([]);
  const [newLora, setNewLora] = useState({ name: "", path: "" });
  const [savingLora, setSavingLora] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);

  useEffect(() => {
    api("/api/loras").then(setLoraRegistry).catch(() => {});
  }, []);

  const activeTriggerWords = useMemo(() => {
    const words = new Set();
    for (const lora of loras) {
      const entry = findLoraEntry(lora, loraRegistry);
      if (entry?.triggers) {
        for (const t of entry.triggers) {
          if (t && t.trim()) words.add(t.trim());
        }
      }
    }
    return Array.from(words);
  }, [loras, loraRegistry]);

  async function uploadLoraFile(file) {
    if (!file) return;
    setSavingLora(true);
    setUploadProgress("Uploading LoRA…");
    try {
      const form = new FormData();
      form.append("file", file);
      const name = newLora.name.trim() || file.name.replace(/\.safetensors$/, "");
      const res = await fetch(
        `${API_BASE}/api/loras/upload?name=${encodeURIComponent(name)}`,
        { method: "POST", body: form }
      );
      if (!res.ok) throw new Error((await res.json()).detail || "upload failed");
      const entry = await res.json();
      setLoraRegistry([
        ...loraRegistry.filter((l) => l.name !== entry.name && l.path !== entry.path),
        entry,
      ]);
      setNewLora({ name: "", path: "" });
      setUploadProgress(`Registered "${entry.name}" — pick it from the dropdown.`);
    } catch (e) {
      onError?.(String(e.message || e));
      setUploadProgress(null);
    } finally {
      setSavingLora(false);
    }
  }

  async function saveNewLora() {
    if (!newLora.name.trim() || !newLora.path.trim()) return;
    setSavingLora(true);
    try {
      const entry = await api("/api/loras", {
        method: "POST",
        body: JSON.stringify(newLora),
      });
      setLoraRegistry([...loraRegistry, entry]);
      setNewLora({ name: "", path: "" });
    } catch (e) {
      onError?.(String(e.message || e));
    } finally {
      setSavingLora(false);
    }
  }

  return {
    loras,
    setLoras,
    loraRegistry,
    setLoraRegistry,
    newLora,
    setNewLora,
    savingLora,
    uploadProgress,
    activeTriggerWords,
    uploadLoraFile,
    saveNewLora,
  };
}
