import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";

export function useModelConfig({ onModelChange }) {
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("flux2-klein-4b");
  const onModelChangeRef = useRef(onModelChange);

  useEffect(() => {
    onModelChangeRef.current = onModelChange;
  }, [onModelChange]);

  const loadModels = useCallback(async () => {
    try {
      const list = await api("/api/models");
      const normalized = list.map((m) => {
        if (m.id === "z-image-turbo" && !m.has_model_override) {
          return {
            ...m,
            default_steps: 8,
            presets: (m.presets || []).map((p) => ({
              ...p,
              steps: p.id === "draft" ? 6 : 8,
              label: p.id === "turbo" ? "⚡ Turbo (~35s)" : p.label,
            })),
          };
        }
        if (m.id === "flux2-klein-9b" && !m.has_model_override) {
          return {
            ...m,
            default_steps: 4,
            presets: (m.presets || []).map((p) => ({
              ...p,
              steps: 4,
            })),
          };
        }
        return m;
      });
      setModels(normalized);
      const m = normalized.find((x) => x.id === "flux2-klein-4b");
      if (m) onModelChangeRef.current?.(m.label);
    } catch {
      /* ignore transient network errors */
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const modelInfo = useMemo(() => {
    return (
      models.find((m) => m.id === model) ?? {
        id: "flux2-klein-4b",
        label: "FLUX.2-klein 4B",
        default_steps: 4,
        default_guidance: 1.0,
        supports_guidance: true,
        supports_fast_vae: true,
        lora_format: "FLUX.2",
        presets: [],
      }
    );
  }, [models, model]);

  return {
    models,
    model,
    setModel,
    modelInfo,
    refreshModels: loadModels,
  };
}
