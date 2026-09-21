export const API_BASE =
  import.meta.env?.VITE_API_BASE || "http://localhost:8001";

export async function api(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const headers = {
    ...(!isFormData && options.body ? { "Content-Type": "application/json" } : {}),
    ...(options.headers || {}),
  };
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      msg = (await res.json()).detail || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export function imageUrl(id, thumb = false) {
  return `${API_BASE}/api/images/${id}/file${thumb ? "?thumb=true" : ""}`;
}

export async function fetchImageBlob(id) {
  const res = await fetch(imageUrl(id));
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.statusText}`);
  return res.blob();
}
