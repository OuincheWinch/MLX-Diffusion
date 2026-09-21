import { useState } from "react";
import GenerateForm from "./components/GenerateForm";
import Gallery from "./components/Gallery";
import ParametersTab from "./components/ParametersTab";
import logo from "./assets/logo.png";
import "./App.css";

export default function App() {
  const [tab, setTab] = useState("generate");
  const [refreshKey, setRefreshKey] = useState(0);
  const [newImage, setNewImage] = useState(null);
  const [initialParams, setInitialParams] = useState(undefined);
  const [modelLabel, setModelLabel] = useState("FLUX.2-klein 4B");

  function handleGenerated(imageMeta) {
    setRefreshKey((k) => k + 1);
    if (imageMeta) {
      setNewImage(imageMeta);
    }
  }

  function handleImageSaved() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <img className="app-logo" src={logo} alt="MLX-DIFFUSION" />
          <h1>MLX-DIFFUSION</h1>
        </div>
        <nav>
          <button
            className={tab === "generate" ? "active" : ""}
            onClick={() => setTab("generate")}
          >
            Generate
          </button>
          <button
            className={tab === "browser" ? "active" : ""}
            onClick={() => setTab("browser")}
          >
            Browser
          </button>
          <button
            className={tab === "params" ? "active" : ""}
            onClick={() => setTab("params")}
          >
            ⚙️ Parameters
          </button>
        </nav>
      </header>

      <main>
        <div style={{ display: tab === "generate" ? "block" : "none" }}>
          <h2>{modelLabel} · MLX</h2>
          <GenerateForm
            onGenerated={handleGenerated}
            initialParams={initialParams}
            onModelChange={setModelLabel}
            onImageSaved={handleImageSaved}
          />
        </div>
        <div style={{ display: tab === "browser" ? "block" : "none" }}>
          <Gallery
            refreshKey={refreshKey}
            newImage={newImage}
            activeTab={tab}
            onReuse={(meta) => {
              setInitialParams({ ...meta, key: Date.now() });
              setTab("generate");
            }}
          />
        </div>
        <div style={{ display: tab === "params" ? "block" : "none" }}>
          <ParametersTab onNavigate={setTab} />
        </div>
      </main>
    </div>
  );
}
