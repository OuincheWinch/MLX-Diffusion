import { useEffect, useState } from "react";
import { api } from "../api";
import {
  GITHUB_REPO_URL,
  GITHUB_LICENSE_URL,
  AUTHOR_WEBSITE,
  AI_CREDITS,
  LICENCE_SECTIONS,
} from "../data/licences";
import { APP_VERSION_LABEL } from "../version";

function ExtLink({ href, children }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
}

export default function LicencesTab() {
  const [apiVersion, setApiVersion] = useState(null);

  useEffect(() => {
    let alive = true;
    api("/api/version")
      .then((v) => {
        if (alive) setApiVersion(v?.version ? v.version : null);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="parameters-tab licences-tab">
      <section className="params-section">
        <h3>⚖ Licences — MLX-Diffusion</h3>
        <p className="params-section-desc">
          What you may and may not do with MLX-Diffusion, and which projects it
          is built from. Every third-party package below keeps its own licence;
          nothing here grants or revokes those terms.
        </p>
      </section>

      <section className="params-section">
        <h3>This project</h3>
        <p>
          <strong>MLX-Diffusion</strong> — {APP_VERSION_LABEL} — is released
          under the <ExtLink href={GITHUB_LICENSE_URL}>MIT License</ExtLink>,
          Copyright © 2026{" "}
          <ExtLink href={AUTHOR_WEBSITE}>Ouinche</ExtLink>.
          Source: <ExtLink href={GITHUB_REPO_URL}>github.com/OuincheWinch/MLX-Diffusion</ExtLink>.
        </p>
        <p>
          <strong>Heavily coded by AI</strong> — assisted by{" "}
          {AI_CREDITS.join(", ")} together with its human author. Reviewed and
          benchmarked by hand.
        </p>
        <p className="licence-hint">
          The MIT licence covers <em>this project's source code only</em>. The
          model weights are not covered by it — see the weights table below.
        </p>
      </section>

      {LICENCE_SECTIONS.map((section) => (
        <section className="params-section" key={section.id}>
          <h3>{section.title}</h3>
          {section.note && <p className="params-section-desc">{section.note}</p>}
          <table className="licence-table">
            <thead>
              <tr>
                <th>Package</th>
                <th>Licence</th>
                <th>Repository</th>
              </tr>
            </thead>
            <tbody>
              {section.packages.map((pkg) => (
                <tr key={pkg.name}>
                  <td>
                    {pkg.name}
                    {pkg.extra && <div className="licence-extra">{pkg.extra}</div>}
                  </td>
                  <td className={pkg.caution ? "licence-caution" : ""}>{pkg.license}</td>
                  <td>
                    <ExtLink href={pkg.url}>
                      {pkg.url.replace("https://", "").replace(/\/$/, "")}
                    </ExtLink>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}

      <section className="params-section">
        <h3>Versions</h3>
        <p className="params-section-desc">
          Frontend {APP_VERSION_LABEL}
          {apiVersion && ` · Backend API ${apiVersion}`}
          {apiVersion && (
            <>
              {" · "}
              <ExtLink href={GITHUB_REPO_URL}>repo</ExtLink>
            </>
          )}
        </p>
      </section>
    </div>
  );
}