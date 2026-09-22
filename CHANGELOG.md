# Changelog

All notable changes to **MLX-Diffusion** are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [0.1.1] — beta — 2026-09-22
First public beta.

### Added
- **⚖ Licences tab** in the app: per-package + per-model licence tables with repo links, our MIT © Ouinche reminder, and a link to the GitHub repository.
- **Version stamping**: `v0.1.1 (beta)` in the page title and app header; new `/api/version` endpoint; `frontend/src/version.js` and `backend/app_version.py` as the single version sources.
- **Public beta docs**: `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, and a GitHub Actions CI workflow (backend `py_compile`, frontend lint + build).
- README: author credit (Ouinche — www.ouinche.com), AI-assist credit (Gemini, 0xAlpha, Big Pickle), beta disclaimer, refreshed License section.

### Changed
- Prompt Enhancer tightened to a strict output contract (prompt only, hard word cap per engine, no commentary) and tagged *experimental* in the UI.
- Qwen-Image 2.1: demoted to *experimental* — now uses the validated `flow_match_euler_discrete` recipe (quality/portrait presets at 40 steps), refuses >768×768 with a clean error instead of OOMing, and shows a confirmation warning before launch.
- Size caps (`max_pixels`/`max_side`) removed for Krea 2 Turbo and Qwen-Image 2.1 (16 GB caveats documented).
- Header brand moved to consistent `MLX-Diffusion` casing.