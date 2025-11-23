# scCAFM: building a causality-aware foundation model through gene regulation modeling

## Frontend docs & landing (Vue + Pinia)

The `website/` directory hosts the Pinia-inspired landing plus three documentation routes (`/guide/`, `/api/`, `/cookbook/`).  
Markdown sources live alongside the code in `docs/` and are fetched at runtime, so writing remains MD-first.

### Structure

- `website/index.html` – Pinia-style landing plus the in-page docs reader (Guide/API/Cookbook tabs).
- `website/{guide,api,cookbook}/index.html` – dedicated doc pages powered by the same Vue + Pinia store.
- `website/assets/css/site.css` – shared theme/layout styles for landing, docs, and UI chrome.
- `website/assets/js/site.js` – Pinia store, navigation/search handlers, markdown renderer, etc.
- `website/assets/js/docs-inline.js` – generated map that inlines the markdown content for offline use.
- `website/assets/js/vendor/` – vendored `vue`, `vue-demi`, `pinia`, and `marked` bundles.
- `docs/*.md` – markdown sources (edit these), and `tools/embed_docs.py` converts them into `docs-inline.js`.

### Usage

1. Serve the repo root with any static server (e.g., `python -m http.server 4173`) and visit `http://localhost:4173/website/`.
2. Scroll or use the nav/search to open Guide/API/Cookbook (either inline or via `/website/guide/index.html`, etc.); the content comes from the embedded markdown.
3. After editing any markdown file, run `python tools/embed_docs.py` to regenerate `website/assets/js/docs-inline.js`, then refresh the browser.

Because everything is plain HTML/CSS/JS, deployment to GitHub Pages, OSS, or any CDN only requires uploading `website/` + `docs/`.  
If you later adopt VitePress/Nuxt, you can still reuse the Pinia store and styles from `site.js`/`site.css`.

> ℹ️ 直接双击 `website/index.html` 也能浏览，但浏览器会把相对路径解析成目录列表。用一个迷你静态服务器能获得与线上一致的导航体验。
