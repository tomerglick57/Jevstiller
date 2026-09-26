# Jevstiller website

The project site and docs, built with [Starlight](https://starlight.astro.build) and hosted on Cloudflare.
The landing page is `src/pages/index.astro` (with `src/styles/landing.css`); docs pages are Markdown files in
`src/content/docs/`, and the sidebar is in `astro.config.mjs`.

```bash
npm install
npm run dev        # live-reloading dev server at http://localhost:4321
npm run preview    # build, then serve through Cloudflare's local runtime (wrangler pages dev)
npm run deploy     # build, then deploy to production (needs `npx wrangler login` once)
npm run media      # redraw the README's pictures (docs/media/*.svg) from src/data/banking77-live.json
```

Hosted on Cloudflare Pages as the `jevstiller` project: https://jevstiller.pages.dev. `wrangler.jsonc`
points Pages at `dist/`; there is no server code.

Deploys normally happen in CI (`.github/workflows/website.yml`): a push to `main` that touches `website/`
goes to production, and a pull request from this repo gets a preview at `https://<branch>.jevstiller.pages.dev`.
`npm run deploy` is the manual fallback.
