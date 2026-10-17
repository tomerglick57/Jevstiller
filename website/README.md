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

## Search engines

`public/robots.txt` allows everything and points at the sitemap Starlight generates (`/sitemap-index.xml`).

**Bing, and through it ChatGPT search and Copilot:** `npm run indexnow` reads the live sitemap and submits every
page to IndexNow. Run it after a deploy that adds or rewrites pages. The key file `public/<key>.txt` proves we own
the host; keep it, and keep it in sync with the script if you ever rotate it.

**Google:** indexing requests go through [Search Console](https://search.google.com/search-console). Add a
URL-prefix property for `https://jevstiller.pages.dev/`, pick the "HTML file" method, put the downloaded
`google<token>.html` in `public/`, deploy, then verify and submit `sitemap-index.xml`. Bing Webmaster Tools can
then import the verified property in one click, which also lets you watch IndexNow submissions there.

## Page titles and descriptions

A synced page's `<title>` and meta description come from its source file's heading and first paragraph, unless the
source carries overrides in HTML comments right under the heading (invisible on GitHub):

```markdown
# Cut your Jev API bill
<!-- title: Cut your Jev API bill: where the spend comes from and three ways to pay less -->
<!-- description: Jev bills per input token. Shorter states, ... -->
```

Write them for the person searching: say "Jev" and the subject, keep the description to about two lines of plain
prose, and make it true of the page. Hand-written pages set `title` and `description` in their frontmatter.

## Structured data

Every docs page carries a schema.org `TechArticle` block (`src/components/Head.astro`, registered as Starlight's
`Head` component override): title, description, canonical URL and the last-updated date from git. The landing page
carries a `SoftwareApplication` block (in `src/pages/index.astro`; `softwareVersion` follows the `version` constant
bumped at each release). A synced page with a `## FAQ` section whose questions are H3s also gets a `FAQPage` block
from `scripts/sync-docs.mjs`. Check a page with Google's Rich Results Test, or:

```bash
curl -s https://jevstiller.pages.dev/guides/cut-jev-cost/ | grep -o '<script type="application/ld+json">[^<]*</script>'
```

## Savings calculator

`src/content/docs/calculator.mdx` embeds `src/components/SavingsCalculator.astro`: a client-side form that turns a
volume and one of the benchmark tasks into a monthly Jev bill with and without the proxy. The constants (price,
audit rate, latencies, warm-up, per-task coverage and agreement) sit at the top of the component with their
sources; update them when `docs/benchmarks.md` changes.
