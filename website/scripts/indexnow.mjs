// Tell Bing (and the other IndexNow engines: Yandex, Seznam, Naver, Yep) which pages changed.
// Reads the deployed sitemap and submits every URL in one request. The key file public/<key>.txt proves we own the host.
//
//   npm run indexnow                 # submit all pages in the live sitemap
//   npm run indexnow -- /guides/cut-jev-cost/ /posts/the-guarantee/   # submit just these paths
//
// Google does not take IndexNow; for Google use Search Console (website/README.md, "Search engines").
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const site = 'https://jevstiller.pages.dev';
const host = new URL(site).host;
const publicDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public');

const keyFile = readdirSync(publicDir).find((f) => /^[0-9a-f]{32}\.txt$/.test(f));
if (!keyFile) throw new Error('no IndexNow key file (public/<32 hex>.txt)');
const key = readFileSync(join(publicDir, keyFile), 'utf8').trim();
if (key !== keyFile.slice(0, -4)) throw new Error(`key file ${keyFile} does not contain its own name`);

async function sitemapUrls() {
	const index = await (await fetch(`${site}/sitemap-index.xml`)).text();
	const parts = [...index.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
	const urls = [];
	for (const part of parts) {
		const xml = await (await fetch(part)).text();
		urls.push(...[...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]));
	}
	return urls;
}

const args = process.argv.slice(2);
const urlList = args.length ? args.map((p) => new URL(p, site).href) : await sitemapUrls();
if (!urlList.length) throw new Error('nothing to submit');

// The engines check that the key file is reachable before accepting the batch.
const check = await fetch(`${site}/${keyFile}`);
if (!check.ok || (await check.text()).trim() !== key) {
	throw new Error(`${site}/${keyFile} is not live yet (HTTP ${check.status}); deploy first`);
}

const res = await fetch('https://api.indexnow.org/indexnow', {
	method: 'POST',
	headers: { 'content-type': 'application/json; charset=utf-8' },
	body: JSON.stringify({ host, key, keyLocation: `${site}/${keyFile}`, urlList }),
});
console.log(`IndexNow: HTTP ${res.status} for ${urlList.length} URLs`);
if (res.status === 200 || res.status === 202) {
	for (const u of urlList) console.log('  ' + u);
} else {
	console.log(await res.text());
	process.exit(1);
}
