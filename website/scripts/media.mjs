// Renders the README's pictures into docs/media as plain SVG (no webfonts, transparent background), one light and
// one dark version of each, so GitHub can pick the right one with <picture>:
//   flow-{light,dark}.svg      the drop-in proxy: services -> Jevstiller -> local model / Jev
//   loop-{light,dark}.svg      how it learns: router, audit, sample store, train, shadow, promote
//   results-{light,dark}.svg   the Banking77 replay against live Jev (src/data/banking77-live.json)
// Run: npm run media
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const out = join(root, 'docs', 'media');
const data = JSON.parse(readFileSync(join(root, 'website', 'src', 'data', 'banking77-live.json'), 'utf8'));

const THEMES = {
	light: { ink: '#121c1b', ink2: '#435553', muted: '#748684', line: '#c9d6d4', grid: '#e6ecec', panel: '#f3f7f6', surface: '#ffffff',
		accent: '#137572', accentWash: '#e2f2f0', violet: '#6b4fb3', amber: '#b8700f', dot: '#121c1b' },
	dark: { ink: '#e8f2f0', ink2: '#a9bfbc', muted: '#6f8d89', line: '#2a4744', grid: '#1f3634', panel: '#132625', surface: '#0d1117',
		accent: '#3ec4bd', accentWash: '#123634', violet: '#b79cf0', amber: '#e8b04a', dot: '#e8f2f0' },
};
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace";
const SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
const MARK_OUTER = 'M32 4 C24 18 12 30 12 42 A20 20 0 0 0 52 42 C52 30 40 18 32 4 Z';
const MARK_INNER = 'M32 25 C28.5 31.5 23 36.5 23 42 A9 9 0 0 0 41 42 C41 36.5 35.5 31.5 32 25 Z';

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const text = (x, y, s, { size = 12, fill, weight = 400, anchor = 'start', font = MONO } = {}) =>
	`<text x="${x}" y="${y}" font-family="${font}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}">${esc(s)}</text>`;
const box = (x, y, w, h, t, { fill, stroke, r = 6, sw = 1.2 } = {}) =>
	`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
const svg = (w, h, body, viewBox = `0 0 ${w} ${h}`) =>
	`<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" width="${w}" height="${h}" role="img">\n${body}\n</svg>\n`;
// a point on a cubic bezier, for placing things along a lane
const bez = (p0, p1, p2, p3, t) => {
	const u = 1 - t;
	return [0, 1].map((i) => u * u * u * p0[i] + 3 * u * u * t * p1[i] + 3 * u * t * t * p2[i] + t * t * t * p3[i]);
};

// ── the drop-in proxy ────────────────────────────────────────────────────────────────────────────────────────────
function flow(t) {
	const lane = (d, dashed) => `<path d="${d}" fill="none" stroke="${t.line}" stroke-width="1.5"${dashed ? ' stroke-dasharray="3 5"' : ''}/>`;
	const dot = (x, y, slow) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4.5" fill="${slow ? t.amber : t.dot}"/>`;
	const parts = [
		lane('M150 150 C 260 150, 300 150, 372 150'),
		lane('M488 150 C 560 150, 600 116, 690 116'),
		lane('M488 150 C 560 150, 600 196, 690 196', true),
		box(20, 112, 130, 76, null, { fill: t.panel, stroke: t.line }),
		text(85, 145, 'your services', { size: 14, weight: 600, fill: t.ink, anchor: 'middle' }),
		text(85, 166, 'unchanged SDK', { size: 11.5, fill: t.muted, anchor: 'middle' }),
		`<g transform="translate(366 60) scale(2.05)"><path d="${MARK_OUTER}" fill="${t.accent}" fill-rule="evenodd"/><path d="${MARK_INNER}" fill="${t.surface}"/></g>`,
		text(430, 220, 'jevstiller', { size: 14, weight: 600, fill: t.ink, anchor: 'middle' }),
		text(430, 240, 'one process · many tasks and tenants', { size: 11.5, fill: t.muted, anchor: 'middle' }),
		box(690, 80, 160, 72, null, { fill: t.accentWash, stroke: t.accent }),
		text(770, 104, 'local model', { size: 14, weight: 600, fill: t.ink, anchor: 'middle' }),
		text(770, 123, 'most requests', { size: 11.5, fill: t.muted, anchor: 'middle' }),
		text(770, 140, '~16 ms', { size: 11.5, fill: t.muted, anchor: 'middle' }),
		box(690, 160, 160, 72, null, { fill: t.panel, stroke: t.line }),
		text(770, 184, 'Jev', { size: 14, weight: 600, fill: t.ink, anchor: 'middle' }),
		text(770, 203, 'unsure, novel, audit', { size: 11.5, fill: t.muted, anchor: 'middle' }),
		text(770, 220, '~300 ms', { size: 11.5, fill: t.muted, anchor: 'middle' }),
	];
	// requests in flight: on the way in, and on the two ways out (three local for every one to Jev)
	[200, 260, 320].forEach((x) => parts.push(dot(x, 150, false)));
	parts.push(dot(345, 150, true));
	[0.3, 0.6, 0.85].forEach((u) => parts.push(dot(...bez([488, 150], [560, 150], [600, 116], [690, 116], u), false)));
	parts.push(dot(...bez([488, 150], [560, 150], [600, 196], [690, 196], 0.55), true));
	parts.push(
		`<circle cx="24" cy="279" r="4" fill="${t.dot}"/>`, text(34, 283, 'answered locally', { size: 11.5, fill: t.muted }),
		`<circle cx="164" cy="279" r="4" fill="${t.amber}"/>`, text(174, 283, 'forwarded to Jev', { size: 11.5, fill: t.muted }),
	);
	return svg(860, 250, parts.join('\n'), '0 40 860 250');
}

// ── how it learns ────────────────────────────────────────────────────────────────────────────────────────────────
function loop(t) {
	const arrow = (d, { dashed = false, color = t.line } = {}) =>
		`<path d="${d}" fill="none" stroke="${color}" stroke-width="1.5"${dashed ? ' stroke-dasharray="4 4"' : ''} marker-end="url(#${color === t.accent ? 'ah' : 'a'})"/>`;
	const node = (x, y, w, h, title, sub, hot) => [
		box(x, y, w, h, null, { fill: hot ? t.accentWash : t.panel, stroke: hot ? t.accent : t.line }),
		text(x + w / 2, y + (sub ? 24 : h / 2 + 5), title, { size: 13.5, weight: 600, fill: t.ink, anchor: 'middle' }),
		sub ? text(x + w / 2, y + 42, sub, { size: 11, fill: t.muted, anchor: 'middle' }) : '',
	].join('\n');
	const lbl = (x, y, s, anchor = 'middle', color = t.ink2) => text(x, y, s, { size: 10.5, fill: color, anchor });
	const marker = (id, color) =>
		`<marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="${color}"/></marker>`;
	const parts = [
		`<defs>${marker('a', t.line)}${marker('ah', t.accent)}</defs>`,
		// serving: request -> router -> student or Jev -> answer
		node(20, 60, 100, 56, 'request', null),
		arrow('M120 88 L168 88'),
		node(170, 60, 150, 56, 'router', 'confidence, OOD gate'),
		arrow('M320 76 C 360 76, 360 68, 400 68 L 498 68'),
		lbl(449, 60, 'confident,'), lbl(449, 82, 'known input'),
		node(500, 40, 150, 56, 'local student', 'student:vN · ~16 ms', true),
		arrow('M320 100 C 360 100, 360 158, 400 158 L 498 158'),
		lbl(449, 150, 'unsure, novel,'), lbl(449, 172, 'or the 2% audit'),
		node(500, 130, 150, 56, 'Jev', 'the teacher · ~300 ms'),
		arrow('M650 68 C 700 68, 710 88, 758 88'),
		arrow('M650 158 C 700 158, 710 88, 758 88'),
		node(760, 60, 100, 56, 'answer', null),
		// learning: Jev's answers -> sample store -> candidate -> shadow -> promote -> production
		arrow('M575 186 L575 248', { dashed: true, color: t.accent }),
		lbl(582, 214, 'answer + Jev’s distribution:', 'start'), lbl(582, 228, 'a training row', 'start'),
		node(500, 250, 150, 56, 'sample store', 'embeddings, kept'),
		arrow('M500 278 L 402 278'),
		lbl(451, 270, 'train, seconds'),
		node(250, 250, 150, 56, 'candidate', 'encoder + small head'),
		arrow('M245 116 L 322 248', { dashed: true }),
		lbl(284, 190, 'shadowed on live traffic', 'end'),
		arrow('M250 278 L 152 278'),
		lbl(201, 270, 'within budget?'),
		node(20, 250, 130, 56, 'promote', 'or fall back'),
		arrow('M85 250 L85 16 L575 16 L575 38', { dashed: true, color: t.accent }),
		lbl(330, 30, 'promote to production, or roll back'),
		lbl(20, 338, 'The audit slice keeps checking production; if agreement drops, or Jev changes, the task falls back to Jev and retrains.', 'start', t.muted),
	];
	return svg(860, 350, parts.join('\n'));
}

// ── the results chart ────────────────────────────────────────────────────────────────────────────────────────────
function results(t) {
	const W = 720, H = 372, left = 62, right = 704, top1 = 30, bottom1 = 200, top2 = 252, bottom2 = 328, xMax = 11_500;
	const x = (m) => left + (m / xMax) * (right - left);
	const y1 = (v) => bottom1 - v * (bottom1 - top1);
	const y2 = (v) => bottom2 - ((v - 0.97) / 0.03) * (bottom2 - top2);
	const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
	const fmt = (n) => n.toLocaleString('en-US');
	const pts = data.points;
	const local = pts.map((p) => [x(p.messages), y1(p.answered_locally)]);
	const localLine = [[x(0), y1(0)], ...local].map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join(' ');
	const lastL = local.at(-1);
	const agreePts = pts.filter((p) => p.agreement != null);
	const agree = agreePts.map((p) => [x(p.messages), y2(p.agreement)]);
	const lastA = agree.at(-1);
	const firstStudent = pts.find((p) => p.student).messages;
	const learnEnd = x((pts.filter((p) => !p.student).at(-1).messages + firstStudent) / 2);
	const parts = [
		`<rect x="${x(0)}" y="${top1}" width="${(learnEnd - x(0)).toFixed(1)}" height="${bottom1 - top1}" fill="${t.ink}" opacity="0.04"/>`,
		text(x(0) + 8, top1 + 16, 'learning', { size: 11, fill: t.muted }),
		text(x(0) + 8, top1 + 30, 'all to Jev', { size: 11, fill: t.muted }),
	];
	for (const v of [0, 0.25, 0.5, 0.75, 1]) {
		parts.push(`<line x1="${left}" x2="${right}" y1="${y1(v)}" y2="${y1(v)}" stroke="${t.grid}"/>`, text(left - 8, y1(v) + 4, pct(v, 0), { size: 11, fill: t.muted, anchor: 'end' }));
	}
	parts.push(
		`<polygon points="${x(0)},${y1(0)} ${localLine} ${lastL[0].toFixed(1)},${y1(0)}" fill="${t.accent}" opacity="0.12"/>`,
		`<polyline points="${localLine}" fill="none" stroke="${t.accent}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`,
		...local.map(([a, b]) => `<circle cx="${a.toFixed(1)}" cy="${b.toFixed(1)}" r="4.5" fill="${t.accent}" stroke="${t.surface}" stroke-width="2"/>`),
		text(lastL[0] - 4, lastL[1] - 14, `${pct(pts.at(-1).answered_locally)} answered locally`, { size: 12.5, weight: 700, fill: t.ink, anchor: 'end' }),
		text(left, top1 - 10, 'Answered locally, without calling Jev', { size: 12, weight: 600, fill: t.ink }),
	);
	for (const v of [0.97, 0.98, 0.99, 1]) {
		const target = v === data.target;
		parts.push(
			`<line x1="${left}" x2="${right}" y1="${y2(v)}" y2="${y2(v)}" stroke="${target ? t.amber : t.grid}"${target ? ' stroke-width="1.5" stroke-dasharray="4 4"' : ''}/>`,
			text(left - 8, y2(v) + 4, pct(v, 0), { size: 11, fill: t.muted, anchor: 'end' }),
		);
	}
	parts.push(
		text(right, y2(data.target) + 14, `target ${pct(data.target, 0)}`, { size: 11, fill: t.amber, anchor: 'end' }),
		`<polyline points="${agree.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join(' ')}" fill="none" stroke="${t.violet}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`,
		...agree.map(([a, b]) => `<circle cx="${a.toFixed(1)}" cy="${b.toFixed(1)}" r="4.5" fill="${t.violet}" stroke="${t.surface}" stroke-width="2"/>`),
		text(lastA[0] - 4, lastA[1] - 12, `${pct(agreePts.at(-1).agreement, 2)} agreement`, { size: 12.5, weight: 700, fill: t.ink, anchor: 'end' }),
		text(left, top2 - 12, `Agreement with Jev, on ${fmt(data.held_out)} held-out messages`, { size: 12, weight: 600, fill: t.ink }),
		...[0, 2500, 5000, 7500, 10000].map((m) => text(x(m), bottom2 + 20, fmt(m), { size: 11, fill: t.muted, anchor: 'middle' })),
		text((left + right) / 2, H - 6, 'messages seen', { size: 12, weight: 600, fill: t.ink, anchor: 'middle' }),
	);
	return svg(W, H, parts.join('\n'));
}

for (const [name, t] of Object.entries(THEMES)) {
	writeFileSync(join(out, `flow-${name}.svg`), flow(t));
	writeFileSync(join(out, `loop-${name}.svg`), loop(t));
	writeFileSync(join(out, `results-${name}.svg`), results(t));
}
console.log(`media: 6 files in docs/media (from ${data.messages.toLocaleString('en-US')} messages, target ${data.target})`);
