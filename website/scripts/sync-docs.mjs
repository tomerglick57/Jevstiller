// Copies the repo's docs into the site before every build (npm's prebuild/predev hooks), so the website and
// docs/ can't drift apart: edit docs/*.md, not the generated pages (they are gitignored).
//
// Each page gets Starlight frontmatter (its H1 becomes the title), an edit link to its source file, and links
// rewritten: to another synced page -> its site URL; to any other repo file -> GitHub.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, posix } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = 'https://github.com/tomerglick57/Jevstiller';
const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const out = join(root, 'website', 'src', 'content', 'docs');

// repo file -> site slug (the generated page lives at src/content/docs/<slug>.md)
const PAGES = {
	'docs/proxy.md': 'proxy/overview',
	'docs/deploy.md': 'proxy/deploy',
	'docs/operations.md': 'proxy/operations',
	'docs/security.md': 'proxy/security',
	'docs/compatibility.md': 'proxy/compatibility',
	'docs/configuration.md': 'reference/configuration',
	'docs/benchmarks.md': 'project/benchmarks',
	'docs/coverage.md': 'concepts/coverage',
	'docs/posts/2026-09-the-guarantee.md': 'posts/the-guarantee',
	'docs/guides/cut-jev-cost.md': 'guides/cut-jev-cost',
	'docs/guides/jev-rate-limits.md': 'guides/jev-rate-limits',
	'docs/compare/stuntd.md': 'compare/stuntd',
	'docs/compare/tracer.md': 'compare/tracer',
	'docs/compare/distil-labs.md': 'compare/distil-labs',
};
// sections of the README published as pages of their own
const SECTIONS = [
	{ file: 'README.md', heading: 'How it compares', slug: 'concepts/comparison' },
	{ file: 'README.md', heading: "TypeSafe's terms", slug: 'project/terms' },
];

// public/llms-full.txt: the docs an agent needs to recommend and set up Jevstiller, as one plain-text file
// (llmstxt.org). public/llms.txt is hand-written and links here.
const LLMS_FULL = ['README.md', 'AGENTS.md', 'docs/guides/cut-jev-cost.md', 'docs/guides/jev-rate-limits.md',
	'docs/proxy.md', 'docs/deploy.md', 'docs/coverage.md', 'docs/configuration.md', 'docs/security.md',
	'docs/compare/stuntd.md', 'docs/compare/tracer.md', 'docs/compare/distil-labs.md'];

// GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens
const anchor = (heading) => heading.toLowerCase().replace(/[^a-z0-9 -]/g, '').replace(/ /g, '-');

function rewriteLinks(md, from) {
	return md.replace(/\]\((?!https?:\/\/|#|mailto:)([^)\s]+)\)/g, (_, target) => {
		const [path, frag] = target.split('#');
		const file = posix.normalize(posix.join(posix.dirname(from), path));
		// a link to a published README section (its GitHub anchor) goes to that section's page
		const section = SECTIONS.find((s) => s.file === file && frag === anchor(s.heading));
		if (section) return `](/${section.slug}/)`;
		const slug = PAGES[file] ?? SECTIONS.find((s) => s.file === file && !frag)?.slug;
		if (slug) return `](/${slug}/${frag ? `#${frag}` : ''})`;
		if (/\.(gif|png|jpe?g|svg|webp)$/i.test(file)) return `](${REPO}/raw/main/${file})`; // images: the file itself
		return `](${REPO}/blob/main/${file}${frag ? `#${frag}` : ''})`;
	});
}

function firstParagraph(md) {
	const para = md.split(/\n\s*\n/).find((p) => p.trim() && !/^[#|`>-]/.test(p.trim())) ?? '';
	const text = para.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/[`*_]/g, '').replace(/\s+/g, ' ').trim();
	return text.length > 200 ? `${text.slice(0, 197).replace(/\s+\S*$/, '')}…` : text;
}

// A source file may carry a website-only title and description in HTML comments right under its heading
// (invisible on GitHub): <!-- title: ... --> and <!-- description: ... -->. They become the page's <title> and
// meta description, which search engines and AI assistants show; the heading in the repo stays as it is.
// Without them the heading and the first paragraph are used.
const OVERRIDE = /^<!--\s*(title|description):\s*([\s\S]*?)\s*-->[ \t]*\n/gm;
function overrides(md) {
	const found = {};
	const body = md.replace(OVERRIDE, (_, key, value) => {
		found[key] = value.replace(/\s+/g, ' ').trim();
		return '';
	});
	return { ...found, body };
}

// Plain text of a Markdown fragment: links to their text, no code marks or emphasis, single spaces.
const plain = (md) => md.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/[`*_>]/g, '').replace(/\s+/g, ' ').trim();

// A "## FAQ" section whose questions are H3s becomes FAQPage structured data (schema.org), which search engines
// and AI assistants read as question-and-answer pairs. The section itself is left in the page as it is.
function faqJsonLd(body) {
	const m = body.match(/\n## FAQ\n([\s\S]*?)(?=\n## |$)/);
	if (!m) return null;
	const pairs = [...m[1].matchAll(/^### (.+)\n([\s\S]*?)(?=^### |(?![\s\S]))/gm)].map(([, q, a]) => ({
		'@type': 'Question',
		name: plain(q),
		acceptedAnswer: { '@type': 'Answer', text: plain(a) },
	}));
	if (!pairs.length) return null;
	return JSON.stringify({ '@context': 'https://schema.org', '@type': 'FAQPage', mainEntity: pairs });
}

function write(slug, heading, md, source) {
	const { title = heading, description, body } = overrides(md);
	const faq = faqJsonLd(body);
	const front = [
		'---',
		`title: ${JSON.stringify(title)}`,
		`description: ${JSON.stringify(description ?? firstParagraph(body))}`,
		...(faq ? [`head: ${JSON.stringify([{ tag: 'script', attrs: { type: 'application/ld+json' }, content: faq }])}`] : []),
		`editUrl: ${JSON.stringify(`${REPO}/edit/main/${source}`)}`,
		'---',
		'',
		`<!-- Generated from ${source} by website/scripts/sync-docs.mjs. Edit that file instead. -->`,
		'',
	].join('\n');
	const file = join(out, `${slug}.md`);
	mkdirSync(dirname(file), { recursive: true });
	writeFileSync(file, front + rewriteLinks(body, source));
}

for (const [source, slug] of Object.entries(PAGES)) {
	const md = readFileSync(join(root, source), 'utf8');
	const m = md.match(/^# (.+)\n/);
	if (!m) throw new Error(`${source}: no H1 title`);
	write(slug, m[1].trim(), md.slice(m[0].length).trimStart(), source);
}

for (const { file, heading, slug } of SECTIONS) {
	const md = readFileSync(join(root, file), 'utf8');
	const start = md.indexOf(`\n## ${heading}\n`);
	if (start < 0) throw new Error(`${file}: no "## ${heading}" section`);
	const rest = md.slice(start + heading.length + 5);
	const end = rest.search(/\n## /);
	write(slug, heading, (end < 0 ? rest : rest.slice(0, end)).trim() + '\n', file);
}

const full = LLMS_FULL.map((f) => {
	const md = readFileSync(join(root, f), 'utf8').replace(/<picture>[\s\S]*?<\/picture>\n*/g, '').replace(OVERRIDE, '');
	return `<!-- ${REPO}/blob/main/${f} -->\n\n${md.trim()}\n`;
});
writeFileSync(join(root, 'website', 'public', 'llms-full.txt'), full.join('\n\n'));

console.log(`sync-docs: ${Object.keys(PAGES).length + SECTIONS.length} pages and llms-full.txt from the repo`);
