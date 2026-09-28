// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

const repo = 'https://github.com/tomerglick57/Jevstiller';

// https://astro.build/config
export default defineConfig({
	site: 'https://jevstiller.pages.dev',
	integrations: [
		starlight({
			title: 'Jevstiller',
			logo: {
				light: './src/assets/logo-light.svg',
				dark: './src/assets/logo-dark.svg',
				replacesTitle: true,
			},
			description:
				'A drop-in proxy for Jev that learns your repeated classification questions and answers them locally, within an agreement budget you set.',
			social: [{ icon: 'github', label: 'GitHub', href: repo }],
			editLink: { baseUrl: `${repo}/edit/main/website/` },
			customCss: ['./src/styles/custom.css'],
			head: [
				{ tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' } },
				{ tag: 'link', attrs: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: true } },
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap',
					},
				},
				{ tag: 'meta', attrs: { property: 'og:image', content: 'https://jevstiller.pages.dev/social-preview.png' } },
			],
			lastUpdated: true,
			sidebar: [
				{ label: 'Getting started', slug: 'getting-started' },
				{
					label: 'Concepts',
					items: [
						{ label: 'How it works', slug: 'concepts/how-it-works' },
						{ label: 'The guarantee', slug: 'concepts/guarantee' },
						{ label: 'When to use it', slug: 'concepts/when-to-use' },
						{ label: 'How it compares', slug: 'concepts/comparison' },
					],
				},
				{
					label: 'Run the proxy',
					items: [
						{ label: 'The drop-in proxy', slug: 'proxy/overview' },
						{ label: 'Deploy', slug: 'proxy/deploy' },
						{ label: 'Operations', slug: 'proxy/operations' },
						{ label: 'Security', slug: 'proxy/security' },
						{ label: 'Compatibility', slug: 'proxy/compatibility' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Configuration', slug: 'reference/configuration' },
						{ label: 'Python API', slug: 'reference/python-api' },
					],
				},
				{
					label: 'Writing',
					items: [{ label: 'A bound, not a threshold', slug: 'posts/the-guarantee' }],
				},
				{
					label: 'Project',
					items: [
						{ label: 'Benchmarks', slug: 'project/benchmarks' },
						{ label: 'Roadmap', slug: 'project/roadmap' },
						{ label: 'Contributing', slug: 'project/contributing' },
						{ label: 'Design document', link: `${repo}/blob/main/DESIGN.md`, attrs: { target: '_blank' } },
						{ label: 'Changelog', link: `${repo}/blob/main/CHANGELOG.md`, attrs: { target: '_blank' } },
					],
				},
			],
		}),
	],
});
