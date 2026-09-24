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
			description:
				'Distill a repeated Jev classification task into a local model, on the fly. Same answers, your hardware.',
			social: [{ icon: 'github', label: 'GitHub', href: repo }],
			editLink: { baseUrl: `${repo}/edit/main/website/` },
			customCss: ['./src/styles/custom.css'],
			lastUpdated: true,
			sidebar: [
				{ label: 'Getting started', slug: 'getting-started' },
				{
					label: 'Concepts',
					items: [
						{ label: 'How it works', slug: 'concepts/how-it-works' },
						{ label: 'The guarantee', slug: 'concepts/guarantee' },
						{ label: 'When to use it', slug: 'concepts/when-to-use' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'Python API', slug: 'reference/python-api' },
						{ label: 'Configuration', slug: 'reference/configuration' },
					],
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
