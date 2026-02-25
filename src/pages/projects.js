import { useState } from 'react';
import Layout from '../components/Layout';

const categories = [
	'Audio & Media',
	'Developer Tools',
	'Web & Interactive',
	'Swift Libraries',
	'Games',
];

const projects = [
	// Audio & Media
	{
		name: 'Crystal',
		description: 'Low latency, synchronized, live audio streaming in Swift.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/crystal',
		stars: 6,
		category: 'Audio & Media',
	},
	{
		name: 'Amethyst',
		description: 'A home audio solution.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/amethyst',
		stars: 2,
		category: 'Audio & Media',
	},
	{
		name: 'Spectrum',
		description: 'Audio samples in. Frequency domain out.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/spectrum',
		stars: 1,
		category: 'Audio & Media',
	},

	// Developer Tools
	{
		name: 'NuGet Extensions',
		description: 'Extensions for the NuGet CLI.',
		language: 'TypeScript',
		url: 'https://github.com/randymarsh77/nuget-extensions',
		stars: 3,
		category: 'Developer Tools',
	},
	{
		name: 'Harvest',
		description: 'A GitHub App for auto-scaling, ephemeral macOS runners.',
		language: 'TypeScript',
		url: 'https://github.com/randymarsh77/harvest',
		stars: 1,
		category: 'Developer Tools',
	},
	{
		name: 'FaithlifeBuildTasks',
		description: 'Contributes VSCode tasks from Faithlife.Build targets.',
		language: 'TypeScript',
		url: 'https://github.com/randymarsh77/FaithlifeBuildTasks',
		stars: 1,
		category: 'Developer Tools',
	},
	{
		name: 'dotnet-assets',
		description: 'Dependency tracking tool for the .NET ecosystem based on project.assets.json files.',
		language: 'C#',
		url: 'https://github.com/randymarsh77/dotnet-assets',
		category: 'Developer Tools',
	},
	{
		name: 'yalcspace',
		description: 'Generate VSCode workspaces from yalc links.',
		language: 'TypeScript',
		url: 'https://github.com/randymarsh77/yalcspace',
		category: 'Developer Tools',
	},
	{
		name: 'tsa-cli',
		description: 'Stats for your nodes.',
		language: 'TypeScript',
		url: 'https://github.com/randymarsh77/tsa-cli',
		category: 'Developer Tools',
	},
	{
		name: 'swiftx',
		description: 'npm-like extensions (and more) for the Swift CLI.',
		language: 'JavaScript',
		url: 'https://github.com/randymarsh77/swiftx',
		category: 'Developer Tools',
	},
	{
		name: 'git-deploy-gh-pages',
		description: 'Yet another script to deploy to gh-pages.',
		language: 'JavaScript',
		url: 'https://github.com/randymarsh77/git-deploy-gh-pages',
		category: 'Developer Tools',
	},
	{
		name: 'static-nix-cache',
		description: 'Nix binary cache.',
		language: 'JavaScript',
		url: 'https://github.com/randymarsh77/static-nix-cache',
		category: 'Developer Tools',
	},

	// Web & Interactive
	{
		name: 'tui2web',
		description: 'A TUI emulator that runs on the web.',
		language: 'Rust',
		url: 'https://github.com/randymarsh77/tui2web',
		category: 'Web & Interactive',
	},
	{
		name: 'Hunky',
		description: 'Live stream your git diffs.',
		language: 'Rust',
		url: 'https://github.com/randymarsh77/hunky',
		category: 'Web & Interactive',
	},
	{
		name: 'wkinterop',
		description: 'Swift plus JavaScript via WKWebView.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/wkinterop',
		category: 'Web & Interactive',
	},
	{
		name: 'wkinteropJS',
		description: 'JS counterpart to wkinterop, a Swift package for talking to WKWebViews.',
		language: 'TypeScript',
		url: 'https://github.com/randymarsh77/wkinteropJS',
		category: 'Web & Interactive',
	},
	{
		name: 'Fractals',
		description: 'React app to explore fractals.',
		language: 'JavaScript',
		url: 'https://github.com/randymarsh77/fractals',
		category: 'Web & Interactive',
	},

	// Swift Libraries
	{
		name: 'Time',
		description: 'A small utility to handle mach_absolute_time and timestamps.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/time',
		stars: 1,
		category: 'Swift Libraries',
	},
	{
		name: 'Mesh',
		description: 'p2p mesh in Swift.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/mesh',
		stars: 1,
		category: 'Swift Libraries',
	},
	{
		name: 'Bonjour',
		description: 'Convenience task-based API around NetServices.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/bonjour',
		category: 'Swift Libraries',
	},
	{
		name: 'Fetch',
		description: 'Fetch data from a URL. Swift. Async.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/fetch',
		category: 'Swift Libraries',
	},
	{
		name: 'Streams',
		description: 'Data in. Data out. Models for data flow in Swift.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/streams',
		category: 'Swift Libraries',
	},
	{
		name: 'Scope',
		description: 'Facilitates cleanup of arbitrary resources, handling multi-ownership concerns.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/scope',
		category: 'Swift Libraries',
	},
	{
		name: 'Promise',
		description: 'Not your average JS promise.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/promise',
		category: 'Swift Libraries',
	},
	{
		name: 'Cancellation',
		description: '.NET CancellationToken[Source]s.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/cancellation',
		category: 'Swift Libraries',
	},
	{
		name: 'bigmath',
		description: 'Swift native numerics for arbitrary precision using MPFR and GMP.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/bigmath',
		category: 'Swift Libraries',
	},
	{
		name: 'Async',
		description: "Swift implementation of C#'s async/await, updated to Swift 3 and provided as a package.",
		language: 'Swift',
		url: 'https://github.com/randymarsh77/async',
		category: 'Swift Libraries',
	},
	{
		name: 'AsyncProcess',
		description: 'A wrapper around Process, providing some async niceness.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/asyncprocess',
		category: 'Swift Libraries',
	},
	{
		name: 'PubSubCache',
		description: 'A protocol for a cache with pubsub.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/pubsubcache',
		category: 'Swift Libraries',
	},
	{
		name: 'Cast',
		description: 'Convenience functions to cast pointers and avoid the verbose syntax of Swift 3.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/cast',
		category: 'Swift Libraries',
	},
	{
		name: 'Using',
		description: 'C# using statement.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/using',
		category: 'Swift Libraries',
	},

	// Games
	{
		name: 'Spyfall',
		description: 'Cryptographic version of playing Spyfall.',
		language: 'Rust',
		url: 'https://github.com/randymarsh77/spyfall',
		category: 'Games',
	},
	{
		name: 'wreck-it',
		description: 'Ralph coming in hot.',
		language: 'Rust',
		url: 'https://github.com/randymarsh77/wreck-it',
		category: 'Games',
	},
];

const languageColors = {
	Swift: 'bg-orange-500',
	Rust: 'bg-amber-700',
	TypeScript: 'bg-blue-500',
	JavaScript: 'bg-yellow-500',
	'C#': 'bg-green-600',
};

export default function Projects() {
	const [activeCategory, setActiveCategory] = useState(null);

	const filteredProjects = activeCategory
		? projects.filter((p) => p.category === activeCategory)
		: projects;

	const groupedProjects = categories
		.filter((cat) => filteredProjects.some((p) => p.category === cat))
		.map((cat) => ({
			name: cat,
			projects: filteredProjects.filter((p) => p.category === cat),
		}));

	return (
		<Layout
			title="Projects"
			description="Open source projects and software built by Full Vibes Dev — from real-time audio streaming to developer tools."
		>
			<h1 className="text-4xl font-bold font-display text-white mb-4 text-center">Projects</h1>
			<p className="text-vibe-gray text-center mb-8 max-w-2xl mx-auto">
				Open source projects and tools. View more on{' '}
				<a
					href="https://github.com/randymarsh77"
					target="_blank"
					rel="noopener noreferrer"
					className="text-vibe-blue hover:text-vibe-pink transition"
				>
					GitHub
				</a>
				.
			</p>

			<div className="flex flex-wrap justify-center gap-3 mb-12">
				<button
					onClick={() => setActiveCategory(null)}
					className={`px-4 py-2 rounded-full text-sm font-medium transition ${
						activeCategory === null
							? 'bg-vibe-purple text-white'
							: 'bg-vibe-dark/60 text-vibe-gray border border-white/10 hover:border-vibe-purple/30'
					}`}
				>
					All
				</button>
				{categories.map((cat) => (
					<button
						key={cat}
						onClick={() => setActiveCategory(activeCategory === cat ? null : cat)}
						className={`px-4 py-2 rounded-full text-sm font-medium transition ${
							activeCategory === cat
								? 'bg-vibe-purple text-white'
								: 'bg-vibe-dark/60 text-vibe-gray border border-white/10 hover:border-vibe-purple/30'
						}`}
					>
						{cat}
					</button>
				))}
			</div>

			{groupedProjects.map((group) => (
				<div key={group.name} className="mb-12">
					<h2 className="text-2xl font-bold font-display text-white mb-6">{group.name}</h2>
					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
						{group.projects.map((project) => (
							<a
								key={project.name}
								href={project.url}
								target="_blank"
								rel="noopener noreferrer"
								className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-6 border border-white/5 shadow-lg shadow-vibe-purple/10 hover:border-vibe-purple/30 hover:transform hover:scale-105 transition duration-300 flex flex-col"
							>
								<div className="flex items-center justify-between mb-3">
									<h3 className="text-xl font-bold font-display text-white">{project.name}</h3>
									{project.stars > 0 && (
										<span className="text-vibe-gray text-sm">⭐ {project.stars}</span>
									)}
								</div>
								<p className="text-vibe-gray mb-4 flex-grow">{project.description}</p>
								<div className="flex items-center gap-2">
									<span
										className={`inline-block w-3 h-3 rounded-full ${languageColors[project.language] || 'bg-vibe-gray'}`}
									></span>
									<span className="text-vibe-gray text-sm">{project.language}</span>
								</div>
							</a>
						))}
					</div>
				</div>
			))}
		</Layout>
	);
}
