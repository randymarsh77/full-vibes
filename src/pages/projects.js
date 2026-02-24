import Layout from '../components/Layout';

const projects = [
	{
		name: 'Crystal',
		description: 'Low latency, synchronized, live audio streaming in Swift.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/crystal',
		stars: 6,
	},
	{
		name: 'Harvest',
		description: 'A GitHub App for auto-scaling, ephemeral macOS runners.',
		language: 'TypeScript',
		url: 'https://github.com/randymarsh77/harvest',
		stars: 1,
	},
	{
		name: 'Hunky',
		description: 'Live stream your git diffs.',
		language: 'Rust',
		url: 'https://github.com/randymarsh77/hunky',
	},
	{
		name: 'tui2web',
		description: 'A TUI emulator that runs on the web.',
		language: 'Rust',
		url: 'https://github.com/randymarsh77/tui2web',
	},
	{
		name: 'OpenCache',
		description: 'Nix binary cache.',
		language: 'JavaScript',
		url: 'https://github.com/randymarsh77/OpenCache',
	},
	{
		name: 'Spyfall',
		description: 'Cryptographic version of playing Spyfall.',
		language: 'Rust',
		url: 'https://github.com/randymarsh77/spyfall',
	},
	{
		name: 'NuGet Extensions',
		description: 'Extensions for the NuGet CLI.',
		language: 'TypeScript',
		url: 'https://github.com/randymarsh77/nuget-extensions',
		stars: 3,
	},
	{
		name: 'dotnet-assets',
		description: 'Dependency tracking tool for the .NET ecosystem based on project.assets.json files.',
		language: 'C#',
		url: 'https://github.com/randymarsh77/dotnet-assets',
	},
	{
		name: 'yalcspace',
		description: 'Generate VSCode workspaces from yalc links.',
		language: 'TypeScript',
		url: 'https://github.com/randymarsh77/yalcspace',
	},
	{
		name: 'Fetch',
		description: 'Fetch data from a URL. Swift. Async.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/fetch',
	},
	{
		name: 'Spectrum',
		description: 'Audio samples in. Frequency domain out.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/spectrum',
	},
	{
		name: 'wkinterop',
		description: 'Swift plus JavaScript via WKWebView.',
		language: 'Swift',
		url: 'https://github.com/randymarsh77/wkinterop',
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
	return (
		<Layout
			title="Projects"
			description="Open source projects and software built by Full Vibes Dev — from real-time audio streaming to developer tools."
		>
			<h1 className="text-4xl font-bold font-display text-white mb-4 text-center">Projects</h1>
			<p className="text-vibe-gray text-center mb-12 max-w-2xl mx-auto">
				A selection of open source projects and tools. View more on{' '}
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

			<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
				{projects.map((project) => (
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
		</Layout>
	);
}
