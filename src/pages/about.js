import Layout from '../components/Layout';

export default function About() {
	return (
		<Layout
			title="About"
			description="Learn about Full Vibes Dev — a software development and contracting business building high-quality solutions."
		>
			<div className="max-w-3xl mx-auto">
				<h1 className="text-4xl font-bold font-display text-white mb-8 text-center">
					About Full Vibes Dev
				</h1>

				<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-8 mb-12 border border-white/5 shadow-lg shadow-vibe-purple/10">
					<h2 className="text-2xl font-bold font-display text-white mb-4">Who We Are</h2>
					<p className="text-vibe-gray mb-4">
						Full Vibes Dev is a software development and contracting business focused on
						building high-quality, performant software. We bring deep expertise across the
						stack — from low-level systems programming and native app development to modern
						web applications and developer tooling.
					</p>
					<p className="text-vibe-gray mb-8">
						We believe great software comes from combining strong engineering fundamentals
						with a passion for craft. Every project we take on gets the same attention to
						detail and commitment to quality.
					</p>

					<h2 className="text-2xl font-bold font-display text-white mb-4">Expertise</h2>
					<ul className="text-vibe-gray mb-8 space-y-2">
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>Native application development (Swift, iOS/macOS)</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>Systems programming (Rust, C/C++)</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>Web applications and services (TypeScript, React, Next.js, Node.js)</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>Developer tools, CLIs, and build infrastructure</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>DevOps, CI/CD, and Nix-based reproducible environments</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>Real-time audio/media streaming</span>
						</li>
					</ul>

					<h2 className="text-2xl font-bold font-display text-white mb-4">Open Source</h2>
					<p className="text-vibe-gray mb-4">
						We&apos;re active contributors to the open source community. Many of our libraries
						and tools are available on{' '}
						<a
							href="https://github.com/randymarsh77"
							target="_blank"
							rel="noopener noreferrer"
							className="text-vibe-blue hover:text-vibe-pink transition"
						>
							GitHub
						</a>
						, covering everything from real-time audio streaming to developer productivity
						tools.
					</p>
				</div>
			</div>
		</Layout>
	);
}
