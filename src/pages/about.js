import Layout from '../components/Layout';

export default function About() {
	return (
		<Layout
			title="About"
			description="Learn about Full Vibes Dev — deploying complete, scalable AI-powered engineering teams to manage your projects."
		>
			<div className="max-w-3xl mx-auto">
				<h1 className="text-4xl font-bold font-display text-white mb-8 text-center">
					About Full Vibes Dev
				</h1>

				<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-8 mb-12 border border-white/5 shadow-lg shadow-vibe-purple/10">
					<h2 className="text-2xl font-bold font-display text-white mb-4">Who We Are</h2>
					<p className="text-vibe-gray mb-4">
						Full Vibes Dev deploys complete, scalable AI-powered engineering teams to
						manage your projects. We combine deep technical expertise with cutting-edge
						AI to deliver production-ready software — from architecture and planning
						through deployment and maintenance.
					</p>
					<p className="text-vibe-gray mb-8">
						We believe the future of software engineering is AI-augmented teams that
						move faster, scale effortlessly, and deliver consistent quality. Every
						project we take on gets a dedicated team built to match its unique needs.
					</p>

					<h2 className="text-2xl font-bold font-display text-white mb-4">Capabilities</h2>
					<ul className="text-vibe-gray mb-8 space-y-2">
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>AI-powered engineering teams that ship production-quality code</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>Full project lifecycle management — from planning to deployment</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>Scalable capacity that grows with your needs — no hiring delays</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>Web applications, APIs, native apps, and cloud infrastructure</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>DevOps, CI/CD, and automated deployment pipelines</span>
						</li>
						<li className="flex items-start">
							<span className="text-vibe-pink mr-2">✓</span>
							<span>Deep expertise across Swift, Rust, TypeScript, and more</span>
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
