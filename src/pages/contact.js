import Layout from '../components/Layout';

export default function Contact() {
	return (
		<Layout
			title="Contact"
			description="Get in touch with Full Vibes Dev to deploy an AI-powered engineering team for your project."
		>
			<div className="max-w-3xl mx-auto">
				<h1 className="text-4xl font-bold font-display text-white mb-8 text-center">
					Get in Touch
				</h1>

				<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-8 mb-12 border border-white/5 shadow-lg shadow-vibe-purple/10">
					<p className="text-vibe-gray mb-8 text-lg text-center">
						Ready to deploy an AI-powered engineering team? We&apos;d love to hear about your project.
					</p>

					<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
						<a
							href="mailto:hello@fullvibes.dev"
							className="flex items-center gap-4 bg-vibe-darker/60 rounded-xl p-6 border border-white/5 hover:border-vibe-pink/30 transition group"
						>
							<div className="text-3xl">📧</div>
							<div>
								<h3 className="text-lg font-bold font-display text-white group-hover:text-vibe-pink transition">
									Email
								</h3>
								<p className="text-vibe-gray text-sm">hello@fullvibes.dev</p>
							</div>
						</a>

						<a
							href="https://github.com/randymarsh77"
							target="_blank"
							rel="noopener noreferrer"
							className="flex items-center gap-4 bg-vibe-darker/60 rounded-xl p-6 border border-white/5 hover:border-vibe-blue/30 transition group"
						>
							<div className="text-3xl">💻</div>
							<div>
								<h3 className="text-lg font-bold font-display text-white group-hover:text-vibe-blue transition">
									GitHub
								</h3>
								<p className="text-vibe-gray text-sm">github.com/randymarsh77</p>
							</div>
						</a>
					</div>

					<div className="mt-12 text-center">
						<h2 className="text-2xl font-bold font-display text-white mb-4">
							What to Expect
						</h2>
						<div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
							<div className="text-center">
								<div className="text-2xl mb-2">1️⃣</div>
								<h3 className="text-white font-bold mb-1">Reach Out</h3>
								<p className="text-vibe-gray text-sm">
									Tell us about your project and what you need built.
								</p>
							</div>
							<div className="text-center">
								<div className="text-2xl mb-2">2️⃣</div>
								<h3 className="text-white font-bold mb-1">Design Your Team</h3>
								<p className="text-vibe-gray text-sm">
									We&apos;ll scope your project and assemble the right AI-powered engineering team.
								</p>
							</div>
							<div className="text-center">
								<div className="text-2xl mb-2">3️⃣</div>
								<h3 className="text-white font-bold mb-1">Ship</h3>
								<p className="text-vibe-gray text-sm">
									Your team gets to work — managing, building, and delivering with full vibes.
								</p>
							</div>
						</div>
					</div>
				</div>
			</div>
		</Layout>
	);
}
