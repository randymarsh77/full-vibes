import Link from 'next/link';
import Layout from '../components/Layout';

export default function Home() {
	return (
		<Layout
			title="AI-Powered Engineering Teams"
			description="Full Vibes Dev — We deploy complete, scalable AI-powered engineering teams to manage your projects from start to finish."
		>
			<section className="text-center mb-16">
				<h1 className="text-4xl md:text-6xl font-bold font-display text-white mb-4">
					AI-Powered Engineering Teams with{' '}
					<span className="bg-clip-text text-transparent bg-gradient-to-r from-vibe-pink to-vibe-blue">
						Full Vibes
					</span>
				</h1>
				<p className="text-xl text-vibe-gray max-w-2xl mx-auto font-light">
					We deploy complete, scalable AI-powered engineering teams to manage your
					projects — from architecture to production.
				</p>
				<div className="mt-8 flex flex-col sm:flex-row justify-center gap-4">
					<Link
						href="/projects"
						className="inline-block bg-gradient-to-r from-vibe-pink to-vibe-blue px-8 py-3 rounded-lg text-white font-medium hover:opacity-90 transition"
					>
						View Projects
					</Link>
					<Link
						href="/contact"
						className="inline-block border border-vibe-blue px-8 py-3 rounded-lg text-vibe-blue font-medium hover:bg-vibe-blue/10 transition"
					>
						Get in Touch
					</Link>
				</div>
			</section>

			<section className="mb-16">
				<h2 className="text-3xl font-bold font-display text-white mb-8 text-center">
					What We Do
				</h2>
				<div className="grid grid-cols-1 md:grid-cols-3 gap-8">
					<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-8 border border-white/5 shadow-lg shadow-vibe-purple/10 text-center">
						<div className="text-4xl mb-4">🤖</div>
						<h3 className="text-xl font-bold font-display text-white mb-3">AI Engineering Teams</h3>
						<p className="text-vibe-gray">
							Complete, autonomous engineering teams powered by AI — ready to take on your
							projects and ship production-quality code.
						</p>
					</div>
					<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-8 border border-white/5 shadow-lg shadow-vibe-purple/10 text-center">
						<div className="text-4xl mb-4">📈</div>
						<h3 className="text-xl font-bold font-display text-white mb-3">
							Scalable by Design
						</h3>
						<p className="text-vibe-gray">
							Scale your engineering capacity on demand. Our AI-powered teams grow with
							your needs — no hiring bottlenecks, no ramp-up time.
						</p>
					</div>
					<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-8 border border-white/5 shadow-lg shadow-vibe-purple/10 text-center">
						<div className="text-4xl mb-4">🚀</div>
						<h3 className="text-xl font-bold font-display text-white mb-3">
							End-to-End Project Management
						</h3>
						<p className="text-vibe-gray">
							From planning and architecture to deployment and maintenance — we manage
							the full lifecycle of your software projects.
						</p>
					</div>
				</div>
			</section>

			<section className="bg-vibe-dark/40 backdrop-blur-lg rounded-xl p-8 text-center border border-white/5 shadow-lg shadow-vibe-purple/10">
				<h2 className="text-3xl font-bold font-display text-white mb-4">
					Let&apos;s Build Something Great
				</h2>
				<p className="text-vibe-gray mb-6 max-w-2xl mx-auto">
					Ready to deploy an AI-powered engineering team for your project? Let&apos;s talk.
				</p>
				<Link
					href="/contact"
					className="inline-block bg-gradient-to-r from-vibe-pink to-vibe-blue px-8 py-3 rounded-lg text-white font-medium hover:opacity-90 transition"
				>
					Contact Us
				</Link>
			</section>
		</Layout>
	);
}
