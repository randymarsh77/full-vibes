import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';

export default function About() {
	return (
		<div className="bg-gradient-to-br from-vibe-dark to-vibe-darker min-h-screen">
			<Head>
				<title>About | Full Vibes</title>
				<meta
					name="description"
					content="Learn about Full Vibes and our approach to coding with AI"
				/>
			</Head>

			<header className="py-6 px-4 md:px-6 lg:px-8 backdrop-blur-sm bg-vibe-darker/70 sticky top-0 z-10">
				<div className="container mx-auto flex justify-between items-center">
					<div className="font-display font-bold text-2xl">
						<Link href="/">
							<span className="cursor-pointer bg-clip-text text-transparent bg-gradient-to-r from-vibe-pink to-vibe-blue">
								Full Vibes
							</span>
						</Link>
					</div>
					<nav>
						<ul className="flex space-x-6 font-medium">
							<li>
								<Link href="/" className="text-vibe-light hover:text-vibe-pink transition">
									Home
								</Link>
							</li>
							<li>
								<Link href="/blog" className="text-vibe-light hover:text-vibe-pink transition">
									Blog
								</Link>
							</li>
							<li>
								<Link href="/about" className="text-vibe-light hover:text-vibe-pink transition">
									About
								</Link>
							</li>
						</ul>
					</nav>
				</div>
			</header>

			<main className="container mx-auto px-4 md:px-6 lg:px-8 py-12">
				<div className="max-w-3xl mx-auto">
					<h1 className="text-4xl font-bold font-display text-white mb-8 text-center">
						About Full Vibes
					</h1>

					<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-8 mb-12 border border-white/5 shadow-lg shadow-vibe-purple/10">
						<div className="flex flex-col md:flex-row items-center mb-8">
							<div className="md:w-1/3 mb-6 md:mb-0">
								<div className="relative w-48 h-48 mx-auto">
									<Image
										src="https://images.unsplash.com/photo-1531297484001-80022131f5a1"
										alt="About Full Vibes"
										fill
										className="rounded-full object-cover border-2 border-vibe-purple/30"
									/>
								</div>
							</div>
							<div className="md:w-2/3 md:pl-8">
								<h2 className="text-2xl font-bold font-display text-white mb-4">Our Mission</h2>
								<p className="text-vibe-gray mb-4">
									At Full Vibes, we believe that coding should be as enjoyable as it is productive.
									We're committed to exploring how AI can enhance creativity, streamline workflows,
									and help developers maintain positive energy throughout their projects.
								</p>
								<p className="text-vibe-gray">
									Our blog is dedicated to sharing insights, tutorials, and reflections on the
									evolving relationship between human creativity and AI assistance in software
									development.
								</p>
							</div>
						</div>

						<h2 className="text-2xl font-bold font-display text-white mb-4">What We Cover</h2>
						<ul className="text-vibe-gray mb-8 space-y-2">
							<li className="flex items-start">
								<span className="text-vibe-pink mr-2">✓</span>
								<span>AI-assisted coding techniques and best practices</span>
							</li>
							<li className="flex items-start">
								<span className="text-vibe-pink mr-2">✓</span>
								<span>Creative coding projects and inspiration</span>
							</li>
							<li className="flex items-start">
								<span className="text-vibe-pink mr-2">✓</span>
								<span>The philosophy of human-AI collaboration</span>
							</li>
							<li className="flex items-start">
								<span className="text-vibe-pink mr-2">✓</span>
								<span>Mental wellness and maintaining good vibes while coding</span>
							</li>
							<li className="flex items-start">
								<span className="text-vibe-pink mr-2">✓</span>
								<span>Emerging trends in AI and development tools</span>
							</li>
						</ul>

						<h2 className="text-2xl font-bold font-display text-white mb-4">Get in Touch</h2>
						<p className="text-vibe-gray mb-6">
							Have questions, ideas, or just want to connect? We'd love to hear from you!
						</p>
						<a
							href="mailto:hello@fullvibes.dev"
							className="inline-block bg-gradient-to-r from-vibe-pink to-vibe-blue px-6 py-3 rounded-lg text-white font-medium hover:opacity-90 transition"
						>
							Email Us
						</a>
					</div>
				</div>
			</main>

			<footer className="bg-vibe-darker text-center py-8 text-vibe-gray border-t border-white/5">
				<div className="container mx-auto px-4">
					<p className="font-mono text-sm">
						© {new Date().getFullYear()} Full Vibes. All rights reserved.
					</p>
				</div>
			</footer>
		</div>
	);
}
