import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import { getSortedPostsData } from '../lib/posts';

export default function Home({ allPostsData }) {
	return (
		<div className="bg-gradient-to-br from-vibe-dark to-vibe-darker min-h-screen">
			<Head>
				<title>Full Vibes - AI Coding with Good Vibes</title>
				<meta name="description" content="A blog about coding with AI and keeping the vibes high" />
				<link rel="icon" href="/favicon.ico" />
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
				<section className="text-center mb-16">
					<h1 className="text-4xl md:text-6xl font-bold font-display text-white mb-4">
						Coding with{' '}
						<span className="bg-clip-text text-transparent bg-gradient-to-r from-vibe-pink to-vibe-blue">
							Immaculate Vibes
						</span>
					</h1>
					<p className="text-xl text-vibe-gray max-w-2xl mx-auto font-light">
						Explore the intersection of AI, creativity, and coding. Let's build beautiful things
						while keeping the vibes high.
					</p>
				</section>

				<section className="mb-16">
					<h2 className="text-3xl font-bold font-display text-white mb-8 text-center">
						Latest Posts
					</h2>
					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
						{allPostsData.map(({ id, date, title, excerpt, coverImage }) => (
							<Link key={id} href={`/posts/${id}`}>
								<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl overflow-hidden hover:transform hover:scale-105 transition duration-300 h-full flex flex-col border border-white/5 shadow-lg shadow-vibe-purple/10">
									<div className="relative h-48">
										<Image
											src={
												coverImage || 'https://images.unsplash.com/photo-1550745165-9bc0b252726f'
											}
											alt={title}
											fill
											className="object-cover"
											unoptimized={true} // This ensures images work on Netlify
										/>
									</div>
									<div className="p-6 flex-grow">
										<p className="text-vibe-pink text-sm mb-2 font-mono">{date}</p>
										<h3 className="text-xl font-bold font-display text-white mb-2">{title}</h3>
										<p className="text-vibe-gray">{excerpt}</p>
									</div>
									<div className="p-6 pt-0">
										<span className="text-vibe-blue hover:text-vibe-pink transition font-medium">
											Read more →
										</span>
									</div>
								</div>
							</Link>
						))}
					</div>
				</section>

				<section className="bg-vibe-dark/40 backdrop-blur-lg rounded-xl p-8 text-center border border-white/5 shadow-lg shadow-vibe-purple/10">
					<h2 className="text-3xl font-bold font-display text-white mb-4">Join the Vibe Tribe</h2>
					<p className="text-vibe-gray mb-6 max-w-2xl mx-auto">
						Subscribe to get the latest in AI coding techniques, creative coding projects, and
						maintain immaculate vibes in your development workflow.
					</p>
					<form className="max-w-md mx-auto">
						<div className="flex">
							<input
								type="email"
								placeholder="Enter your email"
								className="flex-grow px-4 py-2 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-vibe-pink bg-vibe-darker/80 text-vibe-light border border-white/10"
							/>
							<button
								type="submit"
								className="bg-gradient-to-r from-vibe-pink to-vibe-blue px-6 py-2 rounded-r-lg text-white font-medium hover:opacity-90 transition"
							>
								Subscribe
							</button>
						</div>
					</form>
				</section>
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

export async function getStaticProps() {
	const allPostsData = getSortedPostsData();
	return {
		props: {
			allPostsData,
		},
	};
}
