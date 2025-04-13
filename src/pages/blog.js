import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import { getSortedPostsData } from '../lib/posts';
import SubscribeForm from '../components/SubscribeForm';

export default function Blog({ allPostsData }) {
	return (
		<div className="bg-gradient-to-br from-vibe-dark to-vibe-darker min-h-screen">
			<Head>
				<title>Blog | Full Vibes</title>
				<meta
					name="description"
					content="Read our latest articles about AI coding with good vibes"
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
				<h1 className="text-4xl font-bold font-display text-white mb-12 text-center">Blog</h1>

				<div className="grid grid-cols-1 md:grid-cols-2 gap-12">
					{allPostsData.map(({ id, date, title, excerpt, coverImage }, index) => (
						<article
							key={id}
							className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl overflow-hidden hover:transform hover:scale-105 transition duration-300 border border-white/5 shadow-lg shadow-vibe-purple/10"
						>
							<Link href={`/posts/${id}`}>
								<div className="relative h-64">
									<Image
										src={coverImage || 'https://images.unsplash.com/photo-1550745165-9bc0b252726f'}
										alt={title}
										fill
										className="object-cover"
										sizes="(max-width: 768px) 100vw, 50vw"
										priority={index === 0} // Prioritize loading the first image
										placeholder="blur"
										blurDataURL="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
									/>
								</div>
								<div className="p-6">
									<p className="text-vibe-pink text-sm mb-2 font-mono">{date}</p>
									<h2 className="text-2xl font-bold font-display text-white mb-3">{title}</h2>
									<p className="text-vibe-gray mb-4">{excerpt}</p>
									<span className="text-vibe-blue hover:text-vibe-pink transition font-medium">
										Read more →
									</span>
								</div>
							</Link>
						</article>
					))}
				</div>

				<div className="mt-16 bg-vibe-dark/40 backdrop-blur-lg rounded-xl p-8 text-center border border-white/5 shadow-lg shadow-vibe-purple/10">
					<h2 className="text-3xl font-bold font-display text-white mb-4">Stay in the Loop</h2>
					<p className="text-vibe-gray mb-6 max-w-2xl mx-auto">
						Get notified when we publish new articles about AI coding and creative development.
					</p>
					<SubscribeForm />
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

export async function getStaticProps() {
	const allPostsData = getSortedPostsData();
	return {
		props: {
			allPostsData,
		},
	};
}
