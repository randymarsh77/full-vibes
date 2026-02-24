import Link from 'next/link';
import Head from 'next/head';

export default function Layout({ children, title, description }) {
	return (
		<div className="bg-gradient-to-br from-vibe-dark to-vibe-darker min-h-screen">
			<Head>
				<title>{title ? `${title} | Full Vibes Dev` : 'Full Vibes Dev'}</title>
				{description && <meta name="description" content={description} />}
			</Head>

			<header className="py-6 px-4 md:px-6 lg:px-8 backdrop-blur-sm bg-vibe-darker/70 sticky top-0 z-10">
				<div className="container mx-auto flex justify-between items-center">
					<div className="font-display font-bold text-2xl">
						<Link href="/">
							<span className="cursor-pointer bg-clip-text text-transparent bg-gradient-to-r from-vibe-pink to-vibe-blue">
								Full Vibes Dev
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
								<Link href="/projects" className="text-vibe-light hover:text-vibe-pink transition">
									Projects
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
							<li>
								<Link href="/contact" className="text-vibe-light hover:text-vibe-pink transition">
									Contact
								</Link>
							</li>
						</ul>
					</nav>
				</div>
			</header>

			<main className="container mx-auto px-4 md:px-6 lg:px-8 py-12">{children}</main>

			<footer className="bg-vibe-darker text-center py-8 text-vibe-gray border-t border-white/5">
				<div className="container mx-auto px-4">
					<p className="font-mono text-sm">
						© {new Date().getFullYear()} Full Vibes Dev. All rights reserved.
					</p>
				</div>
			</footer>
		</div>
	);
}
