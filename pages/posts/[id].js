import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';
import { getAllPostIds, getPostData } from '../../lib/posts';

export default function Post({ postData }) {
	return (
		<div className="bg-gradient-to-br from-vibe-dark to-vibe-darker min-h-screen">
			<Head>
				<title>{postData.title} | Full Vibes</title>
				<meta name="description" content={postData.excerpt} />
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
				<article className="max-w-3xl mx-auto">
					<div className="mb-8">
						<Link
							href="/blog"
							className="text-vibe-blue hover:text-vibe-pink transition font-medium"
						>
							← Back to blog
						</Link>
					</div>

					<h1 className="text-4xl font-bold font-display text-white mb-4">{postData.title}</h1>
					<p className="text-vibe-pink mb-8 font-mono">{postData.date}</p>

					{postData.coverImage && (
						<div className="relative h-80 mb-8 rounded-xl overflow-hidden border border-white/10 shadow-lg shadow-vibe-purple/10">
							<Image src={postData.coverImage} alt={postData.title} fill className="object-cover" />
						</div>
					)}

					<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-8 border border-white/5 shadow-lg shadow-vibe-purple/10">
						<div
							className="prose prose-invert max-w-none prose-headings:text-white prose-headings:font-display prose-a:text-vibe-blue hover:prose-a:text-vibe-pink prose-code:text-vibe-pink prose-code:font-mono prose-code:bg-black/30 prose-code:px-1 prose-code:rounded"
							dangerouslySetInnerHTML={{ __html: postData.contentHtml }}
						/>
					</div>
				</article>
			</main>

			<footer className="bg-vibe-darker text-center py-8 text-vibe-gray border-t border-white/5 mt-12">
				<div className="container mx-auto px-4">
					<p className="font-mono text-sm">
						© {new Date().getFullYear()} Full Vibes. All rights reserved.
					</p>
				</div>
			</footer>
		</div>
	);
}

export async function getStaticPaths() {
	const paths = getAllPostIds();
	return {
		paths,
		fallback: false,
	};
}

export async function getStaticProps({ params }) {
	const postData = await getPostData(params.id);
	return {
		props: {
			postData,
		},
	};
}
