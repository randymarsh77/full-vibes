import Link from 'next/link';
import Image from 'next/image';
import { getSortedPostsData } from '../lib/posts';
import Layout from '../components/Layout';

export default function Blog({ allPostsData }) {
	return (
		<Layout
			title="Blog"
			description="Articles on software development, developer tools, and technology from Full Vibes Dev."
		>
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
									priority={index === 0}
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
		</Layout>
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
