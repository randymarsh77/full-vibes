import Link from 'next/link';
import Image from 'next/image';
import { getSortedPostsData } from '../lib/posts';
import Layout from '../components/Layout';

const MONTH_NAMES = [
	'January',
	'February',
	'March',
	'April',
	'May',
	'June',
	'July',
	'August',
	'September',
	'October',
	'November',
	'December',
];

function groupPostsByYearAndMonth(posts) {
	const grouped = {};
	for (const post of posts) {
		const [year, month] = post.date.split('-');
		if (!grouped[year]) {
			grouped[year] = {};
		}
		const monthName = MONTH_NAMES[parseInt(month, 10) - 1];
		if (!grouped[year][monthName]) {
			grouped[year][monthName] = [];
		}
		grouped[year][monthName].push(post);
	}
	return grouped;
}

function PostCard({ id, date, title, excerpt, coverImage, priority }) {
	return (
		<article className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl overflow-hidden hover:transform hover:scale-105 transition duration-300 border border-white/5 shadow-lg shadow-vibe-purple/10">
			<Link href={`/posts/${id}`}>
				<div className="relative h-64">
					<Image
						src={coverImage || 'https://images.unsplash.com/photo-1550745165-9bc0b252726f'}
						alt={title}
						fill
						className="object-cover"
						sizes="(max-width: 768px) 100vw, 50vw"
						priority={priority}
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
	);
}

export default function Blog({ allPostsData }) {
	const [latestPost, ...remainingPosts] = allPostsData;
	const grouped = groupPostsByYearAndMonth(remainingPosts);
	const years = Object.keys(grouped).sort((a, b) => b.localeCompare(a));

	return (
		<Layout
			title="Blog"
			description="Articles on software development, developer tools, and technology from Full Vibes Dev."
		>
			<h1 className="text-4xl font-bold font-display text-white mb-12 text-center">Blog</h1>

			{latestPost && (
				<section className="mb-16">
					<h2 className="text-2xl font-bold font-display text-white mb-6">Latest Post</h2>
					<article className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl overflow-hidden hover:transform hover:scale-[1.02] transition duration-300 border border-white/5 shadow-lg shadow-vibe-purple/10">
						<Link href={`/posts/${latestPost.id}`}>
							<div className="md:flex">
								<div className="relative h-72 md:h-auto md:w-1/2">
									<Image
										src={
											latestPost.coverImage ||
											'https://images.unsplash.com/photo-1550745165-9bc0b252726f'
										}
										alt={latestPost.title}
										fill
										className="object-cover"
										sizes="(max-width: 768px) 100vw, 50vw"
										priority
										placeholder="blur"
										blurDataURL="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
									/>
								</div>
								<div className="p-8 md:w-1/2 flex flex-col justify-center">
									<p className="text-vibe-pink text-sm mb-2 font-mono">
										{latestPost.date}
									</p>
									<h3 className="text-3xl font-bold font-display text-white mb-4">
										{latestPost.title}
									</h3>
									<p className="text-vibe-gray mb-6">{latestPost.excerpt}</p>
									<span className="text-vibe-blue hover:text-vibe-pink transition font-medium">
										Read more →
									</span>
								</div>
							</div>
						</Link>
					</article>
				</section>
			)}

			{years.map((year) => {
				const months = Object.keys(grouped[year]).sort((a, b) => {
					return MONTH_NAMES.indexOf(b) - MONTH_NAMES.indexOf(a);
				});
				return (
					<section key={year} className="mb-12">
						<h2 className="text-3xl font-bold font-display text-white mb-8 border-b border-white/10 pb-3">
							{year}
						</h2>
						{months.map((month) => (
							<div key={month} className="mb-10">
								<h3 className="text-xl font-semibold font-display text-vibe-blue mb-6">
									{month}
								</h3>
								<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
									{grouped[year][month].map((post) => (
										<PostCard
											key={post.id}
											id={post.id}
											date={post.date}
											title={post.title}
											excerpt={post.excerpt}
											coverImage={post.coverImage}
											priority={false}
										/>
									))}
								</div>
							</div>
						))}
					</section>
				);
			})}
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
