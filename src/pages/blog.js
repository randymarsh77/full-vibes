import { useState } from 'react';
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

const POSTS_PER_PAGE = 4;

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

function MonthSection({ month, posts, defaultExpanded }) {
	const [expanded, setExpanded] = useState(defaultExpanded);
	const [visibleCount, setVisibleCount] = useState(POSTS_PER_PAGE);

	const visiblePosts = posts.slice(0, expanded ? visibleCount : 0);
	const hasMore = expanded && visibleCount < posts.length;

	return (
		<div className="mb-10">
			<button
				type="button"
				onClick={() => setExpanded(!expanded)}
				className="flex items-center gap-2 text-xl font-semibold font-display text-vibe-blue mb-6 hover:text-vibe-pink transition cursor-pointer"
			>
				<span className="inline-block transition-transform duration-200" style={{ transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)' }}>
					▶
				</span>
				{month} ({posts.length})
			</button>
			{expanded && (
				<>
					<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
						{visiblePosts.map((post) => (
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
					{hasMore && (
						<div className="mt-6 text-center">
							<button
								type="button"
								onClick={() => setVisibleCount((c) => c + POSTS_PER_PAGE)}
								className="text-vibe-blue hover:text-vibe-pink transition font-medium cursor-pointer"
							>
								Load more ({posts.length - visibleCount} remaining) →
							</button>
						</div>
					)}
				</>
			)}
		</div>
	);
}

function YearSection({ year, months, grouped, defaultExpanded, defaultExpandedMonth }) {
	const [expanded, setExpanded] = useState(defaultExpanded);

	return (
		<section className="mb-12">
			<button
				type="button"
				onClick={() => setExpanded(!expanded)}
				className="flex items-center gap-3 text-3xl font-bold font-display text-white mb-8 border-b border-white/10 pb-3 w-full hover:text-vibe-purple transition cursor-pointer"
			>
				<span className="inline-block transition-transform duration-200 text-lg" style={{ transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)' }}>
					▶
				</span>
				{year}
			</button>
			{expanded &&
				months.map((month) => (
					<MonthSection
						key={month}
						month={month}
						posts={grouped[year][month]}
						defaultExpanded={defaultExpanded && month === defaultExpandedMonth}
					/>
				))}
		</section>
	);
}

export default function Blog({ allPostsData }) {
	const [latestPost, ...remainingPosts] = allPostsData;
	const grouped = groupPostsByYearAndMonth(remainingPosts);
	const years = Object.keys(grouped).sort((a, b) => b.localeCompare(a));

	const firstYear = years[0];
	const firstYearMonths = firstYear
		? Object.keys(grouped[firstYear]).sort((a, b) => MONTH_NAMES.indexOf(b) - MONTH_NAMES.indexOf(a))
		: [];
	const firstMonth = firstYearMonths[0];

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
				const isFirstYear = year === firstYear;
				return (
					<YearSection
						key={year}
						year={year}
						months={months}
						grouped={grouped}
						defaultExpanded={isFirstYear}
						defaultExpandedMonth={isFirstYear ? firstMonth : null}
					/>
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
