import Link from 'next/link';
import Image from 'next/image';
import { getAllPostIds, getPostData } from '../../lib/posts';
import Layout from '../../components/Layout';
import CodeRenderer from '../../components/CodeRenderer';

export default function Post({ postData }) {
	return (
		<Layout title={postData.title} description={postData.excerpt}>
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
						<Image
							src={postData.coverImage}
							alt={postData.title}
							fill
							className="object-cover"
							unoptimized={true}
						/>
					</div>
				)}

				<div className="bg-vibe-dark/60 backdrop-blur-lg rounded-xl p-8 border border-white/5 shadow-lg shadow-vibe-purple/10">
					<CodeRenderer html={postData.contentHtml} />
				</div>
			</article>
		</Layout>
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
