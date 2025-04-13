import { getSortedPostsData } from './posts';

export async function fetchAllUrls() {
	// Get real blog posts using the existing getSortedPostsData function
	const allPosts = getSortedPostsData();

	// Map the posts data to the format expected by the sitemap generator
	const blogPosts = allPosts.map((post) => ({
		slug: post.id,
		lastModified: post.date || new Date().toISOString().split('T')[0],
		title: post.title || post.id,
	}));

	return {
		blogPosts,
		// Add other dynamic content types here
	};
}
