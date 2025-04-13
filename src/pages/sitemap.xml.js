import { fetchAllUrls } from '../lib/api';

const Sitemap = () => {
	return null;
};

export const getServerSideProps = async ({ res }) => {
	res.setHeader('Content-Type', 'text/xml');

	// Fetch dynamic URLs
	const { blogPosts } = await fetchAllUrls();

	const baseUrl = 'https://fullvibes.dev';
	const currentDate = new Date().toISOString().split('T')[0];

	// Create the main static URLs
	let sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>${baseUrl}/</loc>
    <lastmod>${currentDate}</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>${baseUrl}/about</loc>
    <lastmod>${currentDate}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>${baseUrl}/blog</loc>
    <lastmod>${currentDate}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.7</priority>
  </url>`;

	// Add blog posts to sitemap
	blogPosts.forEach((post) => {
		sitemap += `
  <url>
    <loc>${baseUrl}/posts/${post.slug}</loc>
    <lastmod>${post.lastModified}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.6</priority>
  </url>`;
	});

	// Close the sitemap
	sitemap += `
</urlset>`;

	res.write(sitemap);
	res.end();

	return {
		props: {},
	};
};

export default Sitemap;
