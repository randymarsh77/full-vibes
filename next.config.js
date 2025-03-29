/** @type {import('next').NextConfig} */
const nextConfig = {
	reactStrictMode: true,
	images: {
		domains: ['images.unsplash.com'],
		unoptimized: true, // This allows images to work correctly on Netlify
	},
};

module.exports = nextConfig;
