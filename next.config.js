/** @type {import('next').NextConfig} */
const nextConfig = {
	reactStrictMode: true,
	images: {
		domains: ['images.unsplash.com'],
		unoptimized: true, // This allows images to work correctly on Netlify
	},
	// Add environment variables that should be available on client-side
	// Don't add the API key here - it should only be available server-side
	env: {
		SITE_URL: process.env.SITE_URL || 'https://fullvibes.dev',
	},
};

module.exports = nextConfig;
