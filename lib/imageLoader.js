export default function imageLoader({ src, width, quality }) {
	// For Unsplash images, we can use their own image CDN with size parameters
	if (src.includes('unsplash.com')) {
		return `${src}${src.includes('?') ? '&' : '?'}w=${width}&q=${quality || 75}`;
	}
	// For other images, return the source as is
	return src;
}
