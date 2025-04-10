import '../styles/globals.css';
import { useEffect } from 'react';
import Head from 'next/head';
import 'prismjs/themes/prism-tomorrow.css'; // Import a Prism theme

export default function MyApp({ Component, pageProps }) {
	useEffect(() => {
		// Load Prism.js on the client side
		if (typeof window !== 'undefined') {
			// When the DOM is ready, highlight all code blocks
			const Prism = require('prismjs');

			// Load commonly used languages
			require('prismjs/components/prism-javascript');
			require('prismjs/components/prism-typescript');
			require('prismjs/components/prism-python');
			require('prismjs/components/prism-rust');
			require('prismjs/components/prism-bash');
			require('prismjs/components/prism-c');
			require('prismjs/components/prism-cpp');
			require('prismjs/components/prism-java');
			require('prismjs/components/prism-json');
			require('prismjs/components/prism-yaml');
			require('prismjs/components/prism-markup');
			require('prismjs/components/prism-css');

			// Highlight all code blocks
			Prism.highlightAll();
		}
	}, []);

	return (
		<>
			<Head>
				<meta name="viewport" content="width=device-width, initial-scale=1" />
				<title>Full Vibes</title>
			</Head>
			<Component {...pageProps} />
		</>
	);
}
