import { useEffect, useRef } from 'react';
import Prism from 'prismjs';

export default function CodeRenderer({ html }) {
	const contentRef = useRef(null);

	useEffect(() => {
		if (contentRef.current) {
			Prism.highlightAllUnder(contentRef.current);
		}
	}, [html]);

	return (
		<div
			ref={contentRef}
			dangerouslySetInnerHTML={{ __html: html }}
			className="prose prose-invert max-w-none"
		/>
	);
}
