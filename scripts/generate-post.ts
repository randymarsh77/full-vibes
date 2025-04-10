import Anthropic from '@anthropic-ai/sdk';
import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { validateAndFixPost } from './fix-post-frontmatter';

const model = 'claude-3-7-sonnet-20250219';

// Not exported from Anthropic
interface TextBlock {
	text: string;
}

interface BlogPostResult {
	title: string;
	slug: string;
	filePath: string;
}

interface PostMeta {
	title: string;
	date: string;
	excerpt: string;
	coverImage: string;
	content: string;
}

// Function to generate a slug from title
function generateSlug(title: string): string {
	return title
		.toLowerCase()
		.replace(/[^\w\s-]/g, '')
		.replace(/[\s_-]+/g, '-')
		.replace(/^-+|-+$/g, '');
}

// Function to get the last N published posts
function getLastNPosts(n: number): PostMeta[] {
	const postsDirectory = path.join(process.cwd(), 'posts');

	// Check if directory exists
	if (!fs.existsSync(postsDirectory)) {
		console.log('Posts directory not found. No examples will be provided.');
		return [];
	}

	const filenames = fs.readdirSync(postsDirectory);

	const allPosts = filenames
		.filter((filename) => filename.endsWith('.md'))
		.map((filename) => {
			const filePath = path.join(postsDirectory, filename);
			const fileContents = fs.readFileSync(filePath, 'utf8');
			const { data, content } = matter(fileContents);

			return {
				title: data.title,
				date: data.date,
				excerpt: data.excerpt,
				coverImage: data.coverImage,
				content: content,
			} as PostMeta;
		})
		.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

	return allPosts.slice(0, n);
}

async function generateBlogPost(): Promise<BlogPostResult> {
	// Ensure API key is available
	const apiKey = process.env.ANTHROPIC_API_KEY;
	if (!apiKey) {
		throw new Error('ANTHROPIC_API_KEY environment variable is not set');
	}

	// Initialize Claude
	const claude = new Anthropic({
		apiKey,
	});

	// Get today's date
	const today = new Date();
	const formattedDate = today.toISOString().split('T')[0];

	// Get last 3 posts as examples
	const lastPosts = getLastNPosts(3);

	// Create examples string
	let examplesText = '';
	if (lastPosts.length > 0) {
		examplesText = 'Here are examples of the last blog posts for reference:\n\n';

		lastPosts.forEach((post, index) => {
			examplesText += `EXAMPLE POST ${index + 1}:\n\n`;
			// Include just first 500 characters of content to keep prompt size reasonable
			examplesText += `${post.content.substring(0, 500)}...\n\n`;
		});
	}

	// Prompt for Claude
	const prompt = `
Write a high-quality blog post for a tech blog called "Full Vibes" that focuses on the intersection of AI and coding.

${examplesText}

Now, create a new original blog post following this exact format for the frontmatter:

---
title: "[An engaging title related to the topic]"
date: "${formattedDate}"
excerpt: "[A brief 1-2 sentence summary of the post]"
coverImage: "https://images.unsplash.com/photo-[a relevant image ID]"
---

# [Title]

[Introduction paragraph]

## [Section 1]

[Content]

## [Section 2]

[Content]

...and so on with 4-6 sections total

## Conclusion

[Closing thoughts]

Make sure the post is informative, has a positive tone, and includes practical insights. The writing style should be conversational but professional.

IMPORTANT: The frontmatter must be formatted exactly as shown above. The quotes must be standard double quotes, not fancy quotes. There must be no line breaks within any frontmatter field. Each key-value pair must be on its own line.

Include code examples where relevant.

Try to come up with a unique topic that is not too similar to the examples provided and provides additional insights and value.

Avoid reusing images from existing posts.
`;

	try {
		console.log('Generating blog post with Claude...');
		const response = await claude.messages.create({
			messages: [{ role: 'user', content: prompt }],
			max_tokens: 12000,
			temperature: 0.7,
			model,
		});

		// Extract the content from the response
		const blogContent = response.content
			.filter((x) => x.type === 'text')
			.map((x) => (x as TextBlock).text)
			.join('\n');

		// Extract title from the markdown frontmatter
		const titleMatch = blogContent.match(/title:\s*['"](.+)['"]/);
		const title = titleMatch ? titleMatch[1] : `Blog Post ${formattedDate}`;

		// Generate slug from title
		const slug = generateSlug(title);

		// Create file path - using posts directory
		const filePath = path.join(process.cwd(), 'posts', `${slug}.md`);

		// Make sure the directory exists
		const postsDir = path.join(process.cwd(), 'posts');
		if (!fs.existsSync(postsDir)) {
			fs.mkdirSync(postsDir, { recursive: true });
		}

		// Write the blog post to a file
		fs.writeFileSync(filePath, blogContent);

		console.log(`Blog post created: ${filePath}`);
		console.log(`Title: ${title}`);

		// Validate and fix the post frontmatter
		console.log('Validating and fixing frontmatter...');
		validateAndFixPost(filePath);

		return {
			title,
			slug,
			filePath,
		};
	} catch (error) {
		console.error('Error generating blog post:', error);
		throw error;
	}
}

// Run the function if executed directly
if (require.main === module) {
	generateBlogPost()
		.then((result) => {
			console.log('Blog post generation completed successfully.');
			console.log(`Title: ${result.title}`);
			console.log(`File: ${result.filePath}`);
		})
		.catch((err) => {
			console.error('Failed to generate blog post:', err);
			process.exit(1);
		});
}

// Export for use in other scripts
export { generateBlogPost, BlogPostResult };
