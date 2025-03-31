const Anthropic = require('@anthropic-ai/sdk');
const fs = require('fs');
const path = require('path');

const model = 'claude-3-7-sonnet-20250219';

// Function to generate a slug from title
function generateSlug(title) {
	return title
		.toLowerCase()
		.replace(/[^\w\s-]/g, '')
		.replace(/[\s_-]+/g, '-')
		.replace(/^-+|-+$/g, '');
}

async function generateBlogPost() {
	// Initialize Claude
	const claude = new Anthropic({
		apiKey: process.env.ANTHROPIC_API_KEY,
	});

	// Get today's date
	const today = new Date();
	const formattedDate = today.toISOString().split('T')[0];

	// Prompt for Claude
	const prompt = `
  Write a high-quality blog post for a tech blog called "Full Vibes" that focuses on the intersection of AI and coding.
  
  The blog should follow this format:
  
  ---
  title: '[An engaging title related to the topic]'
  date: '${formattedDate}'
  excerpt: '[A brief 1-2 sentence summary of the post]'
  coverImage: 'https://images.unsplash.com/photo-[a relevant image ID]'
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
  
  Include code examples where relevant.
  `;

	try {
		console.log('Generating blog post with Claude...');
		const response = await claude.messages.create({
			messages: [{ role: 'user', content: prompt }],
			max_tokens: 12000,
			temperature: 0.7,
			model,
		});

		const blogContent = response.content;

		// Extract title from the markdown frontmatter
		const titleMatch = blogContent.match(/title:\s*['"](.+)['"]/);
		const title = titleMatch ? titleMatch[1] : `Blog Post ${formattedDate}`;

		// Generate slug from title
		const slug = generateSlug(title);

		// Create file path - using src/posts directory
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
		return { title, slug };
	} catch (error) {
		console.error('Error generating blog post:', error);
		throw error;
	}
}

// Run the function if executed directly
if (require.main === module) {
	generateBlogPost()
		.then(() => console.log('Blog post generation completed.'))
		.catch((err) => {
			console.error('Failed to generate blog post:', err);
			process.exit(1);
		});
}

// Export for use in other scripts
module.exports = { generateBlogPost };
