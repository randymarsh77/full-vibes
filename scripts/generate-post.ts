import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import fetch from 'node-fetch';
import { validateAndFixPost, getAllUsedImageUrls } from './fix-post-frontmatter';

const model = 'gpt-4o';

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

interface HNStory {
	title: string;
	url?: string;
	score: number;
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

// Function to get all post titles to avoid duplicating topics
function getAllPostTitles(): string[] {
	const postsDirectory = path.join(process.cwd(), 'posts');

	// Check if directory exists
	if (!fs.existsSync(postsDirectory)) {
		return [];
	}

	const filenames = fs.readdirSync(postsDirectory);

	const titles = filenames
		.filter((filename) => filename.endsWith('.md'))
		.map((filename) => {
			const filePath = path.join(postsDirectory, filename);
			const fileContents = fs.readFileSync(filePath, 'utf8');
			const { data } = matter(fileContents);
			return data.title as string;
		})
		.filter(Boolean);

	return titles;
}

// Fetch top AI-related stories from HackerNews
async function fetchHNStories(count: number = 15): Promise<HNStory[]> {
	console.log('Fetching current stories from HackerNews...');
	const topStoriesRes = await fetch(
		'https://hacker-news.firebaseio.com/v0/topstories.json'
	);
	const topIds = (await topStoriesRes.json()) as number[];

	const storyPromises = topIds.slice(0, 50).map(async (id) => {
		const res = await fetch(
			`https://hacker-news.firebaseio.com/v0/item/${encodeURIComponent(id)}.json`
		);
		return res.json() as Promise<HNStory>;
	});

	const stories = await Promise.all(storyPromises);

	// Filter for AI/tech-related stories and sort by score
	const aiKeywords = [
		'ai', 'llm', 'gpt', 'machine learning', 'neural', 'model', 'copilot',
		'openai', 'anthropic', 'gemini', 'claude', 'deep learning', 'transformer',
		'agent', 'coding', 'developer', 'programming', 'software', 'tech',
		'automation', 'robot', 'data', 'algorithm', 'compute', 'gpu', 'chip',
	];

	const aiStories = stories.filter((s) => {
		if (!s || !s.title) return false;
		const titleLower = s.title.toLowerCase();
		return aiKeywords.some((kw) => titleLower.includes(kw));
	});

	const sorted = (aiStories.length > 0 ? aiStories : stories.filter(Boolean))
		.sort((a, b) => (b.score || 0) - (a.score || 0))
		.slice(0, count);

	console.log(`Found ${sorted.length} relevant stories.`);
	return sorted;
}

// Helper to call the LLM
async function chat(
	client: OpenAI,
	systemPrompt: string,
	userPrompt: string,
	temperature: number = 0.7
): Promise<string> {
	const response = await client.chat.completions.create({
		model,
		messages: [
			{ role: 'system', content: systemPrompt },
			{ role: 'user', content: userPrompt },
		],
		temperature,
		max_tokens: 12000,
	});
	return response.choices[0]?.message?.content ?? '';
}

// Stage 1: Research current events
async function stageResearch(client: OpenAI, stories: HNStory[]): Promise<string> {
	console.log('\n📡 Stage 1: Researching current AI news...');

	const storySummaries = stories
		.map((s, i) => `${i + 1}. "${s.title}" (score: ${s.score})${s.url ? ` - ${s.url}` : ''}`)
		.join('\n');

	const research = await chat(
		client,
		`You are a tech journalist and researcher specializing in AI and software development. Analyze trending stories and identify key themes, debates, and emerging trends.`,
		`Here are the current top stories from HackerNews and tech news:\n\n${storySummaries}\n\nAnalyze these stories and provide:\n1. A summary of the top 3-5 themes or trends you see in AI and software development\n2. Any notable debates or controversies\n3. Emerging technologies or shifts in the industry\n4. Specific stories that could serve as good references for a blog post\n\nFocus on what's most interesting and timely in the AI + coding space.`
	);

	console.log('Research complete.');
	return research;
}

// Stage 2: Formulate topic and hot take
async function stageFormulateTopic(
	client: OpenAI,
	research: string,
	existingTitles: string[]
): Promise<string> {
	console.log('\n🎯 Stage 2: Formulating topic and hot take...');

	let existingTitlesText = '';
	if (existingTitles.length > 0) {
		existingTitlesText = "\nAvoid these topics we've already covered:\n";
		existingTitles.forEach((title) => {
			existingTitlesText += `- "${title}"\n`;
		});
	}

	const topic = await chat(
		client,
		`You are an opinionated tech blogger known for insightful hot takes on AI and coding. You identify angles that others miss and aren't afraid to take a strong position.`,
		`Based on this research of current events in the AI and tech space:\n\n${research}\n\n${existingTitlesText}\n\nPropose a single blog post topic that:\n1. Is rooted in current events and trends from the research above\n2. Has a clear, strong "hot take" opinion that will spark discussion\n3. Is relevant to the intersection of AI and coding/software development\n4. Includes specific references to current events that should be cited\n5. Has not been covered already (see the list of existing topics above)\n\nProvide:\n- TITLE: A compelling title\n- HOT TAKE: Your controversial or bold opinion in 1-2 sentences\n- ANGLE: The specific angle and argument structure for the post\n- REFERENCES: 3-5 specific stories, events, or facts to cite as references\n- KEY POINTS: 4-6 key points to make in the post`
	);

	console.log('Topic formulated.');
	return topic;
}

// Stage 3: Write the blog post
async function stageWrite(
	client: OpenAI,
	topic: string,
	examplesText: string,
	usedImagesText: string,
	formattedDate: string
): Promise<string> {
	console.log('\n✍️  Stage 3: Writing blog post...');

	const draft = await chat(
		client,
		`You are a skilled tech blogger writing for "Full Vibes", a blog about AI and coding. Your writing style is conversational but professional, with strong opinions backed by evidence. You always cite your sources and references.`,
		`Write a blog post based on this topic brief:\n\n${topic}\n\n${examplesText}\n\nFormat the post with this exact frontmatter structure:\n\n---\ntitle: "[An engaging title]"\ndate: "${formattedDate}"\nexcerpt: "[A brief 1-2 sentence summary]"\ncoverImage: "https://images.unsplash.com/photo-[a relevant image ID]"\n---\n\n[Post content with 4-6 sections]\n\nIMPORTANT REQUIREMENTS:\n1. The frontmatter must be formatted exactly as shown. Use standard double quotes, no line breaks within fields. Each key-value pair on its own line.\n2. ALWAYS include a language annotation for ALL code blocks (e.g. \`\`\`python not just \`\`\`). Use \`\`\`text as default.\n3. Include inline links to reference sources where you cite current events or news stories.\n4. The post should have a clear hot take / strong opinion supported by evidence and references.\n5. Include code examples where relevant.\n\n${usedImagesText}\nChoose a high-quality, relevant Unsplash image that has not been used before. Make sure the image ID is valid.`
	);

	console.log('Draft written.');
	return draft;
}

// Stage 4: Editor review and adjustments
async function stageEdit(client: OpenAI, draft: string): Promise<string> {
	console.log('\n📝 Stage 4: Editor reviewing and improving...');

	const editorFeedback = await chat(
		client,
		`You are a senior editor for a popular tech blog. You review posts for quality, clarity, engagement, and accuracy. You provide specific, actionable feedback.`,
		`Review this blog post draft and provide specific feedback:\n\n${draft}\n\nEvaluate:\n1. Is the title compelling and click-worthy?\n2. Is the hot take/opinion clear and well-supported?\n3. Are references and citations properly included?\n4. Is the writing engaging and well-structured?\n5. Are code examples (if any) correct and relevant?\n6. Is the frontmatter properly formatted?\n7. Does it flow well and maintain reader interest?\n\nProvide a list of specific improvements to make.`,
		0.4
	);

	console.log('Editor feedback received. Applying improvements...');

	const finalPost = await chat(
		client,
		`You are a skilled tech blogger. Apply the editor's feedback to improve the blog post. Output ONLY the complete, final blog post with frontmatter — no commentary or explanation.`,
		`Here is the original draft:\n\n${draft}\n\nHere is the editor's feedback:\n\n${editorFeedback}\n\nApply the editor's feedback to produce a polished final version. Output ONLY the complete blog post including the frontmatter block. Do not include any preamble, explanation, or commentary — just the post itself.\n\nIMPORTANT: Preserve the exact frontmatter format:\n---\ntitle: "..."\ndate: "..."\nexcerpt: "..."\ncoverImage: "..."\n---\n\nALWAYS include a language annotation for ALL code blocks.`,
		0.5
	);

	console.log('Editing complete.');
	return finalPost;
}

// Stage 5: Publish — save the final post to disk
function stagePublish(
	blogContent: string,
	formattedDate: string
): BlogPostResult {
	console.log('\n🚀 Stage 5: Publishing...');

	// Extract title from the markdown frontmatter
	const titleMatch = blogContent.match(/title:\s*['"](.+)['"]/);
	const title = titleMatch ? titleMatch[1] : `Blog Post ${formattedDate}`;

	// Generate slug from title
	const slug = generateSlug(title);

	// Create file path
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

	return { title, slug, filePath };
}

async function generateBlogPost(): Promise<BlogPostResult> {
	// Ensure GitHub token is available
	const token = process.env.GITHUB_TOKEN;
	if (!token) {
		throw new Error('GITHUB_TOKEN environment variable is not set');
	}

	// Initialize OpenAI client pointing at GitHub Models API
	const client = new OpenAI({
		baseURL: 'https://models.inference.ai.azure.com',
		apiKey: token,
	});

	// Get today's date
	const today = new Date();
	const formattedDate = today.toISOString().split('T')[0];

	// Get last 3 posts as examples
	const lastPosts = getLastNPosts(3);
	let examplesText = '';
	if (lastPosts.length > 0) {
		examplesText = 'Here are examples of the last blog posts for style reference:\n\n';
		lastPosts.forEach((post, index) => {
			examplesText += `EXAMPLE POST ${index + 1}:\n`;
			examplesText += `${post.content.substring(0, 500)}...\n\n`;
		});
	}

	// Get all post titles to avoid repeats
	const allTitles = getAllPostTitles();

	// Get all used image URLs to avoid duplicates
	const usedImageUrls = getAllUsedImageUrls();
	let usedImagesText = '';
	if (usedImageUrls.length > 0) {
		usedImagesText = '\nAvoid using these image URLs that are already in use:\n';
		usedImageUrls.forEach((url) => {
			usedImagesText += `- ${url}\n`;
		});
		usedImagesText += '\nFind a different Unsplash image that fits the topic.\n';
	}

	try {
		// Stage 1: Research current AI news from HackerNews
		const stories = await fetchHNStories();
		const research = await stageResearch(client, stories);

		// Stage 2: Formulate topic with a hot take
		const topic = await stageFormulateTopic(client, research, allTitles);

		// Stage 3: Write the blog post with references
		const draft = await stageWrite(client, topic, examplesText, usedImagesText, formattedDate);

		// Stage 4: Editor review and improvements
		const finalPost = await stageEdit(client, draft);

		// Stage 5: Publish the result
		return stagePublish(finalPost, formattedDate);
	} catch (error) {
		console.error('Error generating blog post:', error);
		throw error;
	}
}

// Run the function if executed directly
if (require.main === module) {
	generateBlogPost()
		.then((result) => {
			console.log('\n✅ Blog post generation completed successfully.');
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
