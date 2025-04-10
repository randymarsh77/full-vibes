import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import chalk from 'chalk';

interface PostMeta {
	title?: string;
	date?: string;
	excerpt?: string;
	coverImage?: string;
	[key: string]: any;
}

/**
 * Formats markdown content by fixing common issues
 */
function formatMarkdownContent(content: string): string {
	// Replace multiple blank lines with a single blank line
	content = content.replace(/\n{3,}/g, '\n\n');

	// Ensure the file ends with a newline
	if (!content.endsWith('\n')) {
		content += '\n';
	}

	return content;
}

/**
 * Gets all image URLs used in existing posts
 */
function getAllUsedImageUrls(): string[] {
	const postsDirectory = path.join(process.cwd(), 'posts');
	if (!fs.existsSync(postsDirectory)) {
		return [];
	}

	const files = fs
		.readdirSync(postsDirectory)
		.filter((file) => file.endsWith('.md'))
		.map((file) => path.join(postsDirectory, file));

	const usedUrls: string[] = [];

	for (const filePath of files) {
		try {
			const content = fs.readFileSync(filePath, 'utf8');
			const parsed = matter(content);
			const coverImage = parsed.data.coverImage;

			if (coverImage && typeof coverImage === 'string') {
				usedUrls.push(coverImage);
			}
		} catch (err) {
			// Skip files that can't be parsed
			console.error(chalk.red(`Could not parse file to extract image URL: ${filePath}`));
		}
	}

	return usedUrls;
}

/**
 * Validates and fixes frontmatter for a markdown blog post
 */
function validateAndFixPost(filePath: string, checkOnly: boolean = false): boolean {
	console.log(chalk.blue(`Processing: ${filePath}`));

	// Read file content
	let content: string;
	try {
		content = fs.readFileSync(filePath, 'utf8');
	} catch (err) {
		console.error(chalk.red(`Error reading file: ${filePath}`));
		console.error(err);
		return false;
	}

	const filename = path.basename(filePath);
	let needsFix = false;
	let frontmatterNeedsFix = false;

	// Check for multiple consecutive blank lines and ensure file ends with newline
	const multipleBlankLines = content.match(/\n{3,}/g);
	const endsWithNewline = content.endsWith('\n');

	if (multipleBlankLines || !endsWithNewline) {
		if (multipleBlankLines) {
			console.log(chalk.yellow(`Multiple blank lines found in ${filename}`));
			needsFix = true;
		}

		if (!endsWithNewline) {
			console.log(chalk.yellow(`File doesn't end with newline: ${filename}`));
			needsFix = true;
		}
	}

	// Check if frontmatter is not at the beginning but exists elsewhere in the file
	if (!content.startsWith('---')) {
		const frontmatterMatch = content.match(/^[^\n]*?\n---\n([\s\S]*?)---/m);
		if (frontmatterMatch) {
			console.log(chalk.yellow(`File has frontmatter, but it's not at the beginning: ${filename}`));

			// Extract the frontmatter and the content after it
			const entireMatch = content.match(/^[^\n]*?\n(---\n[\s\S]*?---\n[\s\S]*$)/m);
			if (entireMatch && !checkOnly) {
				console.log(chalk.yellow(`Removing content before frontmatter in ${filename}`));
				content = entireMatch[1];
				needsFix = true;
			} else {
				console.error(chalk.red(`Cannot fix misplaced frontmatter in ${filename}`));
				return false;
			}
		} else {
			console.error(chalk.red(`File doesn't have frontmatter: ${filePath}`));
			return false;
		}
	}

	// Check for code blocks without language annotation and ending code blocks with language annotation
	// First, collect all code block positions with their complete context
	const codeBlocksInfo: Array<{
		isOpeningBlock: boolean;
		hasLanguage: boolean;
		position: number;
		match: string;
		lang?: string;
	}> = [];

	// Reset regex before use
	const allCodeBlocksRegex = /```([^\s\n]*)/g;
	let match;

	while ((match = allCodeBlocksRegex.exec(content)) !== null) {
		const isOpeningBlock = isOpeningCodeBlock(content, match.index);
		codeBlocksInfo.push({
			isOpeningBlock,
			hasLanguage: !!match[1],
			position: match.index,
			match: match[0],
			lang: match[1] || undefined,
		});
	}

	// Count blocks that need fixing
	const blocksWithoutLanguage = codeBlocksInfo.filter(
		(block) => block.isOpeningBlock && !block.hasLanguage
	);

	const endingBlocksWithLanguage = codeBlocksInfo.filter(
		(block) => !block.isOpeningBlock && block.hasLanguage
	);

	if (blocksWithoutLanguage.length > 0 || endingBlocksWithLanguage.length > 0) {
		if (blocksWithoutLanguage.length > 0) {
			console.log(
				chalk.yellow(
					`Found ${blocksWithoutLanguage.length} opening code block(s) without language annotation in ${filename}`
				)
			);
			needsFix = true;
		}

		if (endingBlocksWithLanguage.length > 0) {
			console.log(
				chalk.yellow(
					`Found ${endingBlocksWithLanguage.length} ending code block(s) with language annotation in ${filename}`
				)
			);
			needsFix = true;
		}

		// Only attempt to fix if not in check-only mode
		if (!checkOnly) {
			let fixedContent = content;

			// Fix opening blocks without language by adding "text"
			if (blocksWithoutLanguage.length > 0) {
				// Process in reverse order to avoid position shifts
				const sorted = [...blocksWithoutLanguage].sort((a, b) => b.position - a.position);

				for (const block of sorted) {
					console.log(chalk.blue(`Adding 'text' to code block at position ${block.position}`));
					fixedContent =
						fixedContent.substring(0, block.position) +
						'```text' +
						fixedContent.substring(block.position + 3);
				}

				console.log(
					chalk.yellow(
						`Added 'text' language annotation to ${blocksWithoutLanguage.length} code block(s) in ${filename}`
					)
				);
			}

			// Fix ending blocks with language by removing the language
			if (endingBlocksWithLanguage.length > 0) {
				// After fixing opening blocks, we need to re-scan the content to find ending blocks
				// This is necessary because positions may have shifted
				const updatedEndingBlocks: typeof endingBlocksWithLanguage = [];
				const endingRegex = /```([^\s\n]+)/g;
				let endingMatch;

				while ((endingMatch = endingRegex.exec(fixedContent)) !== null) {
					const isOpening = isOpeningCodeBlock(fixedContent, endingMatch.index);
					if (!isOpening && endingMatch[1]) {
						updatedEndingBlocks.push({
							isOpeningBlock: false,
							hasLanguage: true,
							position: endingMatch.index,
							match: endingMatch[0],
							lang: endingMatch[1],
						});
					}
				}

				// Process in reverse order
				updatedEndingBlocks.sort((a, b) => b.position - a.position);

				for (const block of updatedEndingBlocks) {
					console.log(
						chalk.blue(
							`Removing language '${block.lang}' from ending block at position ${block.position}`
						)
					);
					fixedContent =
						fixedContent.substring(0, block.position) +
						'```' +
						fixedContent.substring(block.position + 3 + (block.lang?.length || 0));
				}

				console.log(
					chalk.yellow(
						`Removed language annotation from ${updatedEndingBlocks.length} ending code block(s) in ${filename}`
					)
				);
			}

			content = fixedContent;
		}
	}

	// Parse frontmatter
	let parsed;
	try {
		parsed = matter(content);
	} catch (err) {
		console.error(chalk.red(`Error parsing frontmatter: ${filePath}`));
		console.error(err);
		return false;
	}

	const frontmatter: PostMeta = parsed.data;

	// Check required fields
	const requiredFields = ['title', 'date', 'excerpt', 'coverImage'];
	for (const field of requiredFields) {
		if (!frontmatter[field]) {
			console.error(chalk.red(`Missing required field '${field}' in ${filename}`));
			needsFix = true;
		}
	}

	// Check for multiline fields (common YAML error)
	for (const [key, value] of Object.entries(frontmatter)) {
		if (typeof value === 'string' && value.includes('\n')) {
			console.error(chalk.red(`Field '${key}' contains line breaks in ${filename}`));
			needsFix = true;
			// Fix: replace line breaks with spaces
			frontmatter[key] = value.replace(/\n/g, ' ').trim();
		}
	}

	// Check if cover image is properly formatted
	if (
		frontmatter.coverImage &&
		typeof frontmatter.coverImage === 'string' &&
		!frontmatter.coverImage.startsWith('https://')
	) {
		console.error(chalk.red(`Cover image URL doesn't start with https:// in ${filename}`));
		needsFix = true;
	}

	// Check if cover image URL is unique across all posts
	if (frontmatter.coverImage && typeof frontmatter.coverImage === 'string') {
		const allPosts = fs
			.readdirSync(path.join(process.cwd(), 'posts'))
			.filter((file) => file !== filename && file.endsWith('.md'))
			.map((file) => path.join(process.cwd(), 'posts', file));

		for (const otherPostPath of allPosts) {
			try {
				const otherContent = fs.readFileSync(otherPostPath, 'utf8');
				const otherParsed = matter(otherContent);

				if (otherParsed.data.coverImage === frontmatter.coverImage) {
					console.error(
						chalk.red(
							`Duplicate cover image URL found: "${
								frontmatter.coverImage
							}" also used in ${path.basename(otherPostPath)}`
						)
					);
					needsFix = true;
					break;
				}
			} catch (err) {
				// Skip files that can't be parsed
			}
		}
	}

	// If check only, don't make changes
	if (checkOnly) {
		return !needsFix;
	}

	// Fix the post if needed
	if (needsFix) {
		console.log(chalk.yellow(`Fixing issues in ${filename}`));

		// Check if frontmatter needs fixing
		frontmatterNeedsFix = false;
		for (const field of requiredFields) {
			if (!frontmatter[field]) {
				frontmatterNeedsFix = true;
				break;
			}
		}

		if (frontmatterNeedsFix) {
			// Create a clean frontmatter
			let fixedFrontmatter = '---\n';
			for (const field of requiredFields) {
				if (frontmatter[field]) {
					// Ensure strings are properly quoted
					fixedFrontmatter += `${field}: "${frontmatter[field].toString().replace(/"/g, '\\"')}"\n`;
				} else {
					// Add placeholder for missing required fields
					const today = new Date().toISOString().split('T')[0];

					switch (field) {
						case 'title':
							fixedFrontmatter += `title: "Untitled Post"\n`;
							break;
						case 'date':
							fixedFrontmatter += `date: "${today}"\n`;
							break;
						case 'excerpt':
							fixedFrontmatter += `excerpt: "A blog post about AI and coding."\n`;
							break;
						case 'coverImage':
							fixedFrontmatter += `coverImage: "https://images.unsplash.com/photo-1555066931-4365d14bab8c"\n`;
							break;
						default:
							fixedFrontmatter += `${field}: ""\n`;
					}
				}
			}

			// Add any additional fields from the original frontmatter
			for (const [key, value] of Object.entries(frontmatter)) {
				if (!requiredFields.includes(key)) {
					fixedFrontmatter += `${key}: "${value}"\n`;
				}
			}

			fixedFrontmatter += '---\n\n';

			// Replace the original frontmatter with the fixed one and keep the content
			let newContent = fixedFrontmatter + parsed.content;

			// Apply formatting fixes
			newContent = formatMarkdownContent(newContent);

			// Write back to file
			fs.writeFileSync(filePath, newContent);
			console.log(chalk.green(`Fixed frontmatter and formatting in ${filename}`));
		} else {
			// Only format the content (fix blank lines and newline at end)
			const formattedContent = formatMarkdownContent(content);

			fs.writeFileSync(filePath, formattedContent);
			console.log(chalk.green(`Fixed formatting in ${filename}`));
		}

		return true;
	}

	console.log(chalk.green(`✓ ${filename} has valid frontmatter`));
	return true;
}

/**
 * Check if a code block marker is an opening block or closing block
 * This is determined by counting the number of code markers before this position
 */
function isOpeningCodeBlock(content: string, position: number): boolean {
	const contentBefore = content.substring(0, position);
	const markersBefore = (contentBefore.match(/```/g) || []).length;
	return markersBefore % 2 === 0; // Even count means this is an opening block
}

/**
 * Process a single post file or all post files
 */
function processPosts(targetPath?: string) {
	const postsDirectory = path.join(process.cwd(), 'posts');

	if (!fs.existsSync(postsDirectory)) {
		console.error(chalk.red('Posts directory not found!'));
		process.exit(1);
	}

	if (targetPath) {
		// Process single file
		const fullPath = targetPath.startsWith(postsDirectory)
			? targetPath
			: path.join(postsDirectory, path.basename(targetPath));

		if (!fs.existsSync(fullPath)) {
			console.error(chalk.red(`File not found: ${fullPath}`));
			process.exit(1);
		}

		validateAndFixPost(fullPath);
	} else {
		// Process all files
		console.log(chalk.blue('Validating all posts in the posts directory...'));

		const files = fs
			.readdirSync(postsDirectory)
			.filter((file) => file.endsWith('.md'))
			.map((file) => path.join(postsDirectory, file));

		let validCount = 0;
		let fixedCount = 0;
		let errorCount = 0;

		for (const file of files) {
			const result = validateAndFixPost(file);
			if (result) {
				validCount++;
			} else {
				errorCount++;
			}
		}

		console.log(chalk.blue('\nSummary:'));
		console.log(chalk.green(`✓ ${validCount} posts validated successfully`));
		if (errorCount > 0) {
			console.log(chalk.red(`✗ ${errorCount} posts have errors that need manual attention`));
		}
	}
}

// Run if executed directly
if (require.main === module) {
	const args = process.argv.slice(2);
	const filePath = args[0];
	processPosts(filePath);
}

export { validateAndFixPost, processPosts, getAllUsedImageUrls };
