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

	// Check for balanced code blocks
	const codeBlockMatches = content.match(/```/g);
	if (codeBlockMatches) {
		const codeBlockCount = codeBlockMatches.length;
		if (codeBlockCount % 2 !== 0) {
			console.log(
				chalk.yellow(
					`Unbalanced code blocks found in ${filename}: ${codeBlockCount} backtick groups`
				)
			);
			needsFix = true;

			// Only attempt to fix if not in check-only mode
			if (!checkOnly) {
				// Find the last code block delimiter and remove it if it's unpaired
				const lastIndex = content.lastIndexOf('```');
				if (lastIndex !== -1) {
					// Check if this is the last line or nearly the last line (might have newlines after)
					const contentAfterLastDelimiter = content.substring(lastIndex + 3).trim();
					if (!contentAfterLastDelimiter) {
						// If there's nothing meaningful after the last delimiter, remove it
						content = content.substring(0, lastIndex);
						console.log(
							chalk.yellow(`Removed unpaired code block delimiter at the end of ${filename}`)
						);
					} else {
						// If there's content after the last delimiter, add a closing delimiter at the end
						content = content + '\n```\n';
						console.log(
							chalk.yellow(`Added closing code block delimiter at the end of ${filename}`)
						);
					}
				}
			}
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

export { validateAndFixPost, processPosts };
