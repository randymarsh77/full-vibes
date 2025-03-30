/** @type {import('tailwindcss').Config} */
module.exports = {
	// Update content paths to include all possible locations for components and pages
	content: ['./src/pages/**/*.{js,ts,jsx,tsx}', './src/components/**/*.{js,ts,jsx,tsx}'],
	theme: {
		extend: {
			colors: {
				'vibe-purple': '#8B5CF6',
				'vibe-pink': '#EC4899',
				'vibe-blue': '#3B82F6',
				'vibe-dark': '#0F172A',
				'vibe-darker': '#060A14',
				'vibe-gray': '#94A3B8',
				'vibe-light': '#E2E8F0',
			},
			fontFamily: {
				sans: ['Inter', 'system-ui', 'sans-serif'],
				mono: ['Fira Code', 'JetBrains Mono', 'monospace'],
				display: ['Poppins', 'system-ui', 'sans-serif'],
			},
			typography: (theme) => ({
				DEFAULT: {
					css: {
						color: theme('colors.vibe-light'),
						a: {
							color: theme('colors.vibe-blue'),
							'&:hover': {
								color: theme('colors.vibe-pink'),
							},
						},
						h1: {
							color: theme('colors.white'),
							fontFamily: theme('fontFamily.display').join(', '),
						},
						h2: {
							color: theme('colors.white'),
							fontFamily: theme('fontFamily.display').join(', '),
						},
						h3: {
							color: theme('colors.white'),
							fontFamily: theme('fontFamily.display').join(', '),
						},
						h4: {
							color: theme('colors.white'),
							fontFamily: theme('fontFamily.display').join(', '),
						},
						code: {
							color: theme('colors.vibe-pink'),
							backgroundColor: 'rgba(0, 0, 0, 0.2)',
							padding: '0.25rem',
							borderRadius: '0.25rem',
							fontFamily: theme('fontFamily.mono').join(', '),
						},
					},
				},
			}),
		},
	},
	plugins: [require('@tailwindcss/typography')],
};
