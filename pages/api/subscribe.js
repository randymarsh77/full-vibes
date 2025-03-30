export default async function handler(req, res) {
	// Only accept POST requests
	if (req.method !== 'POST') {
		return res.status(405).json({ error: 'Method not allowed' });
	}

	const { email } = req.body;

	// Basic validation
	if (!email || !email.includes('@')) {
		return res.status(400).json({ error: 'Email is required and must be valid' });
	}

	// Get API key from environment variables
	const apiKey = process.env.MAILER_LITE;

	if (!apiKey) {
		console.error('MAILER_LITE is not defined in environment variables');
		return res.status(500).json({ error: 'Server configuration error' });
	}

	try {
		// Send request to MailerLite API
		const response = await fetch('https://connect.mailerlite.com/api/subscribers', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				Authorization: `Bearer ${apiKey}`,
			},
			body: JSON.stringify({ email }),
		});

		const data = await response.json();

		// Check if the request was successful
		if (!response.ok) {
			// Check if it's because the subscriber already exists
			if (response.status === 409) {
				return res.status(200).json({ message: 'You are already subscribed!' });
			}

			throw new Error(data.message || 'Failed to subscribe');
		}

		return res.status(201).json({ message: 'Subscription successful!' });
	} catch (error) {
		console.error('Subscription error:', error);
		return res.status(500).json({ error: error.message || 'Something went wrong on our end' });
	}
}
