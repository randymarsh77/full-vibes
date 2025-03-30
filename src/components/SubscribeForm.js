import { useState } from 'react';

export default function SubscribeForm() {
	const [email, setEmail] = useState('');
	const [status, setStatus] = useState('idle'); // idle, loading, success, error
	const [message, setMessage] = useState('');

	const handleSubmit = async (e) => {
		e.preventDefault();
		setStatus('loading');
		setMessage('');

		try {
			const response = await fetch('/api/subscribe', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ email }),
			});

			const data = await response.json();

			if (!response.ok) {
				throw new Error(data.error || 'Something went wrong');
			}

			setStatus('success');
			setMessage(data.message);
			setEmail('');

			// Reset success state after 5 seconds
			setTimeout(() => {
				if (status === 'success') {
					setStatus('idle');
					setMessage('');
				}
			}, 5000);
		} catch (error) {
			setStatus('error');
			setMessage(error.message);
		}
	};

	return (
		<div className="max-w-md mx-auto">
			<form onSubmit={handleSubmit}>
				<div className="flex flex-col sm:flex-row">
					<input
						type="email"
						placeholder="Enter your email"
						value={email}
						onChange={(e) => setEmail(e.target.value)}
						className="flex-grow px-4 py-2 rounded-lg sm:rounded-r-none focus:outline-none focus:ring-2 focus:ring-vibe-pink bg-vibe-darker/80 text-vibe-light border border-white/10 mb-2 sm:mb-0"
						disabled={status === 'loading'}
						required
					/>
					<button
						type="submit"
						className={`${
							status === 'loading' ? 'opacity-70 cursor-wait' : 'hover:opacity-90'
						} bg-gradient-to-r from-vibe-pink to-vibe-blue px-6 py-2 rounded-lg sm:rounded-l-none text-white font-medium transition w-full sm:w-auto`}
						disabled={status === 'loading'}
					>
						{status === 'loading' ? 'Subscribing...' : 'Subscribe'}
					</button>
				</div>

				{message && (
					<div
						className={`mt-3 p-2 rounded-lg text-center text-sm ${
							status === 'success'
								? 'bg-green-500/20 text-green-200 border border-green-500/30'
								: 'bg-red-500/20 text-red-200 border border-red-500/30'
						}`}
					>
						{message}
					</div>
				)}
			</form>
		</div>
	);
}
