export async function classifyArticle(text) {
  const response = await fetch('http://localhost:8000/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  if (!response.ok) throw new Error('Failed to classify article');
  return response.json();
} 