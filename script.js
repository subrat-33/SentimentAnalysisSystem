const API_BASE_URL = 'http://localhost:8000';

// Theme Management
const themeToggle = document.getElementById('themeToggle');
const currentTheme = localStorage.getItem('theme') || 'light';

if (currentTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
}

themeToggle.addEventListener('click', () => {
    let theme = document.documentElement.getAttribute('data-theme');
    if (theme === 'dark') {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('theme', 'light');
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    }
});

async function analyzeSentiment() {
    const text = document.getElementById('tweetInput').value.trim();
    if (!text) return;

    // UI State: Loading
    setLoading(true);
    hideError();
    
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.classList.remove('hidden');
    
    // Reset cards to a "calculating" state
    resetCard('tfidf');
    resetCard('bert');

    // Analyze TF-IDF
    const tfidfPromise = fetch(`${API_BASE_URL}/predict/tfidf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    }).then(res => res.ok ? res.json() : Promise.reject('TF-IDF Error'))
      .then(data => updateResultCard('tfidf', data))
      .catch(err => {
          console.error(err);
          showCardError('tfidf', 'TF-IDF model is not responding (check if server is running)');
      });

    // Analyze BERT
    const bertPromise = fetch(`${API_BASE_URL}/predict/bert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    }).then(res => res.ok ? res.json() : Promise.reject('BERT Error'))
      .then(data => updateResultCard('bert', data))
      .catch(err => {
          console.error(err);
          showCardError('bert', 'BERT model failed or is still loading on the server');
      });

    try {
        await Promise.allSettled([tfidfPromise, bertPromise]);
    } finally {
        setLoading(false);
    }
}

function resetCard(modelPrefix) {
    const card = document.getElementById(`${modelPrefix}Card`);
    card.style.opacity = '0.5';
    document.getElementById(`${modelPrefix}Badge`).textContent = '...';
}

function showCardError(modelPrefix, message) {
    const chart = document.getElementById(`${modelPrefix}Chart`);
    const badge = document.getElementById(`${modelPrefix}Badge`);
    const card = document.getElementById(`${modelPrefix}Card`);
    
    card.style.opacity = '1';
    badge.textContent = '⚠️ Error';
    badge.className = 'sentiment-badge badge-negative';
    chart.innerHTML = `<p style="font-size: 0.8rem; color: var(--negative);">${message}</p>`;
}

function updateResultCard(modelPrefix, data) {
    const card = document.getElementById(`${modelPrefix}Card`);
    const badge = document.getElementById(`${modelPrefix}Badge`);
    const progress = document.getElementById(`${modelPrefix}Progress`);
    const confValue = document.getElementById(`${modelPrefix}ConfValue`);
    const chart = document.getElementById(`${modelPrefix}Chart`);

    card.style.opacity = '1';

    // Map labels to visuals
    const moodMap = {
        'Positive': { emoji: '😊', class: 'badge-positive', color: 'var(--positive)' },
        'Negative': { emoji: '😠', class: 'badge-negative', color: 'var(--negative)' },
        'Neutral': { emoji: '😐', class: 'badge-neutral', color: 'var(--neutral)' }
    };

    const mood = moodMap[data.label] || moodMap['Neutral'];

    // Update Badge
    badge.textContent = `${mood.emoji} ${data.label}`;
    badge.className = `sentiment-badge ${mood.class}`;

    // Update Progress Bar
    const confPercent = (data.confidence * 100).toFixed(1);
    progress.style.width = '0%'; // Reset first for animation
    setTimeout(() => {
        progress.style.width = `${confPercent}%`;
        progress.style.backgroundColor = mood.color;
    }, 50);
    confValue.textContent = `${confPercent}%`;

    // Update Top Words Chart
    chart.innerHTML = '';
    
    if (!data.top_words || data.top_words.length === 0) {
        chart.innerHTML = '<p style="font-size: 0.8rem; color: var(--text-secondary);">No significant tokens found.</p>';
        return;
    }

    // Normalize scores for chart (find max absolute score)
    const maxScore = Math.max(...data.top_words.map(w => Math.abs(w.score)), 0.0001);

    data.top_words.forEach((item, index) => {
        const barWidth = (Math.abs(item.score) / maxScore) * 100;
        const color = item.score > 0 ? 'var(--positive)' : 'var(--negative)';
        
        const barWrapper = document.createElement('div');
        barWrapper.className = 'chart-bar-wrapper';
        barWrapper.innerHTML = `
            <div class="bar-label">
                <span>${item.word}</span>
                <span>${(item.score > 0 ? '+' : '') + item.score.toFixed(3)}</span>
            </div>
            <div class="bar-fill-bg">
                <div class="bar-fill" style="width: 0%; background-color: ${color}"></div>
            </div>
        `;
        chart.appendChild(barWrapper);
        
        // Staggered animation for bars
        setTimeout(() => {
            const fill = barWrapper.querySelector('.bar-fill');
            if (fill) fill.style.width = `${barWidth}%`;
        }, 100 + (index * 50));
    });
}


function setLoading(isLoading) {
    const btn = document.getElementById('analyzeBtn');
    const text = document.getElementById('btnText');
    const spinner = document.getElementById('loadingSpinner');

    btn.disabled = isLoading;
    if (isLoading) {
        text.textContent = 'Analyzing...';
        spinner.classList.remove('hidden');
    } else {
        text.textContent = 'Analyze Sentiment';
        spinner.classList.add('hidden');
    }
}

function showError() {
    document.getElementById('errorMessage').classList.remove('hidden');
    document.getElementById('resultsContainer').classList.add('hidden');
}

function hideError() {
    document.getElementById('errorMessage').classList.add('hidden');
}
