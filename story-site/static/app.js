// SPDX-License-Identifier: GPL-3.0-or-later

(() => {
  const search = document.querySelector('#archive-search');
  const collection = document.querySelector('#collection-filter');
  const clear = document.querySelector('#clear-filters');
  const cards = [...document.querySelectorAll('[data-entry-card]')];
  const resultCount = document.querySelector('#result-count');
  const emptyState = document.querySelector('#empty-state');
  const fullText = new Map();

  if (!search || !collection || cards.length === 0) return;

  const applyFilters = () => {
    const query = search.value.trim().toLowerCase();
    const selected = collection.value;
    let visible = 0;

    for (const card of cards) {
      const indexedText = `${card.dataset.search ?? ''} ${fullText.get(card.dataset.entryId) ?? ''}`;
      const matchesQuery = !query || indexedText.includes(query);
      const matchesCollection = selected === 'all' || card.dataset.collection === selected;
      card.hidden = !(matchesQuery && matchesCollection);
      if (!card.hidden) visible += 1;
    }

    resultCount.textContent = `${visible} ${visible === 1 ? 'entry' : 'entries'}`;
    emptyState.hidden = visible !== 0;
  };

  search.addEventListener('input', applyFilters);
  collection.addEventListener('change', applyFilters);
  clear?.addEventListener('click', () => {
    search.value = '';
    collection.value = 'all';
    applyFilters();
    search.focus();
  });

  for (const button of document.querySelectorAll('[data-collection-button]')) {
    button.addEventListener('click', () => {
      collection.value = button.dataset.collectionButton;
      applyFilters();
      document.querySelector('#archive')?.scrollIntoView({ behavior: 'smooth' });
    });
  }

  document.addEventListener('keydown', (event) => {
    if (event.key === '/' && document.activeElement !== search) {
      event.preventDefault();
      search.focus();
    }
  });

  fetch('search-index.json')
    .then((response) => response.ok ? response.json() : Promise.reject(new Error(`search index: ${response.status}`)))
    .then((records) => {
      for (const record of records) fullText.set(record.id, record.text);
      applyFilters();
    })
    .catch(() => {
      // Title/path/excerpt search remains available when viewed directly from file://.
    });
})();
