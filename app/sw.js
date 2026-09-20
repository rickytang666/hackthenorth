const CACHE_NAME = 'voicebridge-offline-v5';
const OFFLINE_ASSETS = [
  './',
  './index.html',
  './offline.html',
  './demo.html',
  './race.html',
  './wer.html',
  './ab.html',
  './clarify.html',
  './hud.html',
  './meter.html',
  './static/demo.css',
  './static/report.css?v=3',
  './static/report.css?v=5',
  './static/single-page.js?v=1',
  './static/race.js',
  './static/diff.js',
  './static/clarify.js',
  './static/hud.js',
  './static/meter.js?v=3',
  './fixtures/session-m02.json',
  './fixtures/m02-source.wav',
  './fixtures/m02-personal-voice.wav',
  './fixtures/generic-tts.wav',
];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(OFFLINE_ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (response.ok && new URL(event.request.url).origin === self.location.origin) {
          const copy = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
        }
        return response;
      }).catch(() => {
        if (event.request.mode === 'navigate') return caches.match('./offline.html');
        return Response.error();
      });
    }),
  );
});
