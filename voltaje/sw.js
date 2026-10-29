// Eureka Voltaje: funciona sin conexión (excepto el duelo online)
const CACHE='eureka-voltaje-202610291147';
const SHELL=['./','index.html','manifest.webmanifest','icon-192.png','icon-512.png','apple-touch-icon.png','vendor/peerjs.min.js'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL)).then(()=>self.skipWaiting()))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{
  const r=e.request; if(r.method!=='GET') return;
  const u=new URL(r.url);
  if(u.origin===location.origin){
    // la página: primero red (para recibir preguntas nuevas), si no hay conexión, la copia guardada
    if(r.mode==='navigate'){e.respondWith(fetch(r).then(res=>{const c=res.clone();caches.open(CACHE).then(x=>x.put(r,c));return res}).catch(()=>caches.match(r).then(m=>m||caches.match('index.html'))));return}
    e.respondWith(caches.match(r).then(m=>m||fetch(r)));
  } else if(u.hostname.endsWith('fonts.googleapis.com')||u.hostname.endsWith('fonts.gstatic.com')){
    e.respondWith(caches.open(CACHE).then(c=>c.match(r).then(m=>m||fetch(r).then(res=>{c.put(r,res.clone());return res}))));
  }
});
