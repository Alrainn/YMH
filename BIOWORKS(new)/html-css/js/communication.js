// Canlı saat
function updateClock() {
    const now = new Date();
    let h = now.getHours().toString().padStart(2, '0');
    let m = now.getMinutes().toString().padStart(2, '0');
    document.getElementById('sideClock').textContent = h + ':' + m;
}
setInterval(updateClock, 1000); updateClock();
// Motive edici sözler
const quotes = [
    'Başarı, cesaretin çocuğudur.',
    'Birlikte daha güçlüyüz!',
    'Her gün yeni bir başlangıçtır.',
    'Takım ruhu her şeydir.',
    'Hayal et, dene, başar!',
    'Kod yaz, hata bul, öğren!',
    'Gülümse, nefes al, devam et!'
];
function randomQuote() {
    const q = quotes[Math.floor(Math.random()*quotes.length)];
    document.getElementById('sideQuote').textContent = q;
}
randomQuote();
// Canvas animasyonu (hareketli noktalar)
const canvas = document.getElementById('sideCanvas');
const ctx = canvas.getContext('2d');
let dots = Array.from({length: 18}, () => ({
    x: Math.random()*canvas.width,
    y: Math.random()*canvas.height,
    r: 6+Math.random()*6,
    dx: (Math.random()-0.5)*0.7,
    dy: (Math.random()-0.5)*0.7,
    c: `rgba(${60+Math.random()*120},${180+Math.random()*60},${60+Math.random()*120},0.5)`
}));
function animate() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for(let d of dots) {
        ctx.beginPath();
        ctx.arc(d.x,d.y,d.r,0,2*Math.PI);
        ctx.fillStyle = d.c;
        ctx.fill();
        d.x += d.dx; d.y += d.dy;
        if(d.x<0||d.x>canvas.width) d.dx*=-1;
        if(d.y<0||d.y>canvas.height) d.dy*=-1;
    }
    requestAnimationFrame(animate);
}
animate();
