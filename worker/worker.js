// WAAM digital twin: planning server (Cloudflare Worker).
//
// Holds the Anthropic API key so the web page never sees it, and forwards
// only WAAM planning requests from the page's own address to Claude.
//
// Settings (Cloudflare dashboard > the Worker > Settings > Variables and Secrets):
//   ANTHROPIC_API_KEY  secret, required  - a key from a workspace with a spend limit
//                      (the name APIKEY is also accepted)
//   ALLOWED_ORIGIN     text,   required  - e.g. https://riskerfru.github.io
//   MODEL              text,   optional  - default claude-haiku-4-5
//   MAX_PER_10_MIN     text,   optional  - planning calls per visitor per 10 min, default 20

const DEFAULT_MODEL = 'claude-haiku-4-5';
const PREFIX = 'You plan robotic wire-arc additive manufacturing (WAAM) builds';
const MAX_CHARS = 24000;      // whole conversation sent by the page
const MAX_TOKENS = 700;       // a plan is a short JSON object
const hits = new Map();       // per-visitor counts; best effort, per Worker instance

function json(body, status, headers) {
  return new Response(JSON.stringify(body), {
    status, headers: { ...headers, 'Content-Type': 'application/json' } });
}

function validMessages(msgs) {
  if (!Array.isArray(msgs) || msgs.length < 1 || msgs.length > 9) return false;
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    if (!m || typeof m.content !== 'string') return false;
    if (m.role !== (i % 2 === 0 ? 'user' : 'assistant')) return false;
  }
  if (msgs[msgs.length - 1].role !== 'user') return false;
  if (!msgs[0].content.startsWith(PREFIX)) return false;
  return msgs.reduce((n, m) => n + m.content.length, 0) <= MAX_CHARS;
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';
    const allowed = (env.ALLOWED_ORIGIN || '').split(',').map(s => s.trim()).filter(Boolean);
    const originOk = allowed.includes(origin);
    const cors = {
      'Access-Control-Allow-Origin': originOk ? origin : 'null',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Vary': 'Origin',
    };
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') return new Response(null, { status: originOk ? 204 : 403, headers: cors });
    if (url.pathname === '/' && request.method === 'GET') return json({ ok: true, service: 'waam-planner' }, 200, cors);
    if (url.pathname !== '/plan' || request.method !== 'POST') return json({ error: 'not found' }, 404, cors);
    if (!originOk) return json({ error: 'origin not allowed' }, 403, cors);
    const apiKey = env.ANTHROPIC_API_KEY || env.APIKEY;
    if (!apiKey) return json({ error: 'server has no API key set' }, 500, cors);

    const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
    const now = Date.now();
    const recent = (hits.get(ip) || []).filter(t => now - t < 600000);
    if (recent.length >= Number(env.MAX_PER_10_MIN || 20)) return json({ error: 'demo limit reached, try again in a few minutes' }, 429, cors);
    recent.push(now);
    hits.set(ip, recent);
    if (hits.size > 5000) hits.clear();

    let body;
    try { body = await request.json(); } catch { return json({ error: 'bad JSON' }, 400, cors); }
    if (!validMessages(body && body.messages)) return json({ error: 'not a WAAM planning request' }, 400, cors);

    let r;
    try {
      r = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
          'content-type': 'application/json',
        },
        body: JSON.stringify({ model: env.MODEL || DEFAULT_MODEL, max_tokens: MAX_TOKENS, messages: body.messages }),
      });
    } catch {
      return json({ error: 'could not reach Claude' }, 502, cors);
    }
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const type = (data.error && data.error.type) || 'upstream error';
      return json({ error: type }, r.status === 429 ? 429 : 502, cors);
    }
    const text = (data.content || []).filter(c => c.type === 'text').map(c => c.text).join('');
    return json({ text }, 200, cors);
  },
};
