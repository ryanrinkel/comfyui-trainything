// Drives the harness ComfyUI's own frontend headless (make_examples.py starts
// it):
//   convert <comfy> <in.json> <out.json>    an API prompt -> the canvas workflow the frontend saves
//   probe   <comfy> <workflow.json>         print the Batch Images order the frontend would queue
//   prompt  <comfy> <workflow.json> <out.json>  the API prompt the frontend would queue for a workflow
//   shoot   <comfy> <workflow.json> <daemon> <slug> <shots dir>
//                                            load a workflow, press Queue, screenshot the Review
//                                            node while it waits and after a scripted review
// playwright-core comes from the TrainyThing repo (TRAINYTHING_REPO); Chromium
// from PLAYWRIGHT_CHROMIUM or Playwright's cache, as that repo's test/ui does.
import { createRequire } from 'node:module';
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';

const repo = process.env.TRAINYTHING_REPO ?? path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../../trainything');
const { chromium, request } = createRequire(path.join(repo, 'package.json'))('playwright-core');

function chromiumPath() {
  if (process.env.PLAYWRIGHT_CHROMIUM) return process.env.PLAYWRIGHT_CHROMIUM;
  const cache = path.join(homedir(), '.cache', 'ms-playwright');
  const dirs = existsSync(cache) ? readdirSync(cache).filter((d) => /^chromium-\d+$/.test(d)).sort().reverse() : [];
  for (const dir of dirs) {
    const exe = path.join(cache, dir, 'chrome-linux64', 'chrome');
    if (existsSync(exe)) return exe;
  }
  throw new Error('no Chromium');
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function open(comfy, viewport = { width: 1600, height: 1000 }) {
  const browser = await chromium.launch({ executablePath: chromiumPath() });
  const page = await browser.newPage({ viewport });
  page.on('pageerror', (e) => console.error('pageerror:', e.message));
  await page.goto(comfy);
  await page.waitForFunction(() => window.app?.graph && window.app?.canvas, null, { timeout: 60000 });
  await sleep(1500);
  await page.keyboard.press('Escape');
  return { browser, page };
}

async function convert(comfy, input, output) {
  const { browser, page } = await open(comfy);
  const api = JSON.parse(readFileSync(input, 'utf8'));
  const wf = await page.evaluate(async (api) => {
    await window.app.loadApiJson(api, 'example');
    return window.app.graph.serialize();
  }, api);
  writeFileSync(output, JSON.stringify(wf, null, 1) + '\n');
  console.log(`wrote ${output} (${wf.nodes.length} nodes)`);
  await browser.close();
}

async function focus(page, type, scale = 1.0) {
  await page.evaluate(([type, scale]) => {
    const node = window.app.graph._nodes.find((n) => n.type === type);
    const c = window.app.canvas;
    c.ds.scale = scale;
    const [w, h] = [c.canvas.width / window.devicePixelRatio, c.canvas.height / window.devicePixelRatio];
    c.ds.offset[0] = -node.pos[0] - node.size[0] / 2 + w / (2 * c.ds.scale);
    c.ds.offset[1] = -node.pos[1] - node.size[1] / 2 + h / (2 * c.ds.scale);
    c.setDirty(true, true);
  }, [type, scale]);
  await sleep(800);
}

async function reviewer(daemon, slug, decide) {
  const ctx = await request.newContext({ baseURL: `${daemon}/api/r/${slug}/` });
  await ctx.post('join', { data: { name: 'Canvas', email: 'canvas@example.com' } });
  let n = 0;
  for (;;) {
    const next = await (await ctx.get('next')).json();
    if (next.done || !next.card) break;
    const i = Number(next.card.ref.split('-').pop());
    const verdict = decide(i);
    await ctx.post('judge', { data: { card_id: next.card.id, verdict, ...(verdict === 'reject' ? { note: `sprite ${i} is off-model` } : {}) } });
    n += 1;
  }
  await ctx.dispose();
  return n;
}

async function shoot(comfy, workflowFile, daemon, slug, shots) {
  const { browser, page } = await open(comfy, { width: 1300, height: 1500 });
  const wf = JSON.parse(readFileSync(workflowFile, 'utf8'));
  await page.evaluate(async (wf) => window.app.loadGraphData(wf), wf);
  await sleep(1000);
  await page.evaluate(async () => window.app.queuePrompt(0, 1));
  // wait until the queue exists on the daemon (the reviewer API answers)
  for (let k = 0; k < 120; k++) {
    const r = await fetch(`${daemon}/r/${slug}`).catch(() => null);
    if (r && r.status === 200) break;
    await sleep(500);
  }
  await sleep(4000);
  await focus(page, 'TrainyThingReview');
  await page.screenshot({ path: path.join(shots, 'review-node-waiting.png') });
  console.log('waiting shot taken');
  const judged = await reviewer(daemon, slug, (i) => (i < 8 ? 'accept' : 'reject'));
  console.log(`judged ${judged}`);
  for (let k = 0; k < 120; k++) {
    const q = await (await fetch(`${comfy}/queue`)).json();
    if (q.queue_running.length === 0 && q.queue_pending.length === 0) break;
    await sleep(500);
  }
  await sleep(1500);
  await focus(page, 'TrainyThingReview');
  await page.screenshot({ path: path.join(shots, 'review-node.png') });
  const info = await page.evaluate(() => {
    const node = window.app.graph._nodes.find((n) => n.type === 'TrainyThingReview');
    return {
      widgets: (node.widgets ?? []).map((w) => w.name),
      imgs: (node.imgs ?? []).length,
      size: node.size,
    };
  });
  console.log('node after run:', JSON.stringify(info));
  const hist = await (await fetch(`${comfy}/history`)).json();
  for (const h of Object.values(hist)) {
    const batch = Object.values(h.prompt[2]).find((n) => n.class_type === 'BatchImagesNode');
    const loads = h.prompt[2];
    console.log('executed batch order:', Object.entries(batch.inputs).map(([k, v]) => `${k}=${loads[v[0]].inputs.image}`).join(' '));
    console.log('status:', h.status.status_str, 'outputs:', Object.keys(h.outputs).length);
  }
  await browser.close();
}

// probe <comfy> <workflow.json>: the API prompt the frontend would queue for it
async function probe(comfy, workflowFile) {
  const { browser, page } = await open(comfy);
  const wf = JSON.parse(readFileSync(workflowFile, 'utf8'));
  const out = await page.evaluate(async (wf) => {
    await window.app.loadGraphData(wf);
    const p = await window.app.graphToPrompt();
    return p.output;
  }, wf);
  for (const n of Object.values(out)) {
    if (n.class_type === 'BatchImagesNode') console.log(Object.entries(n.inputs).map(([k, v]) => `${k}=${out[v[0]].inputs.image}`).join(' '));
  }
  await browser.close();
}
// prompt <comfy> <workflow.json> <out.json>: load a canvas workflow and write
// the API prompt Queue would send (proves the workflow loads in the frontend)
async function prompt(comfy, workflowFile, output) {
  const { browser, page } = await open(comfy);
  const wf = JSON.parse(readFileSync(workflowFile, 'utf8'));
  const out = await page.evaluate(async (wf) => {
    await window.app.loadGraphData(wf);
    const p = await window.app.graphToPrompt();
    return p.output;
  }, wf);
  writeFileSync(output, JSON.stringify(out, null, 1) + '\n');
  console.log(`wrote ${output} (${Object.keys(out).length} nodes)`);
  await browser.close();
}

const [cmd, ...args] = process.argv.slice(2);
const commands = { convert, shoot, probe, prompt };
if (!commands[cmd]) throw new Error(`unknown command ${cmd} (want ${Object.keys(commands).join(', ')})`);
await commands[cmd](...args);
