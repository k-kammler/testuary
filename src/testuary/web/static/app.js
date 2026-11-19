// The pages of the Testuary web interface. Every value comes from the
// server's JSON interface, which reads Testuary's own outputs. Text is always
// inserted as text, never as HTML.

const view = document.getElementById("view");
const messageBox = document.getElementById("message");
let leaving = [];

// An interface started with an access key (--access key or stable_key)
// prints an address with the key in its fragment (#token=...), which the
// browser sends to no server. The page keeps it in the browser's local
// storage, which belongs to this server's origin (its port included), so
// every tab of the same running interface has it, and sends it as a header.
// A cookie would also reach every other server on 127.0.0.1, whatever its
// port. A key of an earlier start opens nothing then. An open interface
// needs no key: the page sends none.
const TOKEN_HEADER = "X-Testuary-Web-Token";
const TOKEN_KEY = "testuary-web-token";

// The token of the address's fragment, which is then stored, else the
// stored one. Without storage the token stays with this tab.
function pageToken(hash, storage) {
  const given = /^#token=([A-Za-z0-9_-]+)$/.exec(hash);
  if (given) {
    try { storage.setItem(TOKEN_KEY, given[1]); } catch (error) { /* kept for this tab only */ }
    return given[1];
  }
  try { return storage.getItem(TOKEN_KEY); } catch (error) { return null; }
}

let tokenStorage = null;
try { tokenStorage = window.localStorage; } catch (error) { tokenStorage = null; }
let token = pageToken(location.hash, tokenStorage);
if (token && location.hash === `#token=${token}`) history.replaceState(null, "", `${location.pathname}#/`);
// A new token can open what the old one could not, such as a page first
// opened without the key, so taking one reads the extensions again. The
// router draws the view of a pasted address, and a view open in this tab
// keeps what was typed into it.
// A later start opened in another tab stores its token: this tab takes it.
window.addEventListener("storage", (event) => {
  if (event.key !== TOKEN_KEY || !event.newValue) return;
  token = event.newValue;
  loadExtensions();
});
// A printed address pasted into an open tab changes only the fragment, so
// its token, from a later start on the same port, replaces the stored one
// here, before the view is drawn.
window.addEventListener("hashchange", () => {
  if (!/^#token=/.test(location.hash)) return;
  token = pageToken(location.hash, tokenStorage);
  if (token && location.hash === `#token=${token}`) history.replaceState(null, "", `${location.pathname}#/`);
  loadExtensions();
});

function tokenHeaders() {
  return token ? { [TOKEN_HEADER]: token } : {};
}

// Small helpers

// A part's children, which may come in lists of lists, without the parts
// left out (null, undefined or false).
function childList(children) {
  return [children].flat(Infinity).filter((child) => child !== undefined && child !== null && child !== false);
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "style") for (const [name, item] of Object.entries(value)) node.style.setProperty(name, String(item));
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of childList(children)) {
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function showMessage(text) {
  messageBox.textContent = text;
  messageBox.hidden = !text;
}

async function call(path, body) {
  const options = { headers: tokenHeaders(), credentials: "omit" };
  if (body !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  let data = null;
  try { data = await response.json(); } catch (error) { data = null; }
  if (!response.ok) {
    // A refusal can name the form field it concerns, and each problem's.
    const error = new Error((data && data.error) || `The server answered ${response.status}.`);
    error.field = data ? data.field : null;
    error.problems = (data && data.problems) || [];
    throw error;
  }
  return data;
}

// The running job, as the header shows it. Views that start jobs follow it.
const activeListeners = new Set();
let lastActive = null;

function onActive(listener) {
  activeListeners.add(listener);
  leaving.push(() => activeListeners.delete(listener));
  listener(lastActive);
  watchActive();
}

// Views that follow each look at the server's state, every few seconds.
const stateListeners = new Set();

function onState(listener) {
  stateListeners.add(listener);
  leaving.push(() => stateListeners.delete(listener));
}

// A job's stream: server-sent events, read with fetch because EventSource
// cannot send the token header. Each event goes to handlers[kind] as text.
async function follow(path, handlers, signal) {
  const response = await fetch(path, { headers: tokenHeaders(), credentials: "omit", signal });
  if (!response.ok || !response.body) {
    const refusal = await response.json().catch(() => null);
    throw new Error((refusal && refusal.error) || `The server answered ${response.status}.`);
  }
  handlers.open();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let kind = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) kind = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (data.length && handlers[kind]) handlers[kind](data.join("\n"));
    }
  }
}

function number(value) {
  return value === null || value === undefined ? "unknown" : Number(value).toLocaleString("en-US");
}

function cell(value) {
  return value === "" || value === null || value === undefined
    ? el("span", { class: "unknown", text: "unknown" }) : String(value);
}

// A table whose rows a narrow screen shows as stacked blocks (style.css,
// table.stack): each cell carries its column's name as its label. A row
// that spans the table, such as a target's details, carries none.
function stackedTable(headings, rows) {
  for (const row of rows.filter((line) => line.childNodes.length === headings.length)) {
    row.childNodes.forEach((item, index) => item.setAttribute("data-label", headings[index]));
  }
  return el("table", { class: "stack" },
    el("thead", {}, el("tr", {}, headings.map((text) => el("th", { scope: "col", text })))),
    el("tbody", {}, rows));
}

function stateBadge(word) {
  const label = { missing_setup: "missing setup" }[word] || word;
  return el("span", { class: `state ${word}`, text: label.toUpperCase() });
}

// What a job's exit status means is Testuary's sentence, which the server
// sends. A force-stopped job, whose status is the system's, has none.
function exitMeaning(job) {
  if (job.meaning) return job.meaning;
  const code = job.exit_code;
  return code === null || code === undefined || job.status === "stopped" ? "" : `Exit status ${code}.`;
}

// A job's text output, or what stands for none: nothing yet while it runs.
function outputText(lines, ended) {
  return lines.join("\n") || (ended ? "No output." : "No output yet.");
}

// The progress view draws the rows of Testuary's view model, as the console
// does. The page adds only its words for states and limits.
// A running row's time (hh:mm:ss) is the run's clock at its newest
// observation, which each view carries, advanced on the page between
// Testuary's events. A row's own time is its last observation's, which
// stands still while other work reports.
const CLOCK = /\b(\d{2}):(\d{2}):(\d{2})\b/;

// A duration or a clock reading as the console writes it, hh:mm:ss.
function hms(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(Math.floor(total / 3600))}:${pad(Math.floor(total / 60) % 60)}:${pad(total % 60)}`;
}

// The seconds a running row's time starts from: the run's clock when the
// view carries it, else the row's own, and null for a row whose time stands.
function clockBase(text, running, clock) {
  const match = running && CLOCK.exec(text || "");
  if (!match) return null;
  return Number.isFinite(clock) ? Math.floor(clock)
    : Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]);
}

function clocked(text, running, attributes = {}, clock = null) {
  const base = clockBase(text, running, clock);
  if (base === null) return el("span", { ...attributes, text });
  return el("span", { ...attributes, text: text.replace(CLOCK, hms(base)), "data-clock": String(base),
    "data-text": text });
}

// A nested run's insides belong to it (nestedRun): a page's drawing and
// clocks leave them to their own part.
function ownNode(root, node) {
  return node.closest(".nested-run") === root.closest(".nested-run");
}

function tickClocks(root, received) {
  const passed = Math.max(0, Math.floor((Date.now() - received) / 1000));
  for (const span of [...root.querySelectorAll("[data-clock]")].filter((node) => ownNode(root, node))) {
    span.textContent = span.dataset.text.replace(CLOCK, hms(Number(span.dataset.clock) + passed));
  }
}

// How long the diagram's current step has been under way, by the run's
// clock. It ticks with the rows.
function stepTime(item, clock) {
  if (item.state !== "running" || !Number.isFinite(item.since_s) || !Number.isFinite(clock)) return null;
  return `for ${hms(clock - item.since_s)}`;
}

// The heading and style of a run's cause by the run's overall tone. An ended
// job that is no failure in Testuary's words (refused, a problem the check
// found) says that word, as its badge does.
function causeHeading(tone, ended) {
  if (ended && ended.word && ended.failure === false) {
    return [`${ended.word[0].toUpperCase()}${ended.word.slice(1)}: `, "panel cause-warning"];
  }
  if (tone === "failed") return ["Why the run failed: ", "panel cause"];
  return ["What went wrong: ", "panel cause"];
}

// Whether a run's cause came from the model service, as the server marks
// it: the Models page then offers to test the model's connection.
function serviceCause(view) {
  return Boolean(view && view.cause_kind === "service");
}

class ViewPanel {
  constructor() {
    // The ended command's own error text, which the cause panel prefers.
    this.error = null;
    // Testuary's word for an ended job's status and whether it is a failure.
    this.endedWord = null;
    // Whether the job has ended: its diagram then shows no current step.
    this.ended = false;
    // An extension's job kind, which names its panel and has no repetitions.
    this.kind = null;
    // Its panels' heading level: one lower inside a nested run.
    this.heading = "h2";
    this.reset();
  }

  reset() {
    this.snapshot = null;
    this.log = [];
    this.summary = null;
  }

  // One line of what happened, the newest 2000 kept.
  addLine(line) {
    this.log.push(line);
    if (this.log.length > 2000) this.log.shift();
  }

  bar(bar, tone, label) {
    if (!bar) return null;
    return el("div", { class: "bar" },
      el("progress", { class: tone ? `tone-${tone}` : "", max: 1, value: bar.ratio, "aria-label": label }),
      clocked(bar.figures, tone === "running", { class: "figures" }, this.snapshot && this.snapshot.clock_s));
  }

  // Why the run failed: the ended command's own error text when it printed
  // one, else the view model's sentence.
  causePanel() {
    const view = this.snapshot;
    const cause = this.error || (view && view.cause);
    if (!cause || this.hideCause) return null;
    const [heading, style] = causeHeading(view && view.layers[0] ? view.layers[0].tone : null, this.endedWord);
    return el("p", { class: `${style} cause-text` }, el("b", { text: heading }), cause,
      !this.error && serviceCause(view) && tabShown("models") ? [" ", el("a", { href: "#/models", text: `Test the model's connection on the ${tabTitle("models")} page` }), "."] : null);
  }

  render() {
    const view = this.snapshot;
    const parts = [];
    const cause = this.causePanel();
    if (cause) parts.push(cause);
    if (view && view.diagram) parts.push(runDiagram(view.diagram, this.ended, view.clock_s, this.kind, this.nested, this.heading));
    if (!view || view.empty) {
      parts.push(el("p", { class: "muted", text: "Waiting for the run's first progress report." }));
    } else {
      const rows = el("div", { class: "panel view" }, el("div", { class: "view-title", text: panelTitle(view.title, this.kind) }),
        view.layers.map((layer) => el("div", { class: `layer${layer.dim ? " muted" : ""}${layer.tone ? ` tone-${layer.tone}` : ""}` },
          el("span", { class: "layer-name", text: layer.name }),
          layer.path ? el("span", { class: "layer-text" }, pathText(layer.path, layer.short))
            : el("span", { class: "layer-text", text: layer.text.slice(layer.name.length).trim() }),
          this.bar(layer.bar, layer.bar_tone, `${layer.name} progress`))));
      if (view.operations.length) {
        rows.append(el("div", { class: "scroll" }, stackedTable(
          ["State", "Operation", "Progress", view.playback ? "Playback" : "Elapsed"],
          view.operations.map((operation) => el("tr", {},
            el("td", {}, el("span", { class: `state${operation.tone ? ` ${operation.tone}` : ""}`, text: operation.state })),
            el("td", { text: operation.name }),
            el("td", {}, operation.bar ? this.bar(operation.bar, operation.tone, `${operation.name} progress`) : operation.progress),
            el("td", { class: "num" }, clocked(operation.elapsed, operation.tone === "running", {}, view.clock_s)))))));
      }
      if (view.hidden_active) rows.append(el("p", { class: "muted small", text: `${view.hidden_active} more running` }));
      for (const line of [view.message, view.metrics]) if (line) rows.append(el("p", { text: line }));
      if (view.usage) rows.append(el("p", { class: "muted small", text: view.usage }));
      for (const issue of view.issues) rows.append(el("p", { class: "tone-retrying", text: issue }));
      parts.push(rows);
    }
    if (!(view && view.diagram)) parts.push(nestedParts(this.nested));
    if (this.summary) {
      parts.push(el("div", { class: "panel" }, el(this.heading, { text: "End of the run" }),
        el("div", { class: "log", tabindex: 0, "aria-label": "End of the run", text: this.summary.join("\n") })));
    }
    parts.push(el("details", { class: "panel", open: true }, el("summary", { text: "What happened, in order" }),
      el("div", { class: "log", tabindex: 0, "aria-label": "What happened, in order", text: this.log.join("\n") || "Nothing yet." })));
    return el("div", {}, parts);
  }
}

// The run's phases and their steps, as Testuary's plan names them and its
// observations show them: the current one highlighted, finished ones
// marked, the rest gray, with the counters the observations carry. A
// phase's start runs once, its cycle repeats and its end runs once after it.
const STEP_WORDS = {
  planned: "upcoming", running: "now", waiting: "waiting", retrying: "retrying", succeeded: "done",
  failed: "failed", refused: "refused", cancelled: "stopped", off: "off", reused: "reused", skipped: "not run",
  limit_reached: "limit reached",
};

// A phase or step that was still under way when its job ended stopped
// there, so a stopped run shows no current step.
function shownState(state, ended) {
  return ended && ["running", "waiting", "retrying"].includes(state) ? "cancelled" : state;
}

// A phase's shown state: as shownState, and stopped once its steps show
// the stop (none under way, one stopped or not run), before the phase's own
// event arrives.
function phaseState(phase, ended) {
  const state = shownState(phase.state, ended);
  const steps = phase.steps.map((step) => shownState(step.state, ended));
  const underway = steps.some((step) => ["running", "waiting", "retrying"].includes(step));
  const stopped = steps.some((step) => step === "cancelled" || step === "skipped");
  return state === "running" && !underway && stopped ? "cancelled" : state;
}

// A step's state in words. A step a limit ended names that limit, as the
// diagram's "reached" says it, else its state's word when its event names
// no limit.
function stepWord(item) {
  return item.state === "limit_reached" && item.reached ? item.reached.toLowerCase()
    : STEP_WORDS[item.state] || item.state;
}

function stepPill(item, clock) {
  const time = stepTime(item, clock);
  return el("li", { class: `step state-${item.state}`, title: `${item.label}: ${stepWord(item)}` },
    el("span", { class: "step-label", text: item.label }),
    el("span", { class: "step-state", text: stepWord(item) }),
    item.count ? el("span", { class: "step-count", text: item.count }) : null,
    time ? clocked(time, true, { class: "step-count" }, clock - item.since_s) : null);
}

// A phase's steps in its three groups. A plan written before the phase end
// existed has none.
function phaseGroups(phase) {
  const end = phase.end ?? phase.steps.length;
  return { start: phase.steps.slice(0, phase.cycle), cycle: phase.steps.slice(phase.cycle, end),
    end: phase.steps.slice(end) };
}

function limitText(limit) {
  if (limit.value === null || limit.value === undefined) return `${limit.name}: none`;
  const value = Number(limit.value);
  if (limit.unit === "s") {
    return `${limit.name}: ${value >= 60 ? `${Math.round(value / 6) / 10} min` : `${value} s`}`;
  }
  return `${limit.name}: ${value}${limit.unit === "%" ? "%" : ` ${limit.unit}`}`;
}

// A path as a page shows it: short, the full path on hover and to copy.
function pathText(full, short) {
  const copy = el("button", { type: "button", class: "link-button small", text: "Copy path",
    onclick: async () => {
      try { await navigator.clipboard.writeText(full); copy.textContent = "Copied"; }
      catch (error) { copy.textContent = "Select the path to copy it"; }
    } });
  return el("span", { class: "path-text" }, el("code", { text: short || full, title: full }), " ", copy);
}

// Finishing, the fourth phase, plans no steps: while it runs, its box says
// what it does, in Testuary's words. A job that ended shows its end.
function finishingText(finishing, ended) {
  return finishing.state === "running" && !ended ? finishing.purpose : null;
}

// The folded parts of an extension's job's Testuary runs (``nested``, by
// path: phase and part) under ``phase``, or, without one, those whose phase
// is not in ``shown``, with the error of the function that names them.
function nestedParts(nested, phase, shown = []) {
  if (!nested) return null;
  const runs = [...nested.runs.values()].filter((run) =>
    (phase === undefined ? !shown.includes(run.phase) : run.phase === phase));
  return [...runs.map((run) => run.part),
    phase === undefined && nested.error ? el("p", { class: "error small", text: nested.error }) : null];
}

function runDiagram(diagram, ended, clock, kind = null, nested = null, heading = "h2") {
  const shown = (item) => ({ ...item, state: shownState(item.state, ended) });
  const phases = diagram.phases.map((given) => {
    const phase = { ...given, state: phaseState(given, ended) };
    const groups = phaseGroups(phase);
    const [start, cycle, end] = [groups.start, groups.cycle, groups.end].map((group) => group.map(shown));
    return el("li", { class: `phase state-${phase.state}` },
      el("div", { class: "phase-head" }, el("span", { class: "phase-label", text: phase.label }),
        el("span", { class: "step-state", text: STEP_WORDS[phase.state] || phase.state })),
      start.length ? el("ol", { class: "steps", "aria-label": `${phase.label}: start` }, start.map((item) => stepPill(item, clock))) : null,
      cycle.length ? el("div", { class: "cycle" }, el("span", { class: "cycle-mark", text: "repeats" }),
        el("ol", { class: "steps", "aria-label": `${phase.label}: cycle` }, cycle.map((item) => stepPill(item, clock)))) : null,
      end.length ? el("div", { class: "phase-end" }, el("span", { class: "cycle-mark", text: "at the end" }),
        el("ol", { class: "steps", "aria-label": `${phase.label}: end` }, end.map((item) => stepPill(item, clock)))) : null,
      phase.count ? el("p", { class: "small phase-count", text: phase.count }) : null,
      (phase.values || []).length ? el("p", { class: "small", text: phase.values.join(", ") }) : null,
      phase.note ? el("p", { class: "small", text: phase.note }) : null,
      phase.limits.length ? el("p", { class: "small muted", text: phase.limits.map(limitText).join(", ") }) : null,
      nestedParts(nested, phase.phase));
  });
  const finishing = shown(diagram.finishing);
  const purpose = finishingText(diagram.finishing, ended);
  phases.push(el("li", { class: `phase state-${finishing.state}` },
    el("div", { class: "phase-head" }, el("span", { class: "phase-label", text: finishing.label }),
      el("span", { class: "step-state", text: STEP_WORDS[finishing.state] || finishing.state })),
    purpose ? el("p", { class: "finishing small", role: "status" },
      el("span", { class: "spinner", "aria-hidden": "true" }), purpose) : null,
    finishing.steps.length ? el("ol", { class: "steps", "aria-label": `${finishing.label}: steps` },
      finishing.steps.map((item) => stepPill(shown(item), clock))) : null,
    finishing.values.length ? el("p", { class: "small", text: finishing.values.join(", ") }) : null));
  const where = [diagram.target ? `Target ${diagram.target}` : null,
    diagram.files.length > 1 ? `${diagram.files.indexOf(diagram.target) + 1} of ${diagram.files.length}` : null,
    // An extension's job has no repetitions, whatever its plan's runs say.
    diagram.runs.length && !kind ? `${diagram.runs.length === 1 ? "repetition" : "repetitions"} ${diagram.runs.join(", ")}` : null].filter(Boolean).join(", ");
  return el("div", { class: "panel diagram" }, el(heading, { text: "Phases and steps" }),
    where ? el("p", { class: "small muted", text: where }) : null,
    el("ol", { class: "phases" }, phases), nestedParts(nested, undefined, diagram.phases.map((phase) => phase.phase)));
}

// Forms: every field has a label, a hint that states its default and an
// error line below it, tied together for assistive technology. The server
// names the field a refusal concerns, and the page shows it there.

// The terminal's option of a field, on hover only: the page's hints are plain.
function optionName(option) {
  return option ? { title: `At the terminal: ${option}` } : {};
}

// The directory browser: a dialog over the page that lists directory names from
// the server, since a browser cannot hand a local path to a page. A click or
// Enter opens a directory, and "Use this directory" takes the directory shown.
function chooseDir({ title, start }) {
  return new Promise((resolve) => {
    const opener = document.activeElement;
    const heading = el("h2", { id: "browse-title", text: title });
    const crumbs = el("nav", { class: "crumbs", "aria-label": "Directory path" });
    const places = el("div", { class: "actions places" });
    const list = el("ul", { class: "dirs", "aria-label": "Directories" });
    const status = el("p", { class: "hint", "aria-live": "polite" });
    const shown = el("span", { class: "small" });
    const use = el("button", { type: "button", class: "primary", text: "Use this directory", disabled: true });
    const cancel = el("button", { type: "button", text: "Cancel" });
    const dialog = el("div", { class: "dialog", role: "dialog", "aria-modal": "true", "aria-labelledby": "browse-title" },
      el("header", {}, heading, crumbs, places),
      el("div", { class: "body" }, list, status),
      el("footer", {}, shown, el("span", { class: "actions" }, cancel, use)));
    const backdrop = el("div", { class: "backdrop" }, dialog);
    let current = null;
    const close = (value) => {
      document.removeEventListener("keydown", keys, true);
      window.removeEventListener("hashchange", leave);
      backdrop.remove();
      if (opener && opener.focus) opener.focus();
      resolve(value);
    };
    const keys = (event) => {
      if (event.key === "Escape") { event.preventDefault(); close(null); return; }
      if (event.key !== "Tab") return;
      // The keyboard focus stays in the dialog.
      const stops = [...dialog.querySelectorAll("button:not(:disabled), a[href]")];
      if (!stops.length) return;
      const first = stops[0], last = stops[stops.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    const open = async (target) => {
      status.textContent = "Opening the directory.";
      let listing;
      try { listing = await call(`/api/dirs?path=${encodeURIComponent(target || "")}`); }
      catch (error) { status.textContent = error.message; return; }
      current = listing.path;
      crumbs.replaceChildren(el("button", { type: "button", class: "link-button", text: "Drives and places", onclick: () => open("") }),
        ...listing.crumbs.flatMap((crumb, index) => [el("span", { class: "sep", text: "/" }),
          index === listing.crumbs.length - 1 ? el("b", { text: crumb.name })
            : el("button", { type: "button", class: "link-button", text: crumb.name, onclick: () => open(crumb.path) })]));
      places.replaceChildren(...listing.shortcuts.map((place) =>
        el("button", { type: "button", text: place.name, onclick: () => open(place.path) })));
      const items = listing.parent ? [el("li", {}, el("button", { type: "button", class: "dir up",
        text: "Up one directory", onclick: () => open(listing.parent) }))] : [];
      list.replaceChildren(...items, ...listing.dirs.map((dir) => el("li", {},
        el("button", { type: "button", class: "dir", onclick: () => open(dir.path) },
          el("span", { text: dir.name }),
          dir.project ? el("span", { class: "hint", text: `project ${dir.project}` }) : null))));
      shown.replaceChildren(current ? el("span", {}, "This directory: ", el("b", { text: current })) : "Choose a drive or place.");
      use.disabled = !current;
      status.textContent = listing.truncated ? `The first ${listing.dirs.length} directories are listed.`
        : listing.dirs.length ? "" : "No directories inside.";
      (list.querySelector("button") || cancel).focus();
    };
    // A page change (the browser's Back button) closes it, choosing nothing.
    const leave = () => close(null);
    cancel.addEventListener("click", () => close(null));
    use.addEventListener("click", () => close(current));
    backdrop.addEventListener("click", (event) => { if (event.target === backdrop) close(null); });
    document.addEventListener("keydown", keys, true);
    window.addEventListener("hashchange", leave);
    document.body.append(backdrop);
    open(start || "");
  });
}

function field({ id, label, option, input, unit, hint, browse }) {
  input.id = id;
  if (!input.name) input.name = id;
  input.setAttribute("aria-describedby", [hint ? `${id}-hint` : null, `${id}-error`].filter(Boolean).join(" "));
  const picker = browse ? el("div", { class: "picker" }, input, el("button", { type: "button", text: "Browse",
    "aria-label": `Browse for the ${label.toLowerCase()}`, onclick: async () => {
      // A relative path starts at the working directory, as the server lists full paths only.
      const typed = input.value.trim();
      const absolute = typed.startsWith("/") || /^[A-Za-z]:[\\/]/.test(typed);
      const chosen = await chooseDir({ title: `Choose the ${label.toLowerCase()}`, start: absolute ? typed : browse });
      if (chosen) { input.value = chosen; input.dispatchEvent(new Event("input", { bubbles: true })); }
    } })) : null;
  return el("div", { class: "field", "data-field": id },
    el("label", { for: id, ...optionName(option) }, label),
    picker || (unit ? el("div", { class: "with-unit" }, input, el("span", { class: "unit", text: unit })) : input),
    hint ? el("span", { class: "hint", id: `${id}-hint`, text: hint }) : null,
    el("span", { class: "error", id: `${id}-error`, "aria-live": "polite" }));
}

// A group of radio buttons. The first option is the default.
function choices({ name, legend, option, options, hint }) {
  return el("fieldset", { "data-field": name, "aria-describedby": hint ? `${name}-hint ${name}-error` : `${name}-error` },
    el("legend", optionName(option), legend),
    options.map(([value, text], index) => el("div", { class: "check" },
      el("input", { type: "radio", name, id: `${name}-${value}`, value, checked: index === 0 }),
      el("label", { for: `${name}-${value}`, text }))),
    hint ? el("span", { class: "hint", id: `${name}-hint`, text: hint }) : null,
    el("span", { class: "error", id: `${name}-error`, "aria-live": "polite" }));
}

function checkbox({ id, label, option, hint }) {
  return el("div", { "data-field": id },
    el("div", { class: "check" },
      el("input", { type: "checkbox", id, name: id, "aria-describedby": hint ? `${id}-hint` : null }),
      el("label", { for: id, ...optionName(option) }, label)),
    hint ? el("span", { class: "hint", id: `${id}-hint`, text: hint }) : null);
}

function clearErrors(form) {
  for (const box of form.querySelectorAll(".error")) box.textContent = "";
  for (const input of form.querySelectorAll("[aria-invalid]")) input.removeAttribute("aria-invalid");
}

function showError(form, error) {
  const box = error.field ? form.querySelector(`[data-field="${CSS.escape(error.field)}"]`) : null;
  const target = box || form.querySelector(".actions");
  target.querySelector(".error").textContent = error.message;
  const input = box ? box.querySelector("input:checked, input") : null;
  if (input) {
    if (input.type === "text") input.setAttribute("aria-invalid", "true");
    input.focus();
  }
}

// A form's primary button starts a job. It is disabled while the request is
// on its way and while another job runs, which the note beside it names.
// ``ready`` tells whether the form can start a job, such as a project
// being there to start it in.
function jobForm(form, label, request, ready = () => true) {
  const button = el("button", { type: "submit", class: "primary", text: label });
  const note = el("span", { class: "hint", "aria-live": "polite" });
  form.append(el("div", { class: "actions" }, button, note,
    el("span", { class: "error", id: `${form.id}-error`, "aria-live": "polite" })));
  let sending = false;
  let active = null;
  const update = () => {
    button.disabled = sending || Boolean(active) || !ready();
    note.replaceChildren(...(active ? ["A job is running: ", el("a", { href: `#/job/${active.id}`, text: active.title }),
      ". Another starts when it has ended."] : []));
  };
  onActive((job) => { active = job; update(); });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearErrors(form);
    sending = true;
    update();
    try {
      const job = await call(...request());
      location.hash = `#/job/${job.id}`;
      watchActive();
    } catch (error) {
      showError(form, error);
    } finally {
      sending = false;
      update();
    }
  });
  return form;
}

// Views

// The projects are project directories: the working directory's project (or, in a
// directory of none, the projects directly below it) and the recent ones, each
// with its settings as Testuary's setup reads them.

// A project's settings by Testuary's plain names, the file's names on hover.
function fieldList(fields, labels) {
  const entries = Object.entries(fields || {});
  if (!entries.length) return el("p", { class: "small muted", text: "No settings of its own." });
  return el("dl", { class: "fields" }, entries.flatMap(([name, value]) => [
    el("dt", { text: (labels || {})[name] || name, title: name }), el("dd", {}, ABSOLUTE.test(String(value))
      ? el("span", { text: shortPath(String(value)), title: String(value) }) : String(value))]));
}

const ABSOLUTE = /^([A-Za-z]:[\\/]|\/)/;

// A path inside a directory as relative to it, any other path short.
function inDir(dir, path) {
  const base = String(dir || "").replace(/[\\/]+$/, "");
  const head = path.slice(0, base.length + 1);
  if (base && head.length > base.length && head.slice(0, -1).toLowerCase() === base.toLowerCase()
      && /[\\/]/.test(head.slice(-1))) {
    return path.slice(base.length + 1).replace(/\\/g, "/");
  }
  return shortPath(path);
}

function catalogueError(catalogue) {
  return catalogue.error ? el("p", { class: "panel error", text: catalogue.error }) : null;
}

const dirQuery = (entry) => `project_dir=${encodeURIComponent(entry.project_dir)}`;
// A directory type's view of a directory: the view reads the directory as dir.
const dirTypeAddress = (type, dir) => `#/dir?type=${encodeURIComponent(type)}&dir=${encodeURIComponent(dir)}`;

function projectActions(entry) {
  return el("div", { class: "actions" },
    tabShown("run") && el("a", { class: "button", href: `#/run?${dirQuery(entry)}`, text: "Run it" }),
    tabShown("check") && el("a", { class: "button", href: `#/check?${dirQuery(entry)}`, text: "Check the setup" }));
}

function projectHeading(entry) {
  return el("div", {}, el("h3", {}, entry.name, entry.working
    ? el("span", { class: "default-mark", text: "this directory's project" })
    : entry.below ? el("span", { class: "default-mark", text: "in this directory" }) : null),
    el("p", { class: "small muted" }, pathText(entry.project_dir, shortPath(entry.project_dir))));
}

// A long path's last directories, for headings and lists. The full path stays on hover.
function shortPath(path) {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.length > 3 ? `.../${parts.slice(-2).join("/")}` : path;
}

// A project's newest results, as Testuary's summary of them states them:
// each file with its runs and, for each value, the mean across them.
function latestResults(latest, dir) {
  if (!latest) return el("p", { class: "small muted", text: "No results in the project's results directory yet." });
  const when = new Date(latest.written * 1000).toLocaleString();
  const value = (item, id) => {
    const across = item.values[id];
    const mean = across && across.mean;
    if (!mean && id === "mutation_score" && item.lower_bound) {
      return el("span", { class: "warn", text: `at least ${item.lower_bound.text} (not fully measured)` });
    }
    return mean ? mean.text : el("span", { class: "muted", text: "not measured" });
  };
  const open = tabShown("results") ? [": ", el("a", { href: `#/results?run_dir=${encodeURIComponent(latest.run_dir)}`, text: "open" })] : ".";
  const measured = latest.files.some((item) => item.lower_bound || latest.values.some((column) =>
    item.values[column.id] && item.values[column.id].mean));
  // A latest run that measured nothing, such as one that could not reach its
  // model, is said so, without a table of empty values.
  if (!measured) {
    return el("p", { class: "small" }, `The latest run, written ${when}, has no suite that meets the minimums`, open);
  }
  return el("div", { class: "latest" },
    el("p", { class: "small" }, tabShown("results") ? "Latest results, written " : "The latest results were written ", when, open),
    latest.files.length ? el("div", { class: "scroll" }, el("table", { class: "small" },
      el("thead", {}, el("tr", {}, ["Target", "Repetitions", ...latest.values.map((item) => `${item.name}, mean`)]
        .map((title) => el("th", { scope: "col", text: title })))),
      el("tbody", {}, latest.files.map((item) => el("tr", {},
        el("td", {}, el("code", { text: item.file ? inDir(dir, item.file) : "unknown", title: item.file || "" })),
        el("td", { class: "num", text: item.runs === null ? "unknown" : `${item.fully_measured_runs ?? "unknown"} of ${item.runs} fully measured` }),
        latest.values.map((column) => el("td", { class: "num" }, value(item, column.id)))))))) : null);
}

function projectEntry(entry) {
  return el("div", { class: "panel entry" }, projectHeading(entry),
    entry.error ? el("p", { class: "error", text: entry.error }) : fieldList(entry.fields, entry.labels),
    entry.error ? null : entry.latest_error ? el("p", { class: "error", text: entry.latest_error })
      : latestResults(entry.latest, entry.project_dir),
    projectActions(entry));
}

// Each file's values as the job's end-of-run summary states them, one line
// per file, which names its file when the job has several. A file without
// a complete suite says how it ended. A demonstration's values are demo
// values.
function valueLines(job) {
  const files = job.values || [];
  const demo = job.kind === "demo" ? " (demo values)" : "";
  return files.map((item) => {
    const values = item.values.map((value) => `${value.name} ${value.text === null ? "not measured" : value.text}`).join(", ");
    const ended = item.outcome === "complete" ? values
      : values ? `${item.outcome_text}: ${values}` : item.outcome_text;
    const line = values ? `${ended}${demo}` : ended;
    return files.length > 1 ? `${item.file}: ${line}` : line;
  });
}

// The run directory a job writes into or wrote, by its last part, the
// full path on hover. An ended job's opens on the results page.
function dirName(path) {
  return path.split(/[\\/]/).filter(Boolean).pop();
}

function jobDir(job) {
  if (!job.run_dir) return null;
  return job.ended !== null && tabShown("results")
    ? el("a", { href: `#/results?run_dir=${encodeURIComponent(job.run_dir)}`, title: job.run_dir, text: dirName(job.run_dir) })
    : el("span", { title: job.run_dir, text: dirName(job.run_dir) });
}

// The model a job names: a demonstration's is the recording's.
function jobModel(job) {
  if (!job.model) return "";
  return job.kind === "demo" ? `${job.model} (recorded)` : job.model;
}

function jobRow(job) {
  const dir = jobDir(job);
  const model = jobModel(job);
  return el("li", {}, el("span", {}, el("a", { href: `#/job/${job.id}`, text: job.title }),
    el("br"), el("span", { class: "small muted", text: `${new Date(job.started * 1000).toLocaleTimeString()}, ${hms(job.elapsed_s)}` }),
    dir || model ? [el("br"), el("span", { class: "small muted" }, dir ? ["Results in ", dir] : null,
      dir && model ? ", " : null, model ? `model ${model}` : null)] : null,
    valueLines(job).map((line) => [el("br"), el("span", { class: "small", text: line })])),
    jobState(job).badge);
}

// What an open Overview redraws after a look at the server: its recent
// jobs when they changed or one runs (its time moves), and its projects'
// latest results once a job has ended since the last look.
function overviewChanges(before, state) {
  const jobs = (look) => JSON.stringify((look.jobs || []).map((job) => [job.id, job.status, job.ended]));
  const ended = Boolean(before && before.active) && (!state.active || state.active.id !== before.active.id);
  return { jobs: !before || Boolean(state.active) || jobs(before) !== jobs(state), projects: ended };
}

// A part of the page drawn anew, the keyboard focus kept on the link it
// was on.
function redraw(box, ...children) {
  const focused = box.contains(document.activeElement) ? document.activeElement.getAttribute("href") : null;
  box.replaceChildren(...childList(children));
  const again = focused && [...box.querySelectorAll("a[href]")].find((link) => link.getAttribute("href") === focused);
  if (again) again.focus();
}

function recentJobs(jobs) {
  return jobs.length ? el("ul", { class: "list" }, jobs.slice(0, 5).map(jobRow))
    : el("p", { class: "muted", text: "No job started yet in this session." });
}

function projectEntries(listed) {
  const projects = listed.projects || [];
  return [catalogueError(listed), projects.length ? projects.map(projectEntry)
    : el("p", { class: "panel muted" }, "No project yet.",
      tabShown("projects") && [" ", el("a", { href: "#/projects", text: "Add a project's directory" }), " to start."])];
}

async function home() {
  // A catalogue that cannot be read leaves the rest of the overview.
  const [state, catalogue, listed] = await Promise.all([call("/api/state"), call("/api/catalogue")
    .catch((error) => ({ error: error.message, models: {} })),
  call("/api/projects").catch((error) => ({ error: error.message, projects: [] }))]);
  const models = Object.entries(catalogue.models || {});
  // The jobs and the projects' latest results follow each look at the server.
  const jobsBox = el("div", { class: "panel" }, recentJobs(state.jobs));
  const projectsBox = el("div", {}, projectEntries(listed));
  let before = state;
  onState(async (look) => {
    const changes = overviewChanges(before, look);
    before = look;
    if (changes.jobs) redraw(jobsBox, recentJobs(look.jobs));
    if (changes.projects) {
      const again = await call("/api/projects").catch(() => null);
      if (again && projectsBox.isConnected) redraw(projectsBox, projectEntries(again));
    }
  });
  return [
    refusedExtensions(),
    el("div", { class: "hero" },
      el("div", {}, el("h1", { text: `${productTitle()} in this directory` }),
        el("p", { class: "muted" }, pathText(state.base_dir, shortPath(state.base_dir)))),
      el("div", { class: "actions" },
        // Until a project exists, adding one is the way on.
        (listed.projects || []).length ? tabShown("run") && el("a", { class: "button primary", href: "#/run", text: "Start a run" })
          : tabShown("projects") && el("a", { class: "button primary", href: "#/projects", text: "Add a project" }),
        tabShown("check") && el("a", { class: "button", href: "#/check", text: "Check a project" }),
        tabShown("results") && el("a", { class: "button", href: "#/results", text: `Open ${tabTitleInText("results")}` }))),
    catalogueError(catalogue),
    el("div", { class: "grid-2" },
      el("section", { "aria-labelledby": "projects-title" },
        el("div", { class: "section-head" }, el("h2", { id: "projects-title", text: "Projects" }),
          tabShown("projects") && el("a", { class: "small", href: "#/projects", text: "All projects" })),
        projectsBox),
      el("aside", {},
        el("div", { class: "section-head" }, el("h2", { id: "models-title", text: "Models" }),
          tabShown("models") && el("a", { class: "small", href: "#/models", text: "All models" })),
        el("div", { class: "panel" }, models.length
          ? el("ul", { class: "list", "aria-labelledby": "models-title" }, models.map(([alias, fields]) => el("li", {},
            el("span", {}, el("b", { text: alias }), modelMarks(alias, catalogue.default_model,
              catalogue.project_model).map((text) => el("span", { class: "default-mark", text: ` ${text}` })),
            el("br"), el("span", { class: "small muted", text: [fields.provider, fields.model].filter(Boolean).join(", ") })))))
          : el("p", { class: "muted" }, "No model yet.", tabShown("models") && [" ", el("a", { href: "#/models", text: `Add one on the ${tabTitle("models")} page` }), "."])),
        el("div", { class: "section-head" }, el("h2", { text: "Recent jobs" }),
          tabShown("jobs") && el("a", { class: "small", href: "#/jobs", text: "All jobs" })),
        jobsBox,
        el("p", { class: "demo-note" }, "To see what a run shows before setting one up, ",
          el("a", { href: "#/demo", text: "play the demonstration" }),
          ": one recorded run of this release, no model, no cost."))),
  ];
}

// The model catalogue and a project's settings, read and changed through
// Testuary's setup (testuary.setup): every check and every message is
// Testuary's, each shown beside the field it concerns.

// Each problem of a refusal beside its field. One the form has no field
// for, and the message without problems, show at the form's end.
function showProblems(form, error) {
  const problems = error.problems && error.problems.length ? error.problems : [[error.field, error.message]];
  let focused = false;
  for (const [field, text] of problems) {
    const box = field ? form.querySelector(`[data-field="${CSS.escape(field)}"]`) : null;
    const target = (box && box.querySelector(".error")) || form.querySelector(".actions .error");
    target.textContent = target.textContent ? `${target.textContent} ${text}` : text;
    const input = box ? box.querySelector("input, select") : null;
    if (input) {
      input.setAttribute("aria-invalid", "true");
      if (!focused) { input.focus(); focused = true; }
    }
  }
}

// A form whose primary button saves through ``path``, then reloads the view.
function saveForm(form, label, request, after = () => render(), cancelled = () => render()) {
  const button = el("button", { type: "submit", class: "primary", text: label });
  const cancel = el("button", { type: "button", text: "Cancel", onclick: () => cancelled() });
  form.append(el("div", { class: "actions" }, button, cancel,
    el("span", { class: "error", "aria-live": "polite" })));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearErrors(form);
    button.disabled = true;
    try {
      const body = request();
      await call(...body);
      after(body[1]);
    } catch (error) {
      showProblems(form, error);
    } finally {
      button.disabled = false;
    }
  });
  return form;
}

// Removal asks first, in place of the entry's actions.
function removeAction(actions, question, path, body = {}, label = "Remove") {
  const ask = el("button", { type: "button", class: "danger", text: label });
  ask.addEventListener("click", () => {
    const note = el("span", { class: "error", "aria-live": "polite" });
    const yes = el("button", { type: "button", class: "danger", text: label });
    const no = el("button", { type: "button", text: "Keep it" });
    const original = [...actions.childNodes];
    no.addEventListener("click", () => { actions.replaceChildren(...original); ask.focus(); });
    yes.addEventListener("click", async () => {
      yes.disabled = true;
      try { await call(path, body); render(); }
      catch (error) { note.textContent = error.message; yes.disabled = false; }
    });
    actions.replaceChildren(el("span", { text: question }), yes, no, note);
    no.focus();
  });
  return ask;
}

const KEY_SOURCES = {
  "models.ini": "stored in models.ini",
  environment: "from the environment variable",
  "not needed": "not needed: a local address",
  missing: "missing",
};

function keyText(entry, list) {
  const words = KEY_SOURCES[entry.key_source] || entry.key_source;
  if (entry.key_source === "environment") return `${words} ${entry.key_variable}`;
  if (entry.key_source === "missing") return `${words}: set ${entry.key_variable}${list ? "" : " or store a key"}`;
  if (list && entry.key_source === "models.ini") return "stored in your model catalogue";
  return words;
}

// The eye icon of a masked field: an open eye while the value is hidden,
// struck through while it is shown. Drawn, so it needs no font or file.
const SVG_NS = "http://www.w3.org/2000/svg";
function eyeIcon(shown) {
  const svg = document.createElementNS(SVG_NS, "svg");
  for (const [name, value] of [["viewBox", "0 0 24 24"], ["aria-hidden", "true"], ["focusable", "false"]]) {
    svg.setAttribute(name, value);
  }
  const shape = (tag, attributes) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    svg.append(node);
  };
  shape("path", { d: "M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" });
  shape("circle", { cx: "12", cy: "12", r: "3" });
  if (shown) shape("path", { d: "M4 4l16 16" });
  return svg;
}

// A masked field with its eye inside it: the eye shows and hides the value
// typed into ``input``.
function secretField(input, label) {
  const eye = el("button", { type: "button", class: "eye", "aria-pressed": "false",
    "aria-label": `Show ${label}`, title: `Show ${label}` });
  const draw = (shown) => {
    input.type = shown ? "text" : "password";
    eye.replaceChildren(eyeIcon(shown));
    eye.setAttribute("aria-pressed", String(shown));
    eye.setAttribute("aria-label", `${shown ? "Hide" : "Show"} ${label}`);
    eye.title = `${shown ? "Hide" : "Show"} ${label}`;
  };
  draw(false);
  eye.addEventListener("click", () => draw(eye.getAttribute("aria-pressed") !== "true"));
  return el("span", { class: "secret" }, input, eye);
}

// The provider name after the service changes to ``chosen``: a built-in
// service's own. For another OpenAI-compatible API, the name typed for it,
// the model's own when it is one, else none, never the name of the
// service chosen before.
function providerName(chosen, typed, previous, entry) {
  if (chosen.kind !== "openai-compatible") return chosen.provider;
  if (previous === chosen.kind) return typed;
  return entry && entry.kind === chosen.kind ? entry.provider : "";
}

function modelForm(view, entry, after, cancelled) {
  const form = el("form", { class: "panel", id: "model-form", novalidate: true });
  const providers = view.providers;
  // An existing model opens with the kind of service it is, as the server
  // judges it, so that Save without a change changes nothing.
  const current = (entry && providers.find((kind) => kind.kind === entry.kind)) || providers[0];
  const alias = el("input", { type: "text", value: entry ? entry.alias : "", autocomplete: "off",
    readonly: entry ? true : undefined });
  const kind = el("select", {}, providers.map((item) => el("option", { value: item.kind, text: item.label,
    selected: item === current || undefined })));
  const provider = el("input", { type: "text", value: entry ? entry.provider : current.provider, autocomplete: "off" });
  const model = el("input", { type: "text", value: entry ? entry.model : "", autocomplete: "off" });
  const address = el("input", { type: "text", value: entry ? entry.address : current.address, autocomplete: "off" });
  const storage = choices({ name: "storage", legend: "The key", options: [
    ["environment", "Read it from an environment variable"], ["file", "Store it in models.ini, readable by you only"]] });
  const variable = el("input", { type: "text", autocomplete: "off",
    value: entry && entry.key_variable ? entry.key_variable : "" });
  const key = el("input", { type: "password", autocomplete: "new-password", value: "" });
  const keyField = field({ id: "key", label: "Key", input: key,
    hint: entry && entry.key_source === "models.ini"
      ? "A key is stored. A key typed here replaces it, and an empty field keeps it."
      : "Sent once, when you save." });
  const place = el("span");
  key.replaceWith(place);
  place.replaceWith(secretField(key, "the key you type"));
  // A list's last file takes a key variable only, and in the project
  // directory neither an address nor a key variable. A copy of a section
  // whose key an earlier file holds needs one: the alias's own, or its
  // provider's once a new address copies the provider's section.
  const list = view.list;
  const inProject = Boolean(list && list.in_project);
  const copyNote = el("p", { class: "small muted", hidden: true });
  const variableField = field({ id: "key_variable", label: "Environment variable", input: variable,
    hint: "Empty: Testuary's own, TESTUARY_<PROVIDER>_API_KEY." });
  const variableHint = variableField.querySelector(".hint");
  const keyHeld = () => entry && (entry.key_file || (address.value.trim() !== entry.address ? entry.provider_key_file : null));
  const markNeeded = () => {
    const held = list && keyHeld();
    variable.required = Boolean(held);
    variableHint.textContent = held ? `The key in ${fileWords(held, view)} is not copied into `
      + `${shortPath(list.file)}. Name the variable that holds it.` : "Empty: Testuary's own, TESTUARY_<PROVIDER>_API_KEY.";
    copyNote.hidden = !(list && entry && entry.provider_from && entry.provider_from !== list.file
      && kind.value === "openai-compatible" && address.value.trim() !== entry.address);
    copyNote.textContent = `Saving also copies the provider section ${provider.value} into ${shortPath(list ? list.file : "")}, `
      + `where it serves every model of ${provider.value}.`;
  };
  address.addEventListener("input", markNeeded);
  const providerField = field({ id: "provider", label: "Provider name", input: provider,
    hint: "It names the provider's section in models.ini." });
  let shownKind = current.kind;
  const sync = () => {
    const chosen = providers.find((item) => item.kind === kind.value);
    const custom = chosen.kind === "openai-compatible";
    providerField.hidden = !custom;
    provider.value = providerName(chosen, provider.value, shownKind, entry);
    shownKind = chosen.kind;
    if (!entry) address.value = chosen.address;
    const needsKey = chosen.key !== "none";
    storage.hidden = !needsKey || Boolean(list);
    const file = !list && storage.querySelector("input:checked").value === "file";
    keyField.hidden = !needsKey || !file;
    variableField.hidden = !needsKey || file || inProject;
  };
  if (entry && entry.key_source === "models.ini") storage.querySelector("#storage-file").checked = true;
  kind.addEventListener("change", sync);
  storage.addEventListener("change", sync);
  form.append(el("h2", { text: entry ? `Edit ${entry.alias}` : "Add a model" }),
    ...(list && entry && entry.file !== list.file ? [el("p", { class: "small muted", text: `Saving copies this `
      + `model's section into ${shortPath(list.file)}, where it takes the place of the one in ${fileWords(entry.file, view)}.` })] : []),
    field({ id: "alias", label: "Alias", input: alias,
      hint: "The name projects and runs use: lowercase letters, digits and _." }),
    field({ id: "kind", label: "Service", input: kind }),
    providerField,
    field({ id: "model", label: "Model", input: model, hint: "The model's name as the service lists it." }),
    ...(inProject ? [] : [field({ id: "address", label: "Address", input: address,
      hint: "The service's API address. Empty: the provider's configured one." })]),
    copyNote, storage, variableField, keyField);
  sync();
  markNeeded();
  return saveForm(form, entry ? "Save" : "Add", () => {
    const file = !list && storage.querySelector("input:checked").value === "file";
    const body = { alias: alias.value, provider: provider.value, model: model.value,
      address: inProject ? "" : address.value, new: !entry, ...listRequest(list) };
    if (file) body.key = key.value;
    else body.key_variable = inProject ? "" : variable.value;
    return ["/api/models", body];
  }, after, cancelled);
}

// What removing a model takes out of models.ini: its own section, with its
// model and any address of its own, and nothing else. A provider section of
// its own stays, which a later model or a run with [api] provider can use.
function removeQuestion(entry) {
  return entry.provider_section
    ? `Remove ${entry.alias} from models.ini? Its provider's section, ${entry.provider}, stays with its address and any stored key: a later model, or a run with [api] provider = ${entry.provider}, can use it.`
    : `Remove ${entry.alias} from models.ini? Its section goes, with its model and any address of its own. The ${entry.provider} provider has no section in models.ini, so nothing else changes.`;
}

// A model's badge says what the catalogue tells: set up for a run, or no
// key. Whether its service answers only a connection test tells.
function modelBadge(entry) {
  return entry.usable ? ["succeeded", "SET UP"] : ["incomplete", "NO KEY"];
}

// A connection test's style: answered and listed, answered without the
// model, or no answer.
function testTone(answer) {
  return answer.listed ? "saved" : answer.reached ? "warn" : "error";
}

// The connection test: the service the run would ask, asked for the models
// it serves, which generates nothing.
function connectionTest(alias, body = {}) {
  const said = el("p", { class: "small", role: "status" });
  const button = el("button", { type: "button", text: "Test the connection" });
  button.addEventListener("click", async () => {
    button.disabled = true;
    said.className = "small muted";
    said.textContent = "Asking the service for its models.";
    try {
      const answer = await call(`/api/models/${encodeURIComponent(alias)}/test`, body);
      said.className = `small ${testTone(answer)}`;
      said.textContent = answer.text;
    } catch (error) {
      said.className = "small error";
      said.textContent = error.message;
    } finally { button.disabled = false; }
  });
  return [button, said];
}

// The Models page's address: the user directory's
// catalogue, or a list of catalogue files a settings group names for a
// project (its scope).
function modelsLink(projectDir, scope) {
  return scope ? `#/models?project_dir=${encodeURIComponent(projectDir)}&scope=${encodeURIComponent(scope)}` : "#/models";
}

function listRequest(list) {
  return list ? { project_dir: list.project_dir, scope: list.scope } : {};
}

// The file a model's section comes from: the user directory's catalogue by
// its choice's name, linked to that choice, any other file short, its full
// path on hover and to copy.
function fileMark(file, view) {
  return file === view.own_file ? el("a", { class: "source source-global", href: modelsLink(), text: "Your model catalogue" })
    : el("span", { class: "source source-listed", title: file, text: shortPath(file) });
}

// A file inside a sentence: the user directory's catalogue in words, any
// other file short.
function fileWords(file, view) {
  return file === view.own_file ? "your model catalogue" : shortPath(file);
}

// Removing a model only an earlier file of the list defines ends the
// import: the question says what follows and asks for a key variable for
// each section whose key an earlier file holds. A section whose address
// holds a credential cannot be copied, and the question then cannot be
// confirmed.
function endImportAction(actions, entry, view) {
  const list = view.list;
  const earlier = list.files.slice(0, -1).map((file) => fileWords(file, view)).join(", ");
  const last = shortPath(list.file);
  const ask = el("button", { type: "button", class: "danger", text: "Remove" });
  ask.addEventListener("click", () => {
    // A last file in the project directory holds no address, key or key
    // variable, so every section that sets one blocks the end of an import.
    const others = (items) => items.filter((item) => item.section !== entry.alias);
    const keys = list.in_project ? [] : others(view.import_keys);
    const blocked = others(view.import_blocked);
    const places = list.in_project ? others(view.import_places) : [];
    const inputs = keys.map((item) => el("input", { type: "text", autocomplete: "off", required: true,
      placeholder: `TESTUARY_${item.section.toUpperCase()}_API_KEY`, "data-section": item.section }));
    const yes = el("button", { type: "button", class: "danger", text: "Remove and end the import" });
    const no = el("button", { type: "button", text: "Keep it" });
    const missing = el("p", { class: "small error", text: "Name a key variable for each key above." });
    const note = el("p", { class: "error", "aria-live": "polite" });
    const check = () => {
      const empty = inputs.some((input) => !input.value.trim());
      yes.disabled = Boolean(blocked.length || places.length) || empty;
      missing.hidden = !empty;
    };
    inputs.forEach((input) => input.addEventListener("input", check));
    const original = [...actions.childNodes];
    no.addEventListener("click", () => { actions.replaceChildren(...original); ask.focus(); });
    yes.addEventListener("click", async () => {
      yes.disabled = true;
      const variables = Object.fromEntries(inputs.map((input) => [input.dataset.section, input.value.trim()]));
      try {
        const answer = await call(`/api/models/${encodeURIComponent(entry.alias)}/remove`,
          { ...listRequest(list), key_variables: variables });
        if (!answer.warning) return render();
        actions.replaceChildren(el("p", { class: "warn", role: "status", text: answer.warning }),
          el("button", { type: "button", text: "OK", onclick: () => render() }));
      } catch (error) { note.textContent = error.message; check(); }
    });
    actions.replaceChildren(el("form", { novalidate: true, onsubmit: (event) => event.preventDefault() },
      el("p", { text: `Remove ${entry.alias}? Every other model is copied into ${last}, and the runs then read `
        + `${last} alone, not ${earlier}. Later changes in ${earlier} do not reach the copies. No earlier file is changed.` }),
      ...blocked.map((item) => el("p", { class: "error", text: `[${item.section}] in ${fileWords(item.file, view)} has `
        + `an address that holds a credential, which is not copied into ${last}. Change its address in `
        + `${fileWords(item.file, view)} to one without the credential, and give the key through a key variable.` })),
      ...places.map((item) => el("p", { class: "error", text: `[${item.section}] in ${fileWords(item.file, view)} sets `
        + `an address, a key or a key variable, which ${last} in the project directory may not hold.` })),
      ...keys.map((item, index) => field({ id: `key-variable-${index}`, label: `Key variable for ${item.section}`,
        input: inputs[index], hint: `The key in ${fileWords(item.file, view)} is not copied. The runs read `
          + "it from the variable named here, so set the variable to the same key before the next run." })),
      keys.length ? missing : null, el("div", { class: "actions" }, yes, no), note));
    check();
    no.focus();
  });
  return ask;
}

// A list's note: the files its runs read in order, the one a change goes
// to, and why it cannot change when it is read-only.
function listNote(view) {
  const list = view.list;
  return el("p", { class: "muted" }, `Runs of ${list.title} in ${list.name} read these models from `,
    ...list.files.flatMap((file, index) => [index ? ", then " : "", fileMark(file, view)]),
    list.read_only ? `. ${list.read_only}` : `. A change goes to ${shortPath(list.file)}${list.files.length > 1
      ? ", and its sections take the place of the earlier files' sections of the same name" : ""}. ${list.in_project
      ? "It lies in the project directory, so it holds no address, key or key variable. Its models take them from "
        + "the earlier files or Testuary's defaults." : "A key is never stored there: name its variable."}`,
    list.exists || list.read_only ? "" : ` ${shortPath(list.file)} does not exist yet. The first change creates it.`);
}

async function modelsView(query) {
  const scope = query.get("scope") || "";
  const view = await call(scope ? `/api/models?project_dir=${encodeURIComponent(query.get("project_dir") || "")}`
    + `&scope=${encodeURIComponent(scope)}` : "/api/models");
  const list = view.list;
  const changes = !list || !list.read_only;
  const box = el("div", {});
  const add = el("button", { type: "button", class: "primary", text: "Add a model", hidden: !changes,
    onclick: () => { box.replaceChildren(modelForm(view, null)); box.querySelector("input").focus(); } });
  const choice = el("select", { id: "models-for", "aria-label": "Models for" },
    el("option", { value: "", text: "Your model catalogue" }),
    view.scopes.map((item, index) => el("option", { value: String(index), text: `The project ${item.name}: models for ${item.title}`,
      selected: Boolean(item.scope === scope && item.project_dir === query.get("project_dir")) })));
  choice.addEventListener("change", () => {
    const item = view.scopes[Number(choice.value)];
    location.hash = item ? modelsLink(item.project_dir, item.scope) : modelsLink();
  });
  const overridden = new Map((view.provider_overrides || []).map((item) => [item.section, item.file]));
  const entries = view.models.map((entry) => {
    const [test, said] = connectionTest(entry.alias, listRequest(list));
    // A model whose section an earlier file holds with an address or key
    // cannot be copied into a last file in the project directory.
    const fixed = Boolean(list && list.in_project && entry.file !== list.file && entry.key_places);
    const actions = el("div", { class: "actions" }, test, changes && !fixed ? el("button", { type: "button", text: "Edit", onclick: () => {
      box.replaceChildren(modelForm(view, entry));
      box.scrollIntoView({ block: "nearest" });
      box.querySelector("select").focus();
    } }) : null);
    const path = `/api/models/${encodeURIComponent(entry.alias)}/remove`;
    if (!list) actions.append(removeAction(actions, removeQuestion(entry), path));
    else if (changes && entry.file === list.file) {
      actions.append(removeAction(actions, entry.overrides ? `Remove ${entry.alias} from ${shortPath(list.file)}? `
        + `The model from ${fileWords(entry.overrides, view)} applies again${entry.provider_from === list.file
          ? `, with the provider section ${entry.provider} that ${shortPath(list.file)} holds` : ""}.` : `Remove ${entry.alias} from `
        + `${shortPath(list.file)}? Its section goes, and nothing else changes.`, path, listRequest(list)));
    } else if (changes && list.end_import) actions.append(endImportAction(actions, entry, view));
    const [style, word] = modelBadge(entry);
    return el("div", { class: "panel entry" },
      el("h3", {}, entry.alias, modelMarks(entry.alias, view.default_model, view.project_model)
        .map((text) => el("span", { class: "default-mark", text })),
        el("span", { class: `state ${style}`, text: word }), list ? fileMark(entry.file, view) : null),
      entry.overrides ? el("p", { class: "small muted", text: `Its section in ${shortPath(list.file)} takes the place `
        + `of the one in ${fileWords(entry.overrides, view)}. Changes in ${fileWords(entry.overrides, view)} do not reach it.` })
        : null,
      list && overridden.has(entry.provider) ? el("p", { class: "small muted", text: `Its provider section in `
        + `${shortPath(list.file)} takes the place of the one in ${fileWords(overridden.get(entry.provider), view)}.` }) : null,
      fixed ? el("p", { class: "small muted", text: `Its section in ${fileWords(entry.file, view)} sets an address, a key `
        + `or a key variable, which ${shortPath(list.file)} in the project directory may not hold. Change it in `
        + `${fileWords(entry.file, view)}.` }) : null,
      el("dl", { class: "fields" },
        el("dt", { text: "provider" }), el("dd", { text: entry.provider }),
        el("dt", { text: "model" }), el("dd", { text: entry.model }),
        el("dt", { text: "address" }), el("dd", { text: entry.address || "the provider's configured address" }),
        el("dt", { text: "key" }), el("dd", { text: keyText(entry, list) })),
      actions, said);
  });
  return [
    el("div", { class: "hero" }, el("h1", { text: "Models" }), el("div", { class: "actions" }, add)),
    view.scopes.length ? el("div", { class: "field" }, el("label", { for: "models-for", text: "Models for" }), choice) : null,
    // models.ini's place shown short, the full path on hover and to copy.
    list ? listNote(view) : view.file ? el("p", { class: "muted" }, "The model catalogue, ", pathText(view.file, shortPath(view.file)),
      ", as Testuary's setup reads and writes it. A stored key is never shown: Edit replaces it.") : null,
    view.error ? el("p", { class: "panel error", text: view.error }) : null,
    box,
    // The last file's provider sections that take the place of an earlier
    // file's, for every model of the provider.
    ...(view.provider_overrides || []).map((item) => {
      const actions = el("div", { class: "actions" });
      if (changes) actions.append(removeAction(actions, `Remove the provider section ${item.section} from `
        + `${shortPath(list.file)}? The one in ${fileWords(item.file, view)} applies again to every model of ${item.section}.`,
      `/api/models/${encodeURIComponent(item.section)}/remove`, listRequest(list)));
      return el("div", { class: "panel entry" }, el("h3", {}, item.section, fileMark(list.file, view)),
        el("p", { class: "small muted", text: `Provider section ${item.section} in ${shortPath(list.file)} takes the `
          + `place of the one in ${fileWords(item.file, view)}, for every model of ${item.section}.` }), actions);
    }),
    entries.length ? entries : el("p", { class: "panel muted", text: "No model yet. Use Add a model above." }),
    // Provider sections the setup keeps after their last model went.
    !list && view.unused && view.unused.length ? el("p", { class: "panel muted small" },
      `Provider sections in models.ini that no model uses: ${view.unused.join(", ")}. `
      + "They stay with their addresses and any stored keys, for a later model or a run with [api] provider. "
      + "Remove one in models.ini when you no longer need it.") : null,
  ];
}

function projectForm(view, entry, aliases) {
  const form = el("form", { class: "panel", id: "project-form", novalidate: true });
  const fields = entry.fields || {};
  const described = Object.fromEntries(view.settings.map((setting) => [setting.name, setting]));
  const language = el("select", {}, el("option", { value: "", text: "any installed language" }),
    view.languages.map((item) => el("option", { value: item, text: item,
      selected: item === fields.language || undefined })));
  const files = el("input", { type: "text", value: fields.source_files || "", autocomplete: "off" });
  const model = el("select", {}, el("option", { value: "", text: "the default model" }),
    aliases.map((alias) => el("option", { value: alias, text: alias, selected: alias === fields.model || undefined })));
  // The language's settings, each once, as the server lists them.
  const extra = el("div", {});
  const drawExtra = () => extra.replaceChildren(...(view.language_settings[language.value] || [])
    .map((setting) => field({ id: setting.name, label: setting.label,
    input: el("input", { type: "text", value: fields[setting.name] || "", placeholder: setting.default || "",
      autocomplete: "off", "data-setting": setting.name }),
    hint: setting.purpose })));
  language.addEventListener("change", drawExtra);
  drawExtra();
  form.append(el("h2", { text: `Settings of ${entry.name}` }),
    el("p", { class: "small muted" }, "Written to ",
      entry.settings_file ? pathText(entry.settings_file, shortPath(entry.settings_file)) : "the project's settings file",
      ". An empty value takes the setting out of the file."),
    ...(tabShown("settings") ? [el("p", {}, el("a", { href: settingsLink(entry.project_dir), text: `All settings of ${entry.name}` }),
      `: coverage, mutation and every other setting, on the ${tabTitle("settings")} page.`)] : []),
    field({ id: "language", label: "Language", input: language,
      hint: described.language ? described.language.purpose : "" }),
    field({ id: "source_files", label: "Source files", input: files,
      hint: described.source_files ? described.source_files.purpose : "" }),
    field({ id: "model", label: "Model", input: model }),
    extra);
  return saveForm(form, "Save", () => {
    const values = { language: language.value, source_files: files.value, model: model.value };
    for (const input of extra.querySelectorAll("[data-setting]")) values[input.dataset.setting] = input.value;
    return ["/api/project", { project_dir: entry.project_dir, values }];
  });
}

// Adding a project: choose its directory, or a GitHub address and the directory
// to clone it into, then answer Testuary's own project setup there, one
// question at a time, as `testuary setup project` asks them. Testuary checks
// each answer and writes the settings file at the end.
function addProjectForm(state) {
  const box = el("div", {});
  const form = el("form", { class: "panel", id: "add-project-form", novalidate: true });
  const address = el("input", { type: "text", inputmode: "url", autocomplete: "off",
    placeholder: "https://github.com/owner/project" });
  const dir = el("input", { type: "text", autocomplete: "off" });
  const button = el("button", { type: "submit", class: "primary", text: "Set it up" });
  const note = el("span", { class: "hint", "aria-live": "polite" });
  form.append(el("h2", { text: "Add a project" }),
    field({ id: "url", label: "GitHub address (optional)", input: address,
      hint: "A project on GitHub by its HTTPS address: git clones it into a new directory, named after the project, in the directory below, and the setup follows there. Needs git." }),
    field({ id: "project_dir", label: "Project directory", input: dir, browse: state.base_dir,
      hint: "The directory that holds the project's source, or, with a GitHub address, the directory to clone it into. Testuary's setup asks about the project next and writes its settings file into the project's Testuary directory there." }),
    el("div", { class: "actions" }, button, el("button", { type: "button", text: "Cancel", onclick: () => render() }),
      note, el("span", { class: "error", "aria-live": "polite" })));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearErrors(form);
    let chosen = dir.value.trim();
    const url = address.value.trim();
    if (url) {
      button.disabled = true;
      note.textContent = `Cloning ${url} with git.`;
      try {
        chosen = (await call("/api/projects/clone", { url, parent_dir: chosen })).project_dir;
      } catch (error) {
        // The one field names the project or, with an address, its parent.
        showError(form, error.field === "parent_dir" ? { ...error, field: "project_dir" } : error);
        return;
      } finally {
        button.disabled = false;
        note.textContent = "";
      }
      dir.value = chosen;
      address.value = "";
    }
    await askingSetup([button], note, () => setupStep(box, form, chosen, {}));
  });
  box.append(form);
  return box;
}

// What an open question offers: Testuary's default answer, then the other
// paths the setup found.
function answerChoices(question) {
  return question.default && !question.choices.includes(question.default)
    ? [question.default, ...question.choices] : [...question.choices];
}

// A text in parts, each address in it a link.
function textParts(text) {
  const parts = [];
  let rest = text;
  for (let found = /https?:\/\/[^\s)]+/.exec(rest); found; found = /https?:\/\/[^\s)]+/.exec(rest)) {
    // A sentence's full stop or comma after the address is no part of it.
    const address = found[0].replace(/[.,;:]+$/, "");
    if (found.index) parts.push(["text", rest.slice(0, found.index)]);
    parts.push(["link", address]);
    rest = rest.slice(found.index + address.length);
  }
  if (rest) parts.push(["text", rest]);
  return parts;
}

// Testuary's own lines, kept as printed, each address in them a link.
function consoleText(text) {
  return el("div", { class: "console-text" }, textParts(text).map(([kind, value]) => kind === "link"
    ? el("a", { href: value, rel: "noopener noreferrer", target: "_blank", text: value }) : value));
}

// Select all and Deselect all for a list's file boxes, beside how many are
// ticked. A box ticked by hand updates the count too.
function tickAll(boxes) {
  const count = el("span", { class: "muted small" });
  const show = () => { count.textContent = `${boxes.filter((box) => box.checked).length} of ${boxes.length} ticked`; };
  const tick = (value) => { boxes.forEach((box) => { box.checked = value; }); show(); };
  boxes.forEach((box) => box.addEventListener("change", show));
  show();
  return el("div", { class: "actions tick-all" },
    el("button", { type: "button", text: "Select all", onclick: () => tick(true) }),
    el("button", { type: "button", text: "Deselect all", onclick: () => tick(false) }), count);
}

// An open question's answer: the chosen path, or the other path typed. A
// question whose setup found no default has none chosen until one is.
function openAnswer(chosen, other) {
  if (!chosen) throw { field: "answer", message: "Choose one of the paths above, or another path." };
  return chosen.value || other.value;
}

function setupInput(view) {
  const question = view.question;
  if (question.key === "targets" && view.files) {
    // Testuary numbers the files it lists, and the answer names the chosen ones.
    const boxes = view.files.map((path) => el("input", { type: "checkbox", value: path, checked: true }));
    const group = el("fieldset", { "data-field": "answer" }, el("legend", { text: question.prompt.split("?")[0] + "?" }),
      tickAll(boxes),
      view.files.map((path, index) => el("label", { class: "check" }, boxes[index], el("code", { text: path }))),
      el("span", { class: "error", "aria-live": "polite" }));
    return [group, () => {
      const picked = boxes.map((box, index) => box.checked ? String(index + 1) : null).filter(Boolean);
      if (!picked.length) throw { ...NO_FILE, field: "answer" };
      return picked.length === boxes.length ? "all" : picked.join(",");
    }];
  }
  if (!question.choices.length && ["yes", "no"].includes(question.default)) {
    const group = choices({ name: "answer", legend: question.prompt, options: [
      ["yes", "Yes"], ["no", "No"]] });
    group.querySelector(`input[value="${question.default}"]`).checked = true;
    return [group, () => group.querySelector("input:checked").value];
  }
  if (question.choices.length && question.open) {
    // What the setup found, each path whole, or another answer its check
    // accepts: a list cut at its width showed two long paths alike.
    const found = answerChoices(question);
    const other = el("input", { type: "text", autocomplete: "off", "aria-label": "Another path",
      placeholder: "another path", disabled: true });
    const items = found.map((choice, index) => el("div", { class: "check" },
      el("input", { type: "radio", name: "answer-choice", id: `answer-choice-${index}`, value: choice,
        checked: choice === question.default || undefined }),
      el("label", { for: `answer-choice-${index}` }, el("code", { text: choice }))));
    const another = el("div", { class: "check" },
      el("input", { type: "radio", name: "answer-choice", id: "answer-choice-other", value: "" }),
      el("label", { for: "answer-choice-other", text: "Another path:" }), other);
    const group = el("fieldset", { "data-field": "answer", class: "open-answer" },
      el("legend", { text: question.prompt }), items, another,
      question.reason ? el("span", { class: "hint", text: question.reason }) : null,
      el("span", { class: "error", "aria-live": "polite" }));
    group.addEventListener("change", () => {
      other.disabled = group.querySelector("input[type=radio]:checked").value !== "";
      if (!other.disabled) other.focus();
    });
    return [group, () => openAnswer(group.querySelector("input[type=radio]:checked"), other)];
  }
  const input = question.choices.length
    ? el("select", {}, question.choices.map((choice) => el("option", { value: choice, text: choice,
      selected: choice === question.default || undefined })))
    : el("input", { type: "text", value: question.default, autocomplete: "off" });
  return [field({ id: "answer", label: question.prompt.replace(/ \(none set up yet\)$/, ""), input,
    hint: question.reason || "" }), () => input.value];
}

// One thing the setup says is still missing, without the console command
// at its end (": run testuary setup model"): the page offers its own way.
// A path in it stays whole, its drive letter included.
function missingItem(text) {
  return text.replace(/: run testuary [a-z -]+$/, "");
}

// A model added in the setup's own place, then the same step again.
async function addModelHere(box, back) {
  const view = await call("/api/models");
  box.replaceChildren(el("h3", { text: "Add a model" }),
    modelForm(view, null, (body) => back(body.alias), () => back(null)));
  box.querySelector("input").focus();
}

// Shown on the projects page once after a setup: what it wrote.
let projectNotice = null;

// While Testuary's setup answers, the form's buttons wait and its note
// says so, so that no second step starts.
async function askingSetup(buttons, note, step) {
  for (const button of buttons) button.disabled = true;
  note.textContent = "Asking Testuary's setup.";
  try { await step(); }
  finally {
    for (const button of buttons) button.disabled = false;
    note.textContent = "";
  }
}

// ``asked`` holds the keys answered so far, in order: Back takes the last
// answer away, and the setup asks that question again.
async function setupStep(box, start, dir, answers, asked = []) {
  let view;
  try { view = await call("/api/project/setup", { project_dir: dir, answers }); }
  catch (error) {
    if (start.isConnected) showError(start, { field: "project_dir", message: error.message });
    else box.replaceChildren(el("p", { class: "panel error", text: error.message }));
    return;
  }
  const said = view.said.length ? consoleText(view.said.join("\n")) : null;
  // The settings the setup is about to write: Testuary's own lines from the
  // first section heading on.
  const lines = view.said.join("\n").split("\n");
  const from = lines.findIndex((line) => /^\[[a-z_]+\]$/.test(line.trim()));
  const preview = from >= 0 ? consoleText(lines.slice(from).join("\n")) : null;
  if (view.refused) {
    box.replaceChildren(el("div", { class: "panel" }, el("h2", { text: "Testuary's setup stopped" }), said,
      el("p", { class: "error", text: view.refused }),
      el("div", { class: "actions" }, el("button", { type: "button", text: "Back", onclick: () => render() }))));
    return;
  }
  if (!view.question) {
    // The list shows the new project at once, and the summary stays above it.
    const missing = view.missing.map(missingItem);
    projectNotice = { project_dir: view.project_dir, written: view.written, missing, preview: preview && preview.textContent };
    await render();
    return;
  }
  const confirm = view.question.key === "project.confirm";
  const [input, answer] = confirm ? [null, () => "yes"] : setupInput(view);
  const name = view.project_dir.split(/[\\/]/).filter(Boolean).pop();
  const extra = [];
  const after = [];
  if (confirm) {
    // The values as the project card shows them, the file's own text folded.
    const target = (/ in (.+)\?\s*$/.exec(view.question.prompt) || [])[1];
    const values = Object.fromEntries((view.pending || []).map((item) => [item.name, item.value]));
    const labels = Object.fromEntries((view.pending || []).map((item) => [item.name, item.label]));
    extra.push(el("p", {}, "The settings to write",
      target ? [" in ", pathText(target, shortPath(target))] : null, ":"),
    fieldList(values, labels),
    preview ? el("details", {}, el("summary", { text: "Show the settings file" }), preview) : null);
  }
  if (view.question.key === "model") {
    // Under the question it offers another answer to.
    after.push(el("p", { class: "small" }, view.question.choices.length ? "Or " : "No model is set up yet. ",
      el("button", { type: "button", class: "link-button", text: "add a model",
        onclick: () => addModelHere(box, (alias) => setupStep(box, start, dir,
          alias ? { ...answers, model: alias } : answers, asked)) }), "."));
  }
  const back = () => {
    if (!asked.length) { render(); return; }
    const earlier = { ...answers };
    delete earlier[asked[asked.length - 1]];
    setupStep(box, start, dir, earlier, asked.slice(0, -1));
  };
  const form = el("form", { class: "panel", id: "setup-form", novalidate: true },
    el("h2", {}, "Set up ", el("span", { text: name, title: view.project_dir })), extra, input, after);
  const next = el("button", { type: "submit", class: "primary", text: confirm ? "Write the settings" : "Next" });
  const backButton = el("button", { type: "button", text: "Back", onclick: back });
  const busy = el("span", { class: "small muted", role: "status" });
  form.append(el("div", { class: "actions" }, next, backButton,
    el("button", { type: "button", text: "Cancel", onclick: () => render() }),
    busy, el("span", { class: "error", "aria-live": "polite" })));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    let value;
    try { value = answer(); } catch (error) { showError(form, error); return; }
    await askingSetup([next, backButton], busy, () => setupStep(box, start, dir,
      { ...answers, [view.question.key]: value }, [...asked, view.question.key]));
  });
  box.replaceChildren(form);
  if (view.problem) showError(form, { field: "answer", message: view.problem });
  else (form.querySelector("select, input, button.primary") || form).focus();
}

async function projectsView() {
  const [view, models, state] = await Promise.all([call("/api/projects"), call("/api/models"), call("/api/state")]);
  const aliases = models.models.map((entry) => entry.alias);
  const box = el("div", {});
  const add = el("button", { type: "button", class: "primary", text: "Add a project",
    onclick: () => { box.replaceChildren(addProjectForm(state)); box.querySelector("input").focus(); } });
  const notice = projectNotice;
  projectNotice = null;
  if (notice) {
    box.append(el("div", { class: "panel notice" }, el("h2", { text: "Project set up" }),
      notice.written.length ? el("p", {}, "Written: ", notice.written.map((path) =>
        pathText(path, path.split(/[\\/]/).slice(-3).join("/")))) : null,
      notice.preview ? consoleText(notice.preview) : null,
      tabShown("settings") && el("p", {}, el("a", { href: settingsLink(notice.project_dir), text: "All settings of the project" }),
        `: coverage, mutation and every other setting, on the ${tabTitle("settings")} page.`),
      notice.missing.map((item) => /model/i.test(item)
        ? el("p", { class: "small" }, "No model yet: ", el("button", { type: "button", class: "link-button",
          text: "add a model", onclick: () => addModelHere(box, async (alias) => {
            if (alias) await call("/api/project", { project_dir: notice.project_dir, values: { model: alias } });
            render();
          }) }), " and the project uses it.")
        : el("p", { class: "small", text: `Still missing: ${item}.` }))));
  }
  const entries = view.projects.map((entry) => {
    const actions = projectActions(entry);
    if (!entry.error) {
      actions.append(el("button", { type: "button", text: "Edit settings", onclick: () => {
        box.replaceChildren(projectForm(view, entry, aliases));
        box.scrollIntoView({ block: "nearest" });
        box.querySelector("select").focus();
      } }));
    }
    // A project of this directory or one below it stays listed while the
    // interface runs here, so only a recent one can be taken off the list.
    if (!entry.working && !entry.below) {
      actions.append(removeAction(actions, `Take ${entry.name} off this list? Its directory, settings and results stay.`,
        "/api/projects/forget", { project_dir: entry.project_dir }, "Forget"));
    }
    return el("div", { class: "panel entry" }, projectHeading(entry),
      entry.error ? el("p", { class: "error", text: entry.error }) : fieldList(entry.fields, entry.labels), actions);
  });
  return [
    el("div", { class: "hero" }, el("h1", { text: "Projects" }), el("div", { class: "actions" }, add)),
    el("p", { class: "muted", text: "Each project is a directory with its own settings file, project.ini in its Testuary directory. The list holds the project in this directory, or, when this directory is none, the projects in the directories directly below it, and the projects used recently. Forgetting one leaves its directory, settings and results." }),
    box,
    entries.length ? entries : el("p", { class: "panel muted", text: "No project yet. Use Add a project above." }),
  ];
}

// A choice's options as [value, text]. An unset value reads "not set", not
// as the first option, which a Save would then write.
function choiceOptions(setting) {
  const options = (setting.kind === "boolean" ? ["true", "false"] : setting.choices).map((option) => [option, option]);
  return setting.value === null || setting.value === undefined || setting.value === ""
    ? [["", "not set"], ...options] : options;
}

function settingInput(setting) {
  if (setting.kind === "boolean" || (setting.kind === "choice" && setting.choices.length)) {
    return el("select", {}, choiceOptions(setting).map(([value, text]) => el("option", { value, text,
      selected: value === (setting.value || "") || undefined })));
  }
  return el("input", { type: "text", value: setting.value || "", autocomplete: "off" });
}

// A purpose shown whole when it is short, such as a limit's sentence that
// zero disables it. Otherwise its first sentence, with the rest on request.
function purposeParts(purpose) {
  const text = purpose || "";
  if (text.length <= 160) return [text, ""];
  const cut = text.search(/[.!?](\s|$)/);
  const first = cut >= 0 ? text.slice(0, cut + 1) : text;
  return [first, text.slice(first.length).trim()];
}

// Testuary's refusal of a value, without the section and key its row
// already names.
function refusalText(message, section, setting) {
  const prefix = `[${section}] ${setting.name}: `;
  const text = message.startsWith(prefix) ? message.slice(prefix.length) : message;
  return `Not saved: ${text}${/[.!?]$/.test(text) ? "" : "."}`;
}

// The settings a run usually changes, by section: a section shows these
// and every setting a file or the environment sets, and keeps its others
// one fold away. The sections a run usually changes come first.
const COMMON_SETTINGS = {
  api: ["model"], project: ["source_files", "language", "python_executable", "java_home"],
  starter: ["suite_count"],
  coverage: ["goal", "minimum", "max_requests_per_phase", "max_wall_clock_seconds_per_phase"],
  mutation: ["goal", "minimum", "max_requests_per_phase", "max_wall_clock_seconds_per_phase"],
  demo: ["playback_seconds"],
};
const FIRST_SECTIONS = ["api", "project", "starter", "coverage", "mutation"];

function shownFirst(section, setting) {
  return (COMMON_SETTINGS[section] || []).includes(setting.name)
    || Boolean(setting.source && setting.source !== "default");
}

// Whether a setting matches every word of a search: in its label, its
// name, its purpose or its section's title.
function settingMatches(sectionTitle, setting, text) {
  const haystack = [sectionTitle, setting.label, setting.name, setting.purpose].join(" ").toLowerCase();
  return text.toLowerCase().split(/\s+/).filter(Boolean).every((word) => haystack.includes(word));
}

// A row's Save and Reset: ``changed`` draws the row anew from the settings
// as they are now, with what happened, so the page keeps its open
// sections and its place.
function settingRow(section, setting, target, other, changed, said = "") {
  const id = `setting-${section}-${setting.name}`;
  const input = settingInput(setting);
  input.id = id;
  // A read-only page shows the values and offers no change.
  input.disabled = Boolean(target.readOnly);
  const note = el("span", { class: "error", "aria-live": "polite" });
  const done = el("span", { class: "small saved", role: "status", text: said });
  // A value changed here says so until it is saved or changed back.
  const loaded = input.value;
  input.addEventListener("input", () => {
    const pending = input.value !== loaded;
    done.className = `small ${pending ? "unsaved" : "saved"}`;
    done.textContent = pending ? "Not saved yet." : said;
  });
  const send = async (body, button, words) => {
    button.disabled = true;
    note.textContent = "";
    done.textContent = "";
    // An extension's setting changes through its own validated update.
    const path = setting.group ? "/api/ext/settings" : "/api/settings";
    try { await call(path, { ...target.request, section, group: setting.group, ...body }); }
    catch (error) { note.textContent = refusalText(error.message, section, setting); button.disabled = false; return; }
    await changed(section, setting.name, words);
  };
  const save = el("button", { type: "button", text: "Save" });
  save.addEventListener("click", () => send({ values: { [setting.name]: input.value } }, save,
    `Saved in ${levelWords(target.file, target.files).toLowerCase()}.`));
  // Reset takes the setting out of the file the changes go to, also when
  // another level's value is in force.
  const reset = canReset(setting, target) ? el("button", { type: "button", text: "Reset" }) : null;
  // [main] extensions adds each level's items, so no level overrides another.
  const combined = setting.combined !== undefined;
  const overridden = combined ? null : overrideNote(setting, target) || contextNote(setting, other, target);
  if (reset) {
    reset.addEventListener("click", () => send({ remove: [setting.name] }, reset,
      `Reset: taken out of ${levelWords(target.file, target.files).toLowerCase()}.`));
  }
  const [first, rest] = purposeParts(setting.purpose);
  return el("div", { class: `setting${setting.active === false ? " inactive" : ""}` },
    el("div", { class: "setting-name" }, el("label", { for: id, text: setting.label || setting.name }),
      el("code", { class: "small muted", text: setting.name }),
      first ? el("div", { class: "small muted", text: first }) : null,
      rest ? el("details", { class: "small" }, el("summary", { text: "More" }), el("p", { class: "muted", text: rest })) : null,
      setting.active === false ? el("div", { class: "small", text: `Not used now: ${setting.inactive}.` }) : null),
    el("div", { class: "setting-value" }, input,
      overridden ? el("div", { class: "small", text: overridden }) : null, note),
    el("div", { class: "setting-source" }, combined ? combinedSource(setting, target.files)
      : sourceBadge(setting.source, target.files)),
    el("div", { class: "setting-actions" }, target.readOnly ? null : [save, reset, done]));
}

// Where a value comes from: the default, the global settings, this
// project's settings, the environment or a command, as a badge's word and
// as a phrase.
function sourceWords(source, files) {
  const kind = !source || source === "default" ? "default"
    : source === files.global ? "global" : source === files.project ? "project"
      : source === files.named ? "named" : (files.list || []).includes(source) ? "listed"
        : source === "environment" ? "environment" : source === "overrides" ? "command" : "file";
  // An earlier file of a named list reads as its short path, its full path
  // on hover.
  const listed = kind === "listed" ? shortPath(source) : "";
  return { kind,
    badge: { default: "Default", global: "Global settings", project: "This project", named: "This file",
      listed, environment: "Environment", command: "Command line", file: "Another file" }[kind],
    phrase: { default: "Testuary's default", global: "the global settings", project: "this project",
      named: "this file", listed, environment: "the environment", command: "the command line",
      file: "another file" }[kind] };
}

function sourceBadge(source, files) {
  const words = sourceWords(source, files);
  return el("span", { class: `source source-${words.kind}`, title: source || "Testuary's default",
    text: words.badge });
}

// The value in force of a setting whose levels add their items: the
// combined list, and each level that adds items.
function combinedSource(setting, files) {
  const levels = (setting.sources || []).filter((source) => source && source !== "default");
  return el("div", {}, el("div", { class: "small", text: `In force: ${setting.combined || "none"}` }),
    ...levels.flatMap((source) => [sourceBadge(source, files), " "]));
}

// A row offers Reset whenever the file the page changes sets the key.
function canReset(setting, target) {
  return Boolean(target.file) && (setting.sources || []).includes(target.file);
}

function levelWords(file, files) {
  return file === files.project ? "This project's settings" : file === files.named ? "This file"
    : "The global settings";
}

// The global page's note on the project it came from: that project's
// value in force, when it is not the global page's.
function contextNote(setting, other, target) {
  if (!other || other.source === setting.source) return null;
  const words = sourceWords(other.source, target.files);
  return `In the project ${target.context}, the value in force is ${other.value}, `
    + `from ${words.kind === "project" ? "its own settings" : words.phrase}.`;
}

// What a settings page reads: the project page the project's settings, a
// named file's page the settings of the runs that read that file, the
// global page the settings without a project, and the project it came
// from for its notes.
function settingsRequests(scope, project) {
  const projectRequest = project ? `/api/settings?project_dir=${encodeURIComponent(project.project_dir)}` : null;
  if (scope === "global") return ["/api/settings", projectRequest].filter(Boolean);
  return [scope === "project" ? projectRequest : `${projectRequest}&scope=${encodeURIComponent(scope)}`];
}

// Where "Settings for" leads: a project's page or the page of a settings
// file its settings groups name, or the global page, which keeps the
// project it was chosen from as context.
function settingsLink(choice, project, scope = "") {
  const named = scope ? `&scope=${encodeURIComponent(scope)}` : "";
  if (choice) return `#/settings?project_dir=${encodeURIComponent(choice)}${named}`;
  return project ? `#/settings?project_dir=${encodeURIComponent(project.project_dir)}&scope=global` : "#/settings";
}

// When the file the page changes sets the key but another level's value is
// in force, the row says which one.
function overrideNote(setting, target) {
  if (!canReset(setting, target) || setting.source === target.file) return null;
  return `${levelWords(target.file, target.files)} set this too. `
    + `The value in force comes from ${sourceWords(setting.source, target.files).phrase}.`;
}

// The installed versions, at the end of the settings page.
function aboutInstallation(state) {
  return el("details", { class: "panel" }, el("summary", { text: "About this installation" }),
    el("p", { text: `Installed: ${state.testuary}.` }));
}

// The project the Settings page is for: the one its address names, else the
// working directory's project when there is one, else none (the global settings).
function settingsDir(query, projects) {
  const working = projects.find((entry) => entry.working);
  return query.get("project_dir") || (working ? working.project_dir : "");
}

async function settingsView(query) {
  const [listed, state] = await Promise.all([call("/api/projects").catch(() => ({ projects: [] })),
    call("/api/state")]);
  const dir = settingsDir(query, listed.projects);
  const project = listed.projects.find((entry) => entry.project_dir === dir) || null;
  const asked = query.get("scope");
  const scope = !project || asked === "global" ? "global" : asked || "project";
  // Each page edits and resets its own level only. The global page shows
  // the global file's own values, and the project's shows only as text.
  const load = () => Promise.all(settingsRequests(scope, project).map((path, index) =>
    index ? call(path).catch(() => null) : call(path)));
  const [view, context] = await load();
  const own = scope === "global" ? null : project;
  const files = context && !context.error ? context.files : view.files;
  // The settings files the project's settings groups name, and the one
  // this page changes.
  const scopes = ((scope === "global" ? context : view) || {}).scopes || [];
  const named = scopes.find((entry) => entry.scope === scope);
  const othersOf = (given) => new Map((given && !given.error ? given.sections : []).flatMap((section) =>
    section.settings.map((setting) => [`${section.name}.${setting.name}`, setting])));
  const others = othersOf(context);
  const target = { file: view.files[named ? "named" : scope], files, context: project && project.name,
    request: own ? { scope, project_dir: project.project_dir } : { scope }, readOnly: Boolean(named && named.read_only) };
  // The files a named list's runs read, the one this page changes last.
  const reads = named ? [...named.files.slice(0, -1).map(shortPath),
    named.global ? "your global settings" : "this file"].join(", then ") : "";
  const choice = el("select", { id: "settings-for", "aria-label": "Settings for" },
    el("option", { value: "", text: "All projects: the global settings" }),
    listed.projects.filter((entry) => !entry.error).flatMap((entry) => [el("option", {
      value: entry.project_dir, text: `The project ${entry.name}`,
      selected: Boolean(own && !named && entry.project_dir === own.project_dir) }),
    ...(project && entry.project_dir === project.project_dir ? scopes : []).map((item) => el("option", {
      value: entry.project_dir, "data-scope": item.scope, selected: item.scope === scope,
      text: `The project ${entry.name}: Testuary settings for ${item.title}` }))]));
  choice.addEventListener("change", () => {
    location.hash = settingsLink(choice.value, project, choice.selectedOptions[0].dataset.scope);
  });
  const parts = [el("h1", { text: "Settings" }),
    el("div", { class: "field" }, el("label", { for: "settings-for", text: "Settings for" }), choice),
    el("p", { class: "muted" }, "Each setting with its value and where it comes from",
      named ? ` for runs of ${named.title} in ${own.name}` : own ? ` for runs in ${own.name}` : "",
      // A refused choice says only what it shows: its error follows.
      ...(view.error ? ["."] : target.readOnly ? [`. ${named.read_only}`, named.global
        ? [" ", el("a", { href: settingsLink("", project), text: "Open the global settings." })] : null]
        : [". A change goes to ",
          target.file ? pathText(target.file, named || own ? inDir(own.project_dir, target.file)
            : "your global config.ini")
            : "the settings file",
          " through Testuary's validated update. Reset takes a setting out of it again."]),
      view.error ? "" : named
        ? ` These runs read ${reads}`
          + ` in place of ${named.global ? "" : "the global settings and "}this project's settings.${
            named.exists || target.readOnly ? "" : " It does not exist yet. The first change creates it."}`
        : own ? " Values set there apply to this project only." : " Values set there apply to every project.",
      target.context && !own ? ` Where the project ${target.context} uses another value, its row says so.` : "")];
  if (view.error) return [...parts, el("p", { class: "panel error", text: view.error }), aboutInstallation(state)];
  // A named list's files in the order its runs read them.
  const sources = ["default", view.files.global, view.files.project, ...(view.files.list || [view.files.named])];
  parts.push(el("p", { class: "small" }, "Values come from: ",
    ...sources.filter(Boolean).flatMap((source) => [sourceBadge(source, files), " "])));
  if ((view.unchecked || []).length) {
    parts.push(el("div", { class: "panel notice" }, el("p", {
      text: "No registered language has these settings. An extension the settings name may add one, "
        + "so the tool check (testuary check --verify) checks them after it loads the extensions." }),
    el("ul", {}, view.unchecked.map((text) => el("li", {}, el("code", { text }))))));
  }
  // Each row's section title and setting, for the search.
  const rows = new Map();
  const setCount = (section) => {
    const set = section.settings.filter((setting) => setting.source && setting.source !== "default").length;
    return set ? ` ${set} set here or in a file` : "";
  };
  // A saved or reset row is drawn anew from the settings as they are now. A
  // failure to read them draws the page anew.
  const changed = async (name, key, said) => {
    let fresh = null;
    try { fresh = await load(); } catch (error) { fresh = null; }
    const section = fresh && !fresh[0].error && fresh[0].sections.find((item) => item.name === name);
    const setting = section && section.settings.find((item) => item.name === key);
    const input = document.getElementById(`setting-${name}-${key}`);
    const old = input && input.closest(".setting");
    // The first change to a named file creates it, which the whole page says.
    if (!setting || !old || (named && !named.exists)) { await render(); return; }
    const row = settingRow(name, setting, target, othersOf(fresh[1]).get(`${name}.${key}`), changed, said);
    old.replaceWith(row);
    rows.set(row, [section.title || section.name, setting]);
    const count = row.closest(".settings-group").querySelector(".set-count");
    if (count) count.textContent = setCount(section);
    document.getElementById(`setting-${name}-${key}`).focus({ preventScroll: true });
  };
  const sectionBox = (section) => {
    const title = section.title || section.name;
    const drawn = section.settings.map((setting) => {
      const row = settingRow(section.name, setting, target, others.get(`${section.name}.${setting.name}`), changed);
      rows.set(row, [title, setting]);
      return row;
    });
    const first = drawn.filter((row, index) => shownFirst(section.name, section.settings[index]));
    const more = drawn.filter((row, index) => !shownFirst(section.name, section.settings[index]));
    return el("details", { class: "panel settings-group", "data-section": section.name,
      open: FIRST_SECTIONS.includes(section.name) || undefined },
    el("summary", {}, el("strong", { text: title }), el("span", { class: "small muted set-count", text: setCount(section) }),
      section.purpose ? el("div", { class: "small muted", text: section.purpose }) : null),
    el("div", { class: "settings-head small muted", "aria-hidden": "true" },
      ["Setting", "Value", "From", ""].map((heading) => el("span", { text: heading }))),
    first, more.length ? el("details", { class: "more-settings" },
      el("summary", { text: `${more.length} more ${more.length === 1 ? "setting" : "settings"} of ${title}` }),
      more) : null);
  };
  // A section without settings of its own has nothing to show here.
  const withSettings = view.sections.filter((section) => section.settings.length);
  const firstBoxes = FIRST_SECTIONS.map((name) => withSettings.find((section) => section.name === name))
    .filter(Boolean).map(sectionBox);
  const laterBoxes = withSettings.filter((section) => !FIRST_SECTIONS.includes(section.name)).map(sectionBox);
  const laterHead = el("h2", { text: "More settings" });
  const find = el("input", { type: "search", id: "settings-find", autocomplete: "off",
    placeholder: "for example goal, requests or timeout" });
  const found = el("p", { class: "small muted", "aria-live": "polite" });
  find.addEventListener("input", () => {
    const text = find.value.trim();
    let count = 0;
    for (const box of [...firstBoxes, ...laterBoxes]) {
      let shown = 0;
      for (const row of box.querySelectorAll(".setting")) {
        const [title, setting] = rows.get(row);
        row.hidden = Boolean(text) && !settingMatches(title, setting, text);
        if (!row.hidden) shown += 1;
      }
      box.hidden = Boolean(text) && !shown;
      if (text && shown) {
        box.open = true;
        const more = box.querySelector(".more-settings");
        if (more) more.open = true;
      }
      count += text ? shown : 0;
    }
    laterHead.hidden = laterBoxes.every((box) => box.hidden);
    found.textContent = text ? `${count} ${count === 1 ? "setting matches" : "settings match"}.` : "";
  });
  parts.push(el("div", { class: "field" }, el("label", { for: "settings-find", text: "Find a setting" }), find, found),
    el("p", { class: "small muted", text: "The sections a run usually changes show their common settings and every setting a file or the environment sets. The others are one fold away, and the search finds every setting." }),
    firstBoxes, laterBoxes.length ? [laterHead, laterBoxes] : null);
  parts.push(aboutInstallation(state));
  return parts;
}

// Where a playback writes: it needs no setup, but it writes a directory.
function demoWrites(dir) {
  return ["Each playback writes its record into a new directory named after the language and the time, "
    + "such as testuary-demo-python-20261031-075100, in ", pathText(dir, shortPath(dir)), ".",
  tabShown("results") && ` The ${tabTitle("results")} page opens it.`];
}

async function demoView() {
  const [defaults, state] = await Promise.all([call("/api/demo/settings").catch(() => ({})), call("/api/state")]);
  const configured = defaults.playback_seconds;
  const seconds = el("input", { type: "text", inputmode: "decimal", autocomplete: "off",
    placeholder: configured ? String(configured) : "" });
  const length = field({
    id: "seconds", label: "Length at normal speed", option: "--demo-playback-seconds", input: seconds, unit: "seconds",
    hint: [configured ? `Empty: ${configured} seconds, as configured.` : "Empty: the configured length.",
      "The playback takes at least this many seconds."].join(" "),
  });
  const pace = choices({ name: "pace", legend: "Speed", option: "--demo-speed", options: [
    ["paced", "Normal speed: plays the recorded run over the length below"],
    ["instant", "Instant: plays without pauses"]] });
  const sync = () => {
    const instant = pace.querySelector("input:checked").value === "instant";
    seconds.disabled = instant;
    length.classList.toggle("disabled", instant);
  };
  pace.addEventListener("change", sync);
  const form = el("form", { class: "panel", id: "demo-form", novalidate: true },
    choices({ name: "language", legend: "Example", option: "--demo-language",
      hint: "The recorded run to play: a Python or a Java target.", options: [
      ["python", "Python: the wait module of the Tenacity retry library"],
      ["java", "Java: JSONCompare of the JSONassert library"]] }),
    pace, length);
  sync();
  jobForm(form, "Start the demonstration", () => {
    const data = new FormData(form);
    const request = { language: data.get("language"), pace: data.get("pace") };
    if (request.pace === "paced" && seconds.value.trim()) request.seconds = seconds.value.trim();
    return ["/api/demo", request];
  });
  return [
    el("h1", { text: "Demonstration" }),
    el("p", { text: "The demonstration plays back one recorded run of this Testuary release: the model's answers and the measurement results come from the recording. It sends no model request, runs no test or mutation tool and needs no setup. Every value it shows is a demo value, not a measurement." }),
    el("p", { class: "small" }, demoWrites(state.base_dir)),
    el("p", { class: "muted small", text: "Step mode, one step per Enter press, needs a terminal: testuary --demo --demo-step" }),
    form,
  ];
}

// The listed projects as a choice by name, the one the address names
// chosen, with the full directory on hover.
function projectChoice(listed, query) {
  const usable = (listed.projects || []).filter((entry) => !entry.error);
  const chosen = query.get("project_dir") || (usable[0] ? usable[0].project_dir : "");
  const select = el("select", {}, usable.map((entry) => el("option", { value: entry.project_dir,
    selected: entry.project_dir === chosen, text: entry.name, title: entry.project_dir })));
  const sync = () => { select.title = select.value; };
  select.addEventListener("change", sync);
  sync();
  return select;
}

// The files the project's settings name that its directory does not hold.
function missingNamed(named, files) {
  return named.filter((path) => !files.includes(path));
}

// The project's source files as a checklist, as Testuary's setup finds
// them: one list for the project's language, else a list per language
// with its own Select all. The files its settings name are ticked, and leaving
// exactly those ticked runs the project's own list. A named file the directory
// lacks is named.
function fileChecklist(project) {
  const box = el("fieldset", { class: "file-checklist", "data-field": "files" });
  let state = { boxes: [], own: [] };
  const load = async () => {
    box.replaceChildren(el("legend", { text: "Source files" }), el("p", { class: "muted small", text: "Finding the project's files." }));
    if (!project.value) { box.lastChild.textContent = "Choose a project."; return; }
    let data;
    try { data = await call(`/api/project/files?project_dir=${encodeURIComponent(project.value)}`); }
    catch (error) { box.lastChild.replaceWith(el("p", { class: "error", text: error.message })); return; }
    const own = new Set(data.project_files);
    const groups = Object.entries(data.files);
    const files = groups.flatMap(([, paths]) => paths);
    const boxes = files.map((path) => el("input", { type: "checkbox", value: path,
      checked: own.size ? own.has(path) : true }));
    state = { boxes, own: data.project_files };
    const missing = missingNamed(data.project_files, files);
    let start = 0;
    const lists = groups.map(([language, paths]) => {
      const group = boxes.slice(start, start += paths.length);
      return [groups.length > 1 ? el("h3", { text: language }) : "", tickAll(group),
        el("div", { class: "file-grid" }, paths.map((path, index) =>
          el("label", { class: "check" }, group[index], el("code", { text: path }))))];
    });
    box.replaceChildren(el("legend", { text: "Source files" }),
      missing.length ? el("p", { class: "small warn" }, "The project's settings name ",
        missing.map((path, index) => [index ? ", " : "", el("code", { text: path })]),
        ", which its directory does not hold. Tick the targets to run, or correct the settings.") : "",
      files.length ? el("div", {}, lists)
        : el("p", { class: "muted", text: `No ${data.language || "source"} files found in the project's directory.` }),
      el("span", { class: "hint", text: own.size
        ? "Ticked: the targets the project's settings name. Tick others to run those instead."
        : "The project's settings name no targets: the ticked ones run." }),
      el("span", { class: "error", "aria-live": "polite" }));
  };
  project.addEventListener("change", load);
  load();
  // The files to send: none for the project's own list.
  const chosen = () => {
    const ticked = state.boxes.filter((item) => item.checked).map((item) => item.value);
    const same = ticked.length === state.own.length && ticked.every((path) => state.own.includes(path));
    return same ? [] : ticked;
  };
  const empty = () => state.boxes.length > 0 && !state.boxes.some((item) => item.checked);
  return { box, chosen, empty };
}

const NO_FILE = { field: "files", message: "Tick at least one target." };

// What the check judges in a project with an earlier run, and what a run
// started on the Run page does there while that page is Testuary's own: a
// replacing page's runs need not use the project's run directory, and a hidden
// page starts none.
function earlierRunNote(saved) {
  const own = !addedTab("run") && tabShown("run");
  const started = `a run started on the ${tabTitle("run")} page`;
  return saved.unfinished.length
    ? `. The check judges continuing that earlier run${own ? `, as ${started} does unless you choose a new directory there` : ""}.`
    : `. The check judges the new directory ${saved.fresh}${own ? `, where ${started} starts a new run` : ""}.`;
}

// The note of a view that needs a project while none is listed.
function noProjectNote() {
  return el("p", { class: "panel muted" }, "No project yet.", tabShown("projects") && [" ", el("a", { href: "#/projects", text: "Add a project's directory" }), "."]);
}

async function checkView(query) {
  const listed = await call("/api/projects")
    .catch((error) => ({ error: error.message, projects: [] }));
  const project = projectChoice(listed, query);
  const files = fileChecklist(project);
  // The run directory the check judges when the project's own one holds an
  // earlier run: the one a run started from the Run page uses unless the
  // person chooses otherwise.
  const judged = el("div", { "aria-live": "polite" });
  let asked = 0;
  const judge = async () => {
    const number = ++asked;
    let saved = { occupied: false };
    if (project.value) {
      try { saved = await call(`/api/run/saved?${new URLSearchParams({ project_dir: project.value })}`); } catch (error) { /* the check says why */ }
    }
    if (number !== asked) return;
    judged.replaceChildren(saved.occupied ? el("div", { class: "notice-box" },
      el("p", {}, "This project already has results from its model in ", pathText(saved.full, saved.run_dir),
        earlierRunNote(saved))) : "");
  };
  project.addEventListener("change", judge);
  judge();
  const form = el("form", { class: "panel", id: "check-form", novalidate: true },
    field({ id: "project_dir", label: "Project", input: project }),
    judged,
    files.box,
    checkbox({ id: "verify", label: toolCheckLabel, option: "--verify",
      hint: "Starts each test, coverage and mutation tool the settings select once, to see that it works, and in a copy of the project tries its build where a run builds it and finds its tests without running one, unless the project has its own build wrapper, which may download its build tool. For a Java project it downloads what a run would when it is missing. Without it the quick check reads the settings and finds the tools." }));
  jobForm(form, "Run the check (sends nothing to a model)", () => {
    if (files.empty()) throw NO_FILE;
    return ["/api/check", { project_dir: project.value, files: files.chosen(),
      verify: form.querySelector("#verify").checked }];
  }, () => project.options.length > 0);
  return [
    el("h1", { text: "Check" }),
    el("p", { text: "The quick check reads your models, your settings and the project's settings, finds the tools, and tells you what is ready and what to fix before a run. The tool check also starts each tool once and, in a copy of the project, tries its build where a run builds it and finds its tests without running one, unless the project has its own build wrapper, which may download its build tool. For a Java project the tool check downloads what a run would when it is missing. Neither runs a test or sends anything to a model, and neither shows a key." }),
    catalogueError(listed),
    project.options.length ? null : noProjectNote(),
    form,
  ];
}

// What the run directory already holds, as Testuary names the directory for
// this project, model and run directory: an earlier run there refuses a new
// one, so the form offers to continue it or to start a new run in a new
// directory.
function savedRunChoice(project, model, output) {
  const box = el("div", { "data-field": "resume", "aria-live": "polite" });
  let saved = { occupied: false };
  let asked = 0;
  const fresh = el("input", { type: "text", autocomplete: "off", id: "fresh_dir", "aria-label": "New run directory" });
  const check = async () => {
    const number = ++asked;
    if (!project.value) { saved = { occupied: false }; box.replaceChildren(); return; }
    const query = new URLSearchParams({ project_dir: project.value, model: model.value, output_dir: output.value.trim() });
    let found;
    try { found = await call(`/api/run/saved?${query}`); } catch (error) { found = { occupied: false }; }
    if (number !== asked) return;
    saved = found;
    if (!saved.occupied) { box.replaceChildren(); return; }
    fresh.value = saved.fresh;
    const left = saved.unfinished.length;
    const options = [
      left ? ["continue", `Continue the earlier run: it finishes the ${left === 1 ? "target" : `${left} targets`} without a suite that meets the minimums and keeps the finished work.`] : null,
      ["fresh", "Start a new run in a new run directory:"],
    ].filter(Boolean);
    const group = choices({ name: "saved", legend: "What to do with the earlier run in this directory", options });
    // The new directory's name on a line of its own, below its choice.
    group.querySelector('label[for="saved-fresh"]').parentElement.after(fresh);
    const sync = () => { fresh.disabled = group.querySelector("input:checked").value !== "fresh"; };
    group.addEventListener("change", sync);
    sync();
    box.replaceChildren(el("div", { class: "notice-box" },
      el("p", {}, "This project already has results from this model in ", pathText(saved.full, saved.run_dir),
        left ? "." : saved.summary ? ". Every target there is finished, so a new run needs a new directory."
          : ". Testuary refuses a new run in it, so a new run needs a new directory."),
      group));
  };
  for (const input of [project, model]) input.addEventListener("change", check);
  output.addEventListener("change", check);
  check();
  // The request's resume and run directory for the chosen answer.
  const chosen = () => {
    if (!saved.occupied) return {};
    const answer = (box.querySelector('input[name="saved"]:checked') || {}).value;
    if (answer === "continue") return { resume: true };
    return { output_dir: fresh.value.trim() };
  };
  return { box, chosen, check };
}

// What a model alias is to the runs here: the default model, the one the
// global settings choose, and the working directory's project's own model.
function modelMarks(alias, defaultModel, projectModel) {
  const marks = alias === defaultModel ? ["default model"] : [];
  if (projectModel && alias === projectModel.model) marks.push(`${projectModel.name}'s model`);
  return marks;
}

// The model a run asks when its Model field is left empty, in words: the
// project's own, else the default one.
function emptyModelText(project, defaultModel) {
  const own = project && project.fields && project.fields.model;
  if (own) return `The project's model (${own})`;
  return defaultModel ? `The default model (${defaultModel})` : "The default model";
}

// A run of a listed project in its directory, as `testuary [FILES]` runs
// there, with the options set here.
async function runView(query) {
  const [catalogue, listed] = await Promise.all([call("/api/catalogue")
    .catch((error) => ({ error: error.message, models: {} })),
  call("/api/projects").catch((error) => ({ error: error.message, projects: [] }))]);
  const models = Object.keys(catalogue.models || {});
  const project = projectChoice(listed, query);
  const inherited = el("option", { value: "" });
  const model = el("select", {}, inherited, models.map((alias) => el("option", { value: alias, text: alias })));
  const nameInherited = () => {
    inherited.textContent = emptyModelText((listed.projects || []).find((entry) => entry.project_dir === project.value),
      catalogue.default_model);
  };
  project.addEventListener("change", nameInherited);
  nameInherited();
  const text = (placeholder) => el("input", { type: "text", autocomplete: "off", placeholder });
  const number = (value) => el("input", { type: "number", min: 1, step: 1, value, inputmode: "numeric" });
  const files = fileChecklist(project);
  // Repetitions per file or one repetition: alternatives, the unused one grayed.
  const runs = number(1);
  const only = number(0);
  only.min = 0;
  const repeat = choices({ name: "runs_mode", legend: "Repetitions", options: [
    ["all", "All repetitions"], ["one", "One repetition only"]] });
  const runsField = field({ id: "runs", label: "Repetitions per target", option: "--runs", input: runs, unit: "repetitions",
    hint: "Each target's suites are generated and measured this many times, each repetition on its own, for statistics across repetitions." });
  const onlyField = field({ id: "run", label: "Which repetition", option: "--run", input: only,
    hint: "Repeat one repetition by its number, 0 for r0, such as one that did not finish." });
  const syncRuns = () => {
    const one = repeat.querySelector("input:checked").value === "one";
    runs.disabled = one;
    only.disabled = !one;
    runsField.classList.toggle("disabled", one);
    onlyField.classList.toggle("disabled", !one);
  };
  repeat.addEventListener("change", syncRuns);
  const form = el("form", { class: "panel", id: "run-form", novalidate: true },
    field({ id: "project_dir", label: "Project", input: project }),
    files.box,
    field({ id: "model", label: "Model", option: "--model", input: model,
      hint: "The model this run asks. Empty: the project's model, else your default model." }),
    repeat, runsField, onlyField,
    field({ id: "output_dir", label: "Run directory", option: "--output-dir", input: text("a path from the project directory"),
      hint: "Where this run writes. A relative path starts at the project directory. Empty: the project's [project] output_dir, by default a directory for the model in the results directory of the project's Testuary directory." }));
  const output = form.querySelector("#output_dir");
  if (query.get("model")) model.value = query.get("model");
  if (query.get("output_dir")) output.value = query.get("output_dir");
  const saved = savedRunChoice(project, model, output);
  form.append(saved.box);
  syncRuns();
  jobForm(form, "Start the run", () => {
    if (files.empty()) throw NO_FILE;
    const one = repeat.querySelector("input:checked").value === "one";
    return ["/api/run", {
      project_dir: project.value, files: files.chosen(), model: model.value,
      runs: one ? "" : runs.value, run: one ? only.value : "", output_dir: output.value,
      ...saved.chosen(),
    }];
  }, () => project.options.length > 0);
  return [
    el("h1", { text: "Start a run" }),
    el("p", { text: "A run generates and measures test suites for the project's targets with the chosen model. It sends model requests, which can cost money with a paid provider. The project's settings can set small limits for a first run." }),
    catalogueError(catalogue), catalogueError(listed),
    project.options.length ? null : noProjectNote(),
    form,
  ];
}

// A check's report: Testuary's own summary as `testuary check` prints it,
// the report's facts and items as data, and Testuary's detailed listing.
function reportView(report, texts, job) {
  if (!report) {
    return el("p", { class: "panel", text: job && job.status === "cancelled"
      ? "The check was stopped before it printed a report."
      : "The check printed no report. Its error output is below." });
  }
  // Testuary's label of each phase the report names.
  const phaseName = (id) => (texts && texts.phases && texts.phases[id]) || id;
  // Each item's plain name and status word, as the summary above says them.
  const words = (key) => (texts && texts.items && texts.items[key]) || {};
  const items = report.items || [];
  const key = report.credentials_configured === true ? "set, not tried" : report.credentials_configured === false ? "not set" : "not known";
  const order = { failed: 0, refused: 0, missing_setup: 1, unsupported: 2, unverified: 3, verified: 4 };
  const sorted = [...items].sort((a, b) => (order[a.status] ?? 5) - (order[b.status] ?? 5));
  const facts = [
    // The job's word for its status, as its badge says it.
    ["Result", el("span", {}, job ? jobState(job).badge : stateBadge(report.status), ` exit status ${report.exit_code}`)],
    ["Model", report.model_alias
      ? `${report.model_alias} (${report.model || "model name unknown"} at the service, provider ${report.provider || "unknown"})`
      : "not configured"],
    ["API key", key],
    ["Source directory", cell(report.source_dir)],
    ["Source files", (report.source_files || []).join(", ") || cell("")],
    ["Phases", (report.phases || []).map(phaseName).join(", ") || cell("")],
    ["Run directory", cell(report.output_dir)],
  ];
  const summary = texts && texts.summary;
  const detailed = texts && texts.detailed;
  return el("div", {},
    summary ? el("div", { class: "panel" }, el("h2", { text: "What the check found" }),
      el("div", { class: "console-text", text: summary.join("\n") })) : null,
    el("div", { class: "panel rows" }, facts.flatMap(([name, value]) => [el("div", { class: "name", text: name }), el("div", {}, value)])),
    // The items repeat the summary above as data: folded, as the limits are.
    el("details", { class: "panel scroll" }, el("summary", { text: `Items (${items.length})` }),
      stackedTable(["Status", "Item", "Reason", "What to do"], sorted.map((item) => el("tr", {},
        el("td", {}, el("span", { class: `state ${item.status}`, text: (words(item.key).status || item.status).toUpperCase() })),
        el("td", {}, words(item.key).name || item.key, el("div", { class: "small muted" }, el("code", { text: item.key })),
          item.selected ? el("div", { class: "small muted", text: item.selected }) : null),
        el("td", { text: item.reason || "" }), el("td", { text: item.action || "" }))))),
    report.limits && report.limits.length ? el("details", { class: "panel scroll" }, el("summary", { text: `Limits (${report.limits.length})` }),
      stackedTable(["Phase", "Limit", "Value", "State"], report.limits.map((limit) => el("tr", {},
        el("td", { text: phaseName(limit.phase) }), el("td", { text: limit.name }),
        el("td", { class: "num", text: limit.value !== null && limit.value !== undefined
          ? `${limit.value} ${limit.unit || ""}` : limit.state === "unlimited" ? "no limit" : "unknown" }),
        el("td", { text: limit.state }))))) : null,
    detailed ? el("details", { class: "panel" }, el("summary", { text: "The detailed report" }),
      el("div", { class: "log", tabindex: 0, "aria-label": "The detailed report", text: detailed.join("\n") })) : null);
}

// A job's status: one word and one color per state, the same in every view.
// Each job state's style and its one word, which the badge shows in
// capitals and the label as it is.
const JOB_STATES = {
  running: ["running", "Running"], stopping: ["stopping", "Stopping"], finished: ["finished", "Finished"],
  cancelled: ["cancelled", "Stopped"], stopped: ["stopped", "Force-stopped"], ended: ["failed", "Failed"],
};

// A run that ended without a suite that meets the minimums for every file, as Testuary marks it.
const incompleteRun = (job) => job.status === "finished" && Boolean(job.outcome && job.outcome.incomplete);

// What a stop meant for a demonstration or a check, in the page's words:
// the terminal's sentence names Ctrl+C and an option the page sets itself.
const STOPPED_MEANINGS = {
  demo: "The demonstration was stopped. Start it again from the Demonstration page.",
  check: "The check was stopped. No setup was changed.",
};

// What Stop does, by the kind of job.
const STOP_HINTS = {
  run: "Stops the run. Work in progress is cleaned up first, and finished work is kept.",
  demo: "Stops the playback.",
  check: "Stops the check. No setup is changed.",
};

// A job state's style, its badge's word and its label.
function stateWords(job) {
  if (incompleteRun(job)) return ["incomplete", "MINIMUMS NOT MET", "Finished without a suite that meets the minimums"];
  if (job.word && (job.status === "finished" || job.status === "ended")) {
    // Testuary's word for the command's status: red only for a failure.
    const style = job.status === "finished" ? "finished" : job.failure ? "failed" : "incomplete";
    return [style, job.word.toUpperCase(), job.word[0].toUpperCase() + job.word.slice(1)];
  }
  const [style, label] = JOB_STATES[job.status] || ["waiting", job.status];
  return [style, label.toUpperCase(), label];
}

function jobState(job) {
  const [style, badge, label] = stateWords(job);
  return { badge: el("span", { class: `state ${style}`, text: badge }), label };
}

// Where a scrolling text box shows its new text, from its last position
// (scrollTop, scrollHeight, clientHeight, or null for a new box): at its end
// while the reader is at the end, so it follows the output, else where the
// reader left it. A hidden box has no height and follows.
function nextScroll(previous, scrollHeight) {
  if (!previous || previous.scrollHeight - previous.scrollTop - previous.clientHeight <= 4) return scrollHeight;
  return previous.scrollTop;
}

const scrollPlace = (box) => ({ scrollTop: box.scrollTop, scrollHeight: box.scrollHeight, clientHeight: box.clientHeight });

// Draws what ``drawn()`` returns into ``box`` anew, keeping each text box's
// place and each part's open or closed state: by their names for what is
// drawn anew, and as they were for a nested run's insides, which the drawing
// moves.
function redrawKeeping(box, drawn) {
  const key = (node) => (ownNode(box, node) ? node.getAttribute("aria-label") : node);
  const places = new Map([...box.querySelectorAll(".log")].map((log) => [key(log), scrollPlace(log)]));
  const summaries = () => [...box.querySelectorAll("details > summary")].filter((node) => ownNode(box, node));
  const opened = new Map(summaries().map((summary) => [summary.textContent, summary.parentElement.open]));
  box.replaceChildren(drawn());
  for (const summary of summaries()) {
    if (opened.has(summary.textContent)) summary.parentElement.open = opened.get(summary.textContent);
  }
  for (const log of box.querySelectorAll(".log")) log.scrollTop = nextScroll(places.get(key(log)) || null, log.scrollHeight);
}

async function jobView(id) {
  const job = await call(`/api/jobs/${id}`);
  const progress = new ViewPanel();
  progress.kind = extensions.job_kinds.find((item) => item.id === job.kind) || null;
  const progressBox = el("div", {});
  const terminal = el("div", { class: "terminal", tabindex: 0, "aria-label": "Terminal text", text: "" });
  const lines = [];
  let current = job;
  let pending = false;

  const draw = () => {
    pending = false;
    redrawKeeping(progressBox, () => (progress.snapshot || progress.log.length ? progress.render()
      : el("div", {}, progress.causePanel(), el("p", { class: "muted", text: !current.events ? ""
        : current.ended === null ? "Waiting for the first progress event." : "The job wrote no progress events." }))));
    const place = scrollPlace(terminal);
    terminal.textContent = outputText(lines, current.ended !== null);
    terminal.scrollTop = nextScroll(place, terminal.scrollHeight);
  };
  const later = () => { if (!pending) { pending = true; requestAnimationFrame(draw); } };

  // The Testuary runs an extension's job names, each folded into its phase:
  // made once and kept, so a redraw keeps each part open and reading. A
  // phase that changes state may have started one, so the page asks again.
  progress.nested = job.nested_runs ? { runs: new Map(), error: null } : null;
  const nestedRuns = (found) => {
    if (!progress.nested || !found.nested_runs) return;
    progress.nested.error = found.nested_runs_error || null;
    for (const run of found.nested_runs) {
      const known = progress.nested.runs.get(run.path);
      if (known) known.phase = run.phase;
      else progress.nested.runs.set(run.path, { phase: run.phase, part: nestedRun(run.path, run.title) });
    }
    // An ended job's runs have ended, also one that wrote no end of its own.
    if (found.ended !== null) for (const run of progress.nested.runs.values()) run.part.dispatchEvent(new Event("jobended"));
    later();
  };
  nestedRuns(job);
  let phasesSeen = null;

  // The status panel is built once. Its state word is announced when it
  // changes. The clock and the force countdown update in place, so a button
  // keeps the keyboard focus.
  const announced = el("span", { role: "status", "aria-live": "polite", class: "status-line" });
  const time = el("span", { class: "muted figures" });
  const meaning = el("p", {});
  const actions = el("div", { class: "actions" });
  const panel = el("div", { class: "panel status-panel", tabindex: -1 },
    el("div", { class: "status-line" }, announced, time), meaning,
    // The terminal's command for this job, folded: the page itself needs none.
    el("details", { class: "small muted command" }, el("summary", { text: "The command" }),
      el("code", { text: job.command })), actions);
  let shown = null;
  let force = null;
  // A button that is pressed is disabled until the server answers, which
  // takes the keyboard focus from it: the focus returns to the new actions.
  let refocus = false;

  const post = async (button, path) => {
    refocus = actions.contains(document.activeElement);
    button.disabled = true;
    // The header shows the new state at once, not at its next look.
    try { current = await call(path, {}); drawStatus(); watchActive(); }
    catch (error) {
      showMessage(error.message);
      button.disabled = false;
      if (refocus) button.focus();
      refocus = false;
    }
  };
  // Returns whether the keyboard focus was on the actions or the panel, to
  // move it to the new actions once their state is set.
  const buildActions = () => {
    const focused = refocus || actions.contains(document.activeElement) || document.activeElement === panel;
    refocus = false;
    force = null;
    if (current.status === "running") {
      const stop = el("button", { class: "danger", type: "button", text: "Stop" });
      stop.addEventListener("click", () => post(stop, `/api/jobs/${id}/stop`));
      actions.replaceChildren(stop, el("span", { class: "hint", text: STOP_HINTS[current.kind] || STOP_HINTS.run }));
    } else if (current.status === "stopping") {
      force = el("button", { class: "danger", type: "button" });
      force.addEventListener("click", () => post(force, `/api/jobs/${id}/force`));
      actions.replaceChildren(force, el("span", { class: "hint", text: "Stopping. Work in progress is being cleaned up, then the job ends." }));
    } else {
      // The directory the ended run wrote: none when it stopped before making it.
      const dir = current.run_dir;
      const left = (current.outcome && current.outcome.unfinished) || 0;
      // An extension's job kind continues as the kind says.
      const kind = extensions.job_kinds.find((item) => item.id === current.kind);
      const resume = left || (kind && kind.resume) ? el("button", { type: "button", class: dir ? "" : "primary",
        text: kind ? "Continue" : "Continue the run" }) : null;
      if (resume) {
        resume.addEventListener("click", async () => {
          resume.disabled = true;
          try {
            const next = await call(`/api/jobs/${id}/continue`, {});
            location.hash = `#/job/${next.id}`;
          } catch (error) { showMessage(error.message); resume.disabled = false; }
        });
      }
      actions.replaceChildren(...[
        dir && tabShown("results") ? el("a", { class: "button primary", href: `#/results?run_dir=${encodeURIComponent(dir)}`, text: `Open the ${tabTitleInText("results")}` }) : null,
        resume,
        resume && !kind ? el("span", { class: "hint", text: `Runs again in the same run directory and finishes the ${left === 1 ? "target" : `${left} targets`} without a suite that meets the minimums. Finished work is kept.` }) : null,
      ].filter(Boolean));
    }
    return focused;
  };
  const drawStatus = () => {
    const { badge, label } = jobState(current);
    panel.className = `panel status-panel ${stateWords(current)[0]}`;
    let focus = false;
    if (shown !== current.status) {
      shown = current.status;
      announced.replaceChildren(badge, label);
      focus = buildActions();
    }
    time.textContent = `${current.ended === null ? "for" : "after"} ${hms(current.elapsed_s)}`;
    if (force) {
      const wait = current.force_after_s || 0;
      force.disabled = wait > 0;
      force.textContent = wait > 0 ? `Force stop, possible in ${Math.ceil(wait)} seconds` : "Force stop";
    }
    // A finished run states its own outcome, as its end-of-run summary does.
    // A run the Stop button ended says so in the page's words: the console's
    // sentence names Ctrl+C and a terminal option.
    const left = (current.outcome && current.outcome.unfinished) || 0;
    // A run that stopped before making its run directory says so. Its own
    // error text, when it printed one, is the cause panel's.
    const wroteNone = current.ended !== null && current.kind !== "check" && !current.run_dir
      && !extensions.job_kinds.some((item) => item.id === current.kind);
    meaning.textContent = current.status === "finished" && current.outcome && current.outcome.headline
      ? current.outcome.headline
      : current.kind === "run" && current.status === "cancelled"
        ? (wroteNone ? "The run was stopped before it wrote any results."
          : `The run was stopped. Work that already finished is kept.${left ? " Continue the run picks it up." : ""}`)
        : current.status === "cancelled" && STOPPED_MEANINGS[current.kind] ? STOPPED_MEANINGS[current.kind]
        : wroteNone
          ? `The run wrote no results.${current.error ? "" : ` ${exitMeaning(current)}`}`
          : exitMeaning(current);
    if (focus) (actions.querySelector("button:not(:disabled), a") || panel).focus();
  };

  const timer = setInterval(() => {
    if (current.ended === null) {
      if (progress.received) tickClocks(progressBox, progress.received);
      current.elapsed_s = (Date.now() / 1000) - current.started;
      if (current.force_after_s) current.force_after_s = Math.max(0, current.force_after_s - 1);
      drawStatus();
    }
  }, 1000);
  leaving.push(() => clearInterval(timer));

  const reportBox = el("div", {});
  // A run that ended before its plan, in a run directory that holds a saved
  // run: Testuary refused it for that directory. Said plainly, with the run
  // form's choices one step away.
  const refusedBox = el("div", {});
  const showSavedRefusal = async () => {
    if (current.kind !== "run" || current.status !== "ended" || !current.project
        || (progress.snapshot && progress.snapshot.diagram)) return;
    const option = (name) => {
      const found = (current.arguments || []).find((item) => item.startsWith(`${name}=`));
      return found ? found.slice(name.length + 1) : "";
    };
    const query = new URLSearchParams({ project_dir: current.project, model: option("--model"),
      output_dir: option("--output-dir") });
    let saved;
    try { saved = await call(`/api/run/saved?${query}`); } catch (error) { return; }
    if (!saved.occupied) return;
    progress.hideCause = true;
    later();
    refusedBox.replaceChildren(el("div", { class: "panel notice" },
      el("p", {}, "The run did not start: this project already has results from this model in ",
        pathText(saved.full, saved.run_dir), ". Continue that earlier run, or start a new run in a new run directory."),
      tabShown("run") && el("div", { class: "actions" }, el("a", { class: "button primary", href: `#/run?${query}`,
        text: `Choose on the ${tabTitle("run")} page` }))));
  };
  const warningsBox = el("div", {});
  const showWarnings = async () => {
    // The run an ended job wrote. While it runs, run_dir names where it writes.
    const dir = current.ended !== null ? current.run_dir : null;
    if (!dir) return;
    try {
      const found = await call(`/api/results?run_dir=${encodeURIComponent(dir)}`);
      const listed = [];
      for (const run of found.runs || []) {
        for (const file of run.files || []) {
          for (const item of file.runs || []) {
            for (const warning of item.warnings || []) listed.push({ ...warning, where: `${file.name}, run ${item.name}` });
          }
        }
      }
      warningsBox.replaceChildren(listed.length ? el("div", { class: "panel notice" },
        el("h2", { text: "Warnings" }),
        el("ul", { class: "warnings" }, listed.map((warning) => el("li", {},
          el("b", { text: `${warning.where}: ` }), warningList([warning], settingsOf(current.project || found.project_dir))))))
        : el("div", {}));
    } catch (error) {
      warningsBox.replaceChildren(el("p", { class: "small muted", text: `The run's warnings cannot be read: ${error.message}` }));
    }
  };
  const stopping = new AbortController();
  leaving.push(() => stopping.abort());
  let ended = false;
  const handlers = {
    // Every connection replays the job from its start.
    open: () => { progress.reset(); lines.length = 0; later(); },
    view: (text) => {
      progress.snapshot = JSON.parse(text);
      progress.received = Date.now();
      const phases = progress.snapshot.diagram ? progress.snapshot.diagram.phases.map((phase) => phase.state).join() : "";
      if (progress.nested && phases !== phasesSeen) {
        phasesSeen = phases;
        call(`/api/jobs/${id}`).then(nestedRuns, () => {});
      }
      later();
    },
    log: (text) => { progress.addLine(JSON.parse(text)); later(); },
    summary: (text) => { progress.summary = JSON.parse(text); later(); },
    text: (text) => {
      lines.push(JSON.parse(text).line);
      if (lines.length > 5000) lines.shift();
      later();
    },
    end: (text) => {
      ended = true;
      progress.ended = true;
      current = JSON.parse(text);
      // A refused or problem job: the cause panel says so, as the badge does.
      progress.endedWord = current.status === "ended" ? { word: current.word, failure: current.failure } : null;
      drawStatus();
      watchActive();
      nestedRuns(current);
      if (current.kind === "check") {
        // A check's report and error text arrive once, with its final state.
        reportBox.replaceChildren(reportView(current.report, current.report_text, current));
        lines.length = 0;
        if (current.errors) lines.push(...current.errors.replace(/\n$/, "").split("\n"));
      } else {
        progress.error = current.error || null;
        showWarnings();
        showSavedRefusal();
        // An extension's job that wrote a setup report shows it as a check does.
        if (current.report) reportBox.replaceChildren(reportView(current.report, current.report_text, current));
      }
      later();
    },
  };
  (async () => {
    // A connection that breaks before the end is opened again, a few times.
    for (let attempt = 0; !ended && !stopping.signal.aborted; attempt += 1) {
      try {
        await follow(`/api/jobs/${id}/stream`, handlers, stopping.signal);
        attempt = 0;
      } catch (error) {
        if (stopping.signal.aborted) return;
        if (attempt >= 5) { showMessage(`The job's progress stopped arriving: ${error.message}`); return; }
      }
      if (!ended) await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  })();
  drawStatus();
  draw();
  const parts = [el("h1", { text: job.title }), panel, refusedBox, warningsBox];
  if (job.kind === "check") parts.push(reportBox);
  else parts.push(progressBox, reportBox);
  // A run's progress view already lists what happened: its raw text stays folded.
  parts.push(el("details", { class: "panel" },
    el("summary", { text: job.kind === "check" ? "Error output" : "Terminal text" }), terminal));
  return parts;
}

function fileView(box, root, path) {
  box.replaceChildren(el("p", { class: "muted", text: `Opening ${path}.` }));
  call(`/api/results/file?run_dir=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`).then((file) => {
    const head = el("h3", {}, el("code", { text: file.path }), ` ${number(file.size)} bytes`);
    if (file.text === null) { box.replaceChildren(head, el("p", { class: "muted", text: file.note })); return; }
    const count = file.text.split("\n").length;
    const numbers = Array.from({ length: count }, (_, i) => String(i + 1)).join("\n");
    box.replaceChildren(head, el("div", { class: "code", tabindex: 0, "aria-label": `Contents of ${file.path}` },
      el("pre", { class: "gutter", "aria-hidden": "true", text: numbers }), el("pre", { text: file.text })));
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }).catch((error) => box.replaceChildren(el("p", { class: "error", text: error.message })));
}

function fileList(files, open) {
  return el("ul", { class: "files" }, files.map((file) => el("li", {},
    el("button", { type: "button", class: "link-button", text: file.path, onclick: () => open(file.path) }),
    el("span", { class: "small muted", text: ` ${number(file.size)} bytes` }))));
}

// The results dashboard. Every number and word comes from Testuary's result
// files as the server reads them. The page lays them out and computes no
// value of its own: no mean, no sum, no comparison.

function known(value) {
  return value && value.text !== null && value.text !== undefined
    ? value.text : el("span", { class: "unknown", text: "unknown" });
}

// A track from 0 to 100 percent. The text beside it states the value, so
// assistive technology skips the drawing.
function percentBar(value) {
  return el("span", { class: "track", "aria-hidden": "true" },
    value && value.value !== null ? el("span", { class: "fill", style: { "--value": value.value } }) : null);
}

// The lowest to the highest value of a file's runs, with a mark at their
// mean, as Testuary's summary across the runs states them.
function rangeBar(item) {
  const has = (value) => value && value.value !== null;
  return el("span", { class: "track", "aria-hidden": "true" },
    item && has(item.minimum) && has(item.maximum)
      ? el("span", { class: "range", style: { "--low": item.minimum.value, "--high": item.maximum.value } }) : null,
    item && has(item.mean) ? el("span", { class: "tick", style: { "--median": item.mean.value } }) : null);
}

function runName(file, item) {
  return [el("code", { text: item.name }), item.fully_measured === false ? el("span", { class: "small muted", text: " not fully measured" }) : null];
}

// A cut mutation measurement: its lower bound, marked incomplete.
function cutText(cut) {
  const left = cut.untested ? `: ${cut.untested} ${cut.untested === 1 ? "mutant" : "mutants"} not tested` : "";
  return el("span", { class: "warn", title: "Mutation testing left mutants untested. Untested mutants count as not killed.",
    text: `at least ${cut.lower_bound.text} (not fully measured${left})` });
}

function fileRows(file, chosen) {
  const rows = file.runs.map((item) => chosen === "mutation_score" && !item.values[chosen] && item.cut
    ? el("div", { class: "chart-row" },
      el("span", { class: "chart-name" }, "Repetition ", runName(file, item)),
      percentBar(item.cut.lower_bound), el("span", { class: "figures" }, cutText(item.cut)))
    : el("div", { class: "chart-row" },
    el("span", { class: "chart-name" }, "Repetition ", runName(file, item), item.warnings && item.warnings.length
      ? el("span", { class: "warn small", title: item.warnings.map((warning) => warning.text).join(" "),
        text: ` ${item.warnings.length === 1 ? "1 warning" : `${item.warnings.length} warnings`}` }) : null),
    percentBar(item.values[chosen]), el("span", { class: "figures" }, known(item.values[chosen]))));
  const across = file.across && file.across.values[chosen];
  if (file.runs.length > 1 && across) {
    const text = (value) => (value && value.text) || "unknown";
    rows.push(el("div", { class: "chart-row" }, el("span", { class: "chart-name", text: "Across the repetitions" }),
      rangeBar(across), el("span", { class: "figures", text: across.runs
        ? `mean ${text(across.mean)}, lowest ${text(across.minimum)}, highest ${text(across.maximum)}, of ${across.runs} ${across.runs === 1 ? "repetition" : "repetitions"}`
        : "no repetition measured it" })));
  }
  const starter = file.across && file.across.starter_median && file.across.starter_median[chosen];
  if (starter) {
    // With several repetitions, the file's value is the mean of each
    // repetition's median.
    const which = file.runs.length > 1 ? "mean of the repetitions' starter-suite medians"
      : "median of the starter suites";
    rows.push(el("div", { class: "chart-row starter" },
      el("span", { class: "chart-name", text: "Starter suites" }),
      percentBar(starter), el("span", { class: "figures" }, `${which} ${starter.text}`)));
  }
  return rows;
}

// A distribution: one mark per file on a track from 0 to 100 percent.
function dotStrip(values, starter) {
  return el("span", { class: "track strip", "aria-hidden": "true" },
    (starter || []).map((value) => el("span", { class: "dot starter", style: { "--value": value.value } })),
    values.map((value) => el("span", { class: "dot", style: { "--value": value.value } })));
}

// The project across its files, as Testuary's project summary states it.
// A project card's figures across its files: a median, range and standard
// deviation describe several files, so one file's card names only that it
// is one.
function statRows(delivered, starter) {
  const text = (value) => (value && value.text) || "unknown";
  const rows = delivered.count > 1 ? [["Median", text(delivered.median)],
    ["Range", `${text(delivered.minimum)} to ${text(delivered.maximum)}`],
    ["Standard deviation", delivered.std ? `${delivered.std.text} points` : "unknown"]] : [];
  rows.push(["Targets", String(delivered.count)]);
  if (starter && starter.count) {
    rows.push(["Starter suites", `median ${text(starter.median)}, mean ${text(starter.mean)} of the targets' values`]);
  }
  return rows;
}

function projectView(run) {
  if (!run.project) return null;
  const text = (value) => (value && value.text) || "unknown";
  const cards = run.values.map((value) => {
    const item = run.project[value.id] || {};
    const delivered = item.delivered;
    const starter = item.starter;
    const bound = item.lower_bound;
    const cut = bound && bound.count ? el("p", { class: "small warn" },
      `${bound.count} ${bound.count === 1 ? "target" : "targets"} not fully measured: mutation testing left mutants untested. `,
      bound.count === 1 ? `At least ${text(bound.mean)}.` : `At least ${text(bound.minimum)} to ${text(bound.maximum)}.`) : null;
    if (!delivered || !delivered.count) {
      return el("div", { class: "stat-card" }, el("h4", { text: value.name }),
        cut || el("p", { class: "muted small", text: "No target measured it." }));
    }
    return el("div", { class: "stat-card" }, el("h4", { text: value.name }),
      el("p", { class: "stat-main" }, el("span", { class: "big", text: text(delivered.mean) }),
        delivered.count > 1 ? " mean" : " in its one target"),
      el("dl", { class: "fields" }, statRows(delivered, starter).map(([name, shown]) =>
        [el("dt", { text: name }), el("dd", { text: shown })])),
      el("div", { class: "strip-row" }, dotStrip(delivered.values, starter && starter.values),
        el("span", { class: "small muted", text: "Filled: each target's mean. Hollow: the median of its starter suites, with several repetitions the mean of the repetitions' medians." })), cut);
  });
  return el("div", { class: "panel" }, el("h3", { text: "The project across its targets" }),
    el("div", { class: "stat-cards" }, cards));
}

function scoreChart(run, index) {
  if (!run.values.length || !run.files.some((file) => file.runs.length)) return null;
  let chosen = run.values.some((value) => value.id === "mutation_score") ? "mutation_score" : run.values[0].id;
  const body = el("div", {});
  const draw = () => body.replaceChildren(...run.files.filter((file) => file.runs.length).map((file) =>
    el("div", { class: "chart-group" }, el("div", { class: "chart-label" }, el("code", { text: file.name })),
      fileRows(file, chosen))));
  const switcher = el("fieldset", { class: "score-switch" }, el("legend", { text: "Value" }),
    run.values.map((value) => el("label", { class: "check" },
      el("input", { type: "radio", name: `score-${index}`, value: value.id, checked: value.id === chosen || undefined,
        onchange: () => { chosen = value.id; draw(); } }), value.name)));
  draw();
  return el("div", { class: "panel" }, el("h3", { text: "Values per target" }), switcher, body,
    el("p", { class: "small muted", text: "Bars run from 0 to 100 percent. Each repetition's bar is its delivered suite's value. With several repetitions, the bar across them spans their lowest to highest value, with a mark at their mean. The starter suites' bar is the median of the accepted starter suites before any loop, and with several repetitions the mean of the repetitions' medians, as Testuary's summary of the target states it." }));
}

function endOfRun(end) {
  if (!end) return el("p", { class: "panel small muted", text: "No end-of-run summary in this run's directory: the run has not ended, or it ended before writing one." });
  if (!end.lines) return el("p", { class: "panel small muted", text: "This run's end-of-run summary cannot be read." });
  return el("details", { class: "panel" }, el("summary", { text: "End of the run, as the console printed it" }),
    el("div", { class: "console-text", text: end.lines.join("\n") }));
}

// A run's details as Testuary's summary holds them, named by its labels.
// A value a phase's summary holds, else "none": the phase does not
// record it, such as finishing's model use.
function recorded(values, name) {
  return name in values ? known(values[name]) : el("span", { class: "muted", text: "none" });
}

function detailGrids(item, labels) {
  const word = (name) => labels[name] || name;
  const details = item.details || {};
  const parts = [];
  const phases = Object.entries(details.phases || {});
  if (phases.length) {
    const names = [...new Set(phases.flatMap(([, values]) => Object.keys(values)))];
    parts.push(el("h4", { text: "Each phase's suite, model use and time" }),
      grid(["", ...phases.map(([phase]) => word(phase)), word("finishing")],
        names.map((name) => [word(name), ...phases.map(([, values]) => recorded(values, name)),
          recorded(details.finishing || {}, name)])));
  }
  const starters = Object.entries(details.starter || {});
  if (starters.length && starters.some(([, values]) => Object.values(values).some(Boolean))) {
    const statistics = Object.keys(starters[0][1]);
    parts.push(el("h4", { text: "The accepted starter suites" }),
      grid(["", ...starters.map(([name]) => word(name))],
        statistics.map((statistic) => [word(statistic), ...starters.map(([, values]) => known(values[statistic]))])));
  }
  const time = Object.entries(details.time || {}).filter(([, value]) => value);
  if (time.length) {
    parts.push(el("h4", { text: "Where the time went" }),
      el("dl", { class: "fields" }, time.flatMap(([name, value]) => [el("dt", { text: word(name) }), el("dd", { text: value.text })])));
  }
  return parts;
}

function grid(head, rows) {
  return el("div", { class: "scroll" }, el("table", {},
    el("thead", {}, el("tr", {}, head.map((title) => el("th", { scope: "col", text: title })))),
    el("tbody", {}, rows.map(([name, ...cells]) => el("tr", {}, el("th", { scope: "row", text: name }),
      cells.map((value) => el("td", { class: "num" }, value)))))));
}

// How to use a run's delivered suite, under its suite directory: where the
// test file goes and the setup files it can need.
function suiteGuide(suite, demo) {
  if (!suite.tests.length) return null;
  if (demo) return "These tests belong to the demonstration's recorded project, not to a project of yours.";
  const use = "Use the delivered suite: copy the test file to where the project keeps its tests and run it "
    + "with the project's own test command.";
  const setup = suite.setup.map((path) => path.split("/").pop());
  if (setup.length) {
    const them = setup.length > 1 ? "them" : "it";
    const own = setup.length > 1 ? "a file of the same name" : `a ${setup[0]} of its own`;
    return `${use} Its tests ran with ${setup.join(" and ")} beside them and can need ${them}: copy ${them} `
      + `into the same directory too. Where that directory already has ${own}, the two have to be joined by hand.`;
  }
  return use;
}

// Copying a delivered test file into its project, into a directory the person
// chooses, by default where the project's own tests are. Nothing is
// written over a file.
function adoptAction(root, path) {
  const box = el("div", { class: "adopt" });
  const name = path.split("/").pop();
  const start = el("button", { type: "button", text: `Copy ${name} into the project` });
  start.addEventListener("click", async () => {
    let offer;
    try { offer = await call(`/api/results/adopt?run_dir=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`); }
    catch (error) { box.replaceChildren(start, el("span", { class: "error", text: ` ${error.message}` })); return; }
    const input = el("input", { type: "text", value: offer.test_dir, autocomplete: "off", name: "test_dir" });
    const shown = field({ id: `adopt-dir-${name}`, label: `Copy ${name} to the directory`, input, browse: offer.project,
      hint: "A directory inside the project, made when it is missing. The file keeps its name, and nothing is written over a file." });
    // A refusal names the field test_dir.
    shown.dataset.field = "test_dir";
    const copy = el("button", { type: "submit", class: "primary", text: "Copy" });
    const done = el("p", { class: "small saved", role: "status" });
    const form = el("form", { novalidate: true, class: "adopt-form" }, shown,
      el("div", { class: "actions" }, copy,
        el("button", { type: "button", text: "Cancel", onclick: () => { box.replaceChildren(start); start.focus(); } }),
        el("span", { class: "error", "aria-live": "polite" })), done);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      clearErrors(form);
      done.replaceChildren();
      copy.disabled = true;
      try {
        const answer = await call("/api/results/adopt", { run_dir: root, path, test_dir: input.value });
        done.replaceChildren("Copied to ", pathText(answer.written, shortPath(answer.written)), ".");
      } catch (error) { showError(form, error); }
      finally { copy.disabled = false; }
    });
    box.replaceChildren(form);
    input.focus();
  });
  box.append(start);
  return box;
}

function runDetails(file, item, summary, open) {
  const record = (entry) => entry ? `${entry.status_text || "unknown"}${entry.failure_kind ? `: ${entry.failure_kind_text || entry.failure_kind}` : ""}` : null;
  const parts = [
    el("p", { class: "small" }, "Target: ", el("code", { text: file.name })),
    item.summarized ? null : el("p", { class: "small muted", text: "This repetition has no summary yet: Testuary writes it when its row of tests.csv is written." }),
    el("h4", { text: "How each phase ended" }),
    el("div", { class: "scroll" }, stackedTable(["Phase", "How it ended", "Phase record"], item.phases.map((phase) =>
      el("tr", {}, el("th", { scope: "row", text: phase.label }),
        el("td", { class: "num" }, known({ text: phase.status_text || (phase.record && phase.record.status_text) })),
        el("td", { class: "num" }, known({ text: record(phase.record) })))))),
  ];
  if (item.phase_ends && item.phase_ends.length) {
    parts.push(el("h4", { text: "How each phase end chose its suite" }), item.phase_ends.map(phaseEnd));
  }
  if (item.refusals && item.refusals.length) {
    parts.push(el("h4", { text: "Measurements a tool could not make" }),
      el("ul", { class: "refusals" }, item.refusals.map((refusal) =>
        el("li", {}, el("b", { text: `${refusal.label}: ` }), refusal.text))));
  }
  if (item.warnings && item.warnings.length) {
    parts.push(el("h4", { text: "Warnings" }),
      warningList(item.warnings, settingsOf(summary.project_dir)));
  }
  parts.push(...detailGrids(item, summary.runs[0].labels || {}));
  if (item.suite) {
    const guide = suiteGuide(item.suite, summary.demo);
    parts.push(el("h4", {}, "Suite directory ", el("code", { text: item.suite.suite_dir })),
      item.suite.files.length ? fileList(item.suite.files, open) : el("p", { class: "muted", text: "The directory is empty." }),
      guide ? el("p", { class: "small", text: guide }) : null,
      summary.demo ? null : [...item.suite.tests, ...item.suite.setup].map((path) => adoptAction(summary.root, path)));
  }
  if (item.row !== null && summary.rows[item.row]) {
    const row = summary.rows[item.row];
    parts.push(el("details", {}, el("summary", { text: `All ${summary.columns.length} columns of this repetition's row of tests.csv` }),
      el("div", { class: "scroll" }, el("table", { class: "csv-row" }, el("tbody", {}, summary.columns.map((column) => el("tr", {},
        el("th", { scope: "row", text: column }), el("td", {}, cell(row[column])))))))));
  }
  return el("div", { class: "details" }, parts);
}

// A phase end's choice as its record states it: what it delivered and why,
// the starter suites that left without complete numbers, and each
// candidate with its numbers and outcome, one not chosen with its reason.
function phaseEnd(end) {
  return el("div", { class: "phase-end" },
    el("p", {}, el("b", { text: `${end.label}: ` }), end.summary || "no choice was recorded."),
    end.freed ? el("p", { class: "small", text: end.freed }) : null,
    end.candidates.length ? el("div", { class: "scroll" }, stackedTable(
      ["Candidate", "Tests", "Line coverage", "Branch coverage", "Mutation score", "Outcome"],
      end.candidates.map((candidate) => el("tr", { class: candidate.state === "delivered" ? "chosen" : null },
        el("th", { scope: "row", title: candidate.label, text: candidate.text }),
        el("td", { class: "num" }, known(candidate.tests)),
        ["line", "branch", "mutation"].map((axis) => el("td", { class: "num" }, candidateScore(candidate, axis))),
        el("td", {}, candidate.state_text, candidate.reason ? reasonText(candidate.reason) : null))))) : null);
}

// A candidate's score on one axis, a missing one as not measured, and a
// mutation score its record marks as derived with that label.
function candidateScore(candidate, axis) {
  const value = candidate[axis];
  if (!value) return el("span", { class: "muted", text: "not measured" });
  return axis === "mutation" && candidate.mutation_derived
    ? [value.text, el("span", { class: "small muted", text: " (derived, not measured)" })] : value.text;
}

// A candidate's reason as its record gives it: one line whole, more lines,
// such as a test's output, behind the first.
function reasonText(reason) {
  const [first, ...rest] = reason.split("\n");
  if (!rest.length) return el("div", { class: "small muted", text: reason });
  return el("details", { class: "small" }, el("summary", { text: first }),
    el("pre", { class: "reason", text: rest.join("\n") }));
}

// A warning's short line: its kind and phase, how often, its sentence on hover.
function warningLine(warning) {
  const times = warning.count > 1 ? `, ${warning.count} times` : "";
  return el("div", { class: "warn", title: warning.text || "" },
    `${warning.kind_label || "Warning"}: ${warning.phase_label || "the run"}${times}`);
}

// The settings page of the listed project a run belongs to, as the server
// names it, else the settings page.
function settingsOf(project) {
  return project ? `#/settings?project_dir=${encodeURIComponent(project)}` : "#/settings";
}

// A run's warnings as its summary states them: Testuary's sentence, then the
// setting it names by its plain label, its configured value, how often it
// was reached and where to raise it.
function warningList(warnings, settings = "#/settings") {
  return el("ul", { class: "warnings" }, warnings.map((warning) => el("li", {},
    warning.phase_label ? el("b", { text: `${warning.phase_label}: ` }) : null,
    el("span", { text: warningSentence(warning) }),
    warning.setting ? el("span", { class: "small muted" }, " ",
      el("span", { title: warning.setting, text: warning.setting_label || warning.setting }),
      warning.value ? `: ${warning.value.text}.` : ".",
      tabShown("settings") && [" ", el("a", { href: settings, text: "Raise it in the settings" }), "."]) : null)));
}

// Testuary's sentence without its bracket naming the setting's key and
// value, which the line after it names in plain words. A sentence written
// otherwise stays whole.
function warningSentence(warning) {
  const text = warning.text || `${warning.kind_label || "Warning"}: ${warning.phase_label || "the run"}`;
  if (!warning.setting || !warning.value) return text;
  const unit = warning.unit ? ` ${warning.unit}` : "";
  return text.replace(` (${warning.setting} = ${warning.value.value}${unit})`, "");
}

function filesTable(run, summary, open, index) {
  const head = ["Target", "Repetition", ...run.values.map((value) => value.name), "How the phases ended"];
  const body = [];
  run.files.forEach((file, fileIndex) => file.runs.forEach((item, runIndex) => {
    const id = `details-${index}-${fileIndex}-${runIndex}`;
    const detail = el("tr", { id, class: "detail", hidden: true },
      el("td", { colspan: head.length }, runDetails(file, item, summary, open)));
    const toggle = el("button", { type: "button", class: "link-button", "aria-expanded": "false", "aria-controls": id,
      onclick: () => {
        detail.hidden = !detail.hidden;
        toggle.setAttribute("aria-expanded", String(!detail.hidden));
      } }, file.name);
    body.push(el("tr", {},
      el("td", {}, toggle),
      el("td", {}, runName(file, item)),
      run.values.map((value) => el("td", { class: "num" }, value.id === "mutation_score" && !item.values[value.id] && item.cut
        ? cutText(item.cut) : known(item.values[value.id]))),
      el("td", { class: "small" }, item.phases.filter((phase) => phase.status || phase.record).map((phase) =>
        el("div", {}, `${phase.label}: `, known({ text: phase.status_text || (phase.record && phase.record.status_text) }))),
        (item.warnings || []).map(warningLine))), detail);
  }));
  const table = stackedTable(head, body);
  table.classList.add("files-table");
  return el("div", { class: "panel scroll" }, table,
    el("p", { class: "small muted", text: "Select a target for each phase end, the delivered suite and the repetition's row of tests.csv." }));
}

function runSection(run, summary, open, index) {
  const parts = [el("h2", { text: `Results in ${run.name}` }),
    el("p", { class: "small muted path" }, "Run directory ", pathText(run.run_dir, run.name))];
  if (!run.files.length) {
    parts.push(endOfRun(run.summary),
      el("p", { class: "panel muted", text: "This run directory has no results yet: no target's directory, summary or phase record." }));
    return parts;
  }
  parts.push(projectView(run), scoreChart(run, index), el("h3", { text: "Targets and repetitions" }),
    filesTable(run, summary, open, index), endOfRun(run.summary));
  return parts;
}

// A project's runs, as its Testuary directory's results directory holds them,
// the newest first: each opens on the dashboard.
async function projectRuns(project, dir) {
  if (!project.value) return null;
  const box = el("div", { class: "panel" });
  let found;
  try { found = await call(`/api/project/results?project_dir=${encodeURIComponent(project.value)}`); }
  catch (error) { box.append(el("p", { class: "error", text: error.message })); return box; }
  box.append(el("h2", { text: "Runs of this project" }));
  if (!found.runs.length) {
    box.append(el("p", { class: "muted", text: "No results yet in the project's results directory." }));
    return box;
  }
  box.append(el("ul", { class: "list" }, found.runs.map((item) => el("li", { class: item.run_dir === dir ? "chosen" : "" },
    el("a", { href: `#/results?project_dir=${encodeURIComponent(project.value)}&run_dir=${encodeURIComponent(item.run_dir)}`,
      text: item.name, title: item.run_dir }),
    el("span", { class: "small muted", text: item.written
      ? ` ${new Date(item.written * 1000).toLocaleString()}: ${item.files} ${item.files === 1 ? "target" : "targets"}, ${item.fully_measured_runs} fully measured ${item.fully_measured_runs === 1 ? "repetition" : "repetitions"}`
      : " not summarized yet" })))));
  return box;
}

async function resultsView(query) {
  const [state, listed] = await Promise.all([call("/api/state"),
    call("/api/projects").catch(() => ({ projects: [] }))]);
  const dir = query.get("run_dir") || "";
  const usable = (listed.projects || []).filter((entry) => !entry.error);
  const inProject = usable.find((entry) => dir && (dir + "/").replace(/\\/g, "/").startsWith(
    (entry.project_dir + "/").replace(/\\/g, "/")));
  const project = projectChoice({ projects: usable }, new URLSearchParams(
    `project_dir=${encodeURIComponent(query.get("project_dir") || (inProject ? inProject.project_dir : ""))}`));
  project.addEventListener("change", () => { location.hash = `#/results?project_dir=${encodeURIComponent(project.value)}`; });
  const input = el("input", { type: "text", value: dir, autocomplete: "off" });
  const form = el("form", { id: "results-form", novalidate: true },
    field({ id: "run_dir", label: "Run directory", input, browse: state.base_dir,
      hint: "A run's directory or a demonstration's directory." }),
    el("div", { class: "actions" }, el("button", { type: "submit", text: "Open" })));
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    location.hash = `#/results?run_dir=${encodeURIComponent(input.value.trim())}`;
  });
  const parts = [el("h1", { text: "Results" }),
    usable.length ? el("div", { class: "panel" }, field({ id: "project", label: "Project", input: project }))
      : noProjectNote(),
    await projectRuns(project, dir),
    el("details", { class: "panel", open: !usable.length || undefined }, el("summary", { text: "Open another run directory" }), form)];
  if (!dir) return parts;
  let summary;
  try { summary = await call(`/api/results?run_dir=${encodeURIComponent(dir)}`); }
  catch (error) {
    // Why the directory named in the address shows nothing, in plain view.
    form.closest("details").open = true;
    showError(form, { field: "run_dir", message: error.message });
    return parts;
  }
  if ((summary.views || []).length) {
    parts.push(el("p", { class: "panel" }, "This directory also opens as ",
      summary.views.map((item, index) => [index ? ", " : "", el("a", {
        href: dirTypeAddress(item.type, dir), text: item.title })]), "."));
  }
  parts.push(...runDirParts(summary, "", query.get("file")));
  return parts;
}

// A run directory's results below the Results page's own controls: a
// demonstration's note, each run, the log, every file and the file viewer,
// ``file`` opened in it. A nested run's directory shows them too, its ids kept
// apart from the page's by ``prefix``.
function runDirParts(summary, prefix, file) {
  const parts = [];
  const viewer = el("div", {});
  const open = (path) => fileView(viewer, summary.root, path);
  if (summary.demo && summary.demo_record) {
    const result = summary.demo_record.result || {};
    parts.push(el("div", { class: "panel notice" },
      el("p", {}, el("span", { class: "demo-badge", text: "demo" }), " ", summary.demo_record.statement || ""),
      el("div", { class: "rows" }, [
        ["Outcome", result.outcome], ["Answers", result.answers], ["Playback", result.playback],
        ["Processes started", (result.processes || []).length ? result.processes.join(", ") : "none"],
        ["Network connections", result.network_connections],
      ].flatMap(([name, value]) => [el("div", { class: "name", text: name }), el("div", {}, cell(value))]))));
  }
  summary.runs.forEach((run, index) => parts.push(runSection(run, summary, open, `${prefix}${index}`)));
  if (summary.log) parts.push(el("div", { class: "actions" }, el("button", { type: "button", text: "Show the run's log", onclick: () => open(summary.log) })));
  if (summary.files) parts.push(el("details", { class: "panel" }, el("summary", { text: `Every file (${summary.files.length})` }), fileList(summary.files, open)));
  parts.push(viewer);
  // A link can name a file to open: #/results?run_dir=...&file=...
  if (file) open(file);
  return parts;
}

// A Testuary run folded into a page (testuaryWeb.nestedRun). Opened, it
// shows an event file's run as a job's page does, following it while it is
// written, or a run directory's results as the Results page does. It reads
// only while open. Its summary names it a Testuary run beside the page's
// title, so the run's phases stay apart from the page's own.
let nestedRunCount = 0;
function nestedRun(path, title) {
  const box = el("div", {});
  const part = el("details", { class: "panel nested-run" },
    el("summary", { text: title ? `Testuary run: ${title}` : "Testuary run" }), box);
  nestedRunCount += 1;
  const prefix = `nested-${nestedRunCount}-`;
  let reading = null;
  let timer = null;
  let progress = null;
  let pending = false;
  // The job page says when the job that started the run has ended, so a
  // run that wrote no end of its own (a forced stop or a crash) ends too.
  let jobEnded = false;
  const draw = () => { pending = false; redrawKeeping(box, () => progress.render()); };
  const later = () => { if (!pending) { pending = true; requestAnimationFrame(draw); } };
  const stop = () => {
    if (reading) reading.abort();
    clearInterval(timer);
    reading = null;
  };
  part.addEventListener("jobended", () => {
    jobEnded = true;
    clearInterval(timer);
    if (progress && !progress.ended) { progress.ended = true; later(); }
  });
  part.addEventListener("toggle", () => {
    stop();
    if (!part.open) return;
    progress = new ViewPanel();
    progress.ended = jobEnded;
    progress.heading = "h3";
    reading = new AbortController();
    timer = setInterval(() => { if (progress.received && !progress.ended) tickClocks(box, progress.received); }, 1000);
    follow(`/api/nested-run?path=${encodeURIComponent(path)}`, {
      open: () => { progress.reset(); box.replaceChildren(el("p", { class: "muted", text: "Reading the run." })); },
      view: (text) => { progress.snapshot = JSON.parse(text); progress.received = Date.now(); later(); },
      log: (text) => { progress.addLine(JSON.parse(text)); later(); },
      summary: (text) => { progress.summary = JSON.parse(text); later(); },
      end: (text) => {
        const found = JSON.parse(text).results;
        clearInterval(timer);
        if (found) box.replaceChildren(...childList(runDirParts(found, prefix, "")));
        else { progress.ended = true; later(); }
      },
    }, reading.signal).catch((error) => {
      if (error.name !== "AbortError") box.replaceChildren(el("p", { class: "error", text: error.message }));
    });
  });
  leaving.push(stop);
  return part;
}

// The session's jobs, drawn anew at each look at the server that changes
// them, as the overview's recent jobs are: a status and time stay current.
async function jobsView() {
  const state = await call("/api/state");
  const box = el("div", {}, jobsTable(state.jobs));
  let before = state;
  onState((look) => {
    const changes = overviewChanges(before, look);
    before = look;
    if (changes.jobs) redraw(box, jobsTable(look.jobs));
  });
  return [el("h1", { text: "Jobs" }), box];
}

function jobsTable(jobs) {
  if (!jobs.length) {
    return el("p", { class: "muted" }, "No job started yet in this session. ", tabShown("run")
      ? [el("a", { href: "#/run", text: "Start a run" }), " or ", el("a", { href: "#/demo", text: "play the demonstration" })]
      : el("a", { href: "#/demo", text: "Play the demonstration" }), ".");
  }
  return el("div", { class: "panel scroll" }, stackedTable(["Started", "Job", "Run directory", "Model", "Values", "Status", "Time"],
    jobs.map((job) => {
      const { badge, label } = jobState(job);
      return el("tr", {},
        el("td", { class: "num", text: new Date(job.started * 1000).toLocaleTimeString() }),
        el("td", {}, el("a", { href: `#/job/${job.id}`, text: job.title })),
        el("td", {}, jobDir(job)),
        el("td", { text: jobModel(job) }),
        el("td", { class: "small" }, valueLines(job).map((line) => el("div", { text: line }))),
        el("td", {}, badge, " ", label),
        el("td", { class: "num", text: hms(job.elapsed_s) }));
    })));
}

// Web interface extensions

// What the extension modules add, as the server read them at
// its start (/api/state): the product name, added and renamed tabs, directory
// types and job kinds, and the extensions it refused, with why.
let extensions = { product: null, tabs: [], tab_titles: {}, tab_positions: {}, hidden_tabs: [], dir_types: [], job_kinds: [], refused: [] };
// Testuary's own tabs as index.html lists them, read once at the start:
// every header is built from them, never from a header built before, and a
// page drawn without access names them by their titles too.
const testuaryTabs = [...document.querySelectorAll(".top nav a[data-view]")].map((link) => ({ id: link.dataset.view, title: link.textContent }));
// Testuary's tabs' own titles by id.
const testuaryTitles = new Map(testuaryTabs.map((tab) => [tab.id, tab.title]));
// The views the extensions' scripts register, and each script's loading.
const extensionViews = new Map();
const extensionScripts = new Map();

// What an extension's view script builds its view with.
window.testuaryWeb = Object.freeze({
  registerView(id, viewFunction) { extensionViews.set(id, viewFunction); },
  el,
  call,
  panel: (...children) => el("div", { class: "panel" }, ...children),
  table: (headings, rows) => el("div", { class: "panel scroll" }, el("table", {},
    el("thead", {}, el("tr", {}, headings.map((text) => el("th", { scope: "col", text })))),
    el("tbody", {}, rows.map((row) => el("tr", {}, row.map((value) => el("td", {}, value))))))),
  tabTitle,
  nestedRun,
  jobLink: (job) => el("a", { href: `#/job/${job.id}`, text: job.title }),
  resultsLink: (dir, text = `Open the ${tabTitleInText("results")}`) =>
    tabShown("results") ? el("a", { href: `#/results?run_dir=${encodeURIComponent(dir)}`, text }) : null,
  pathText,
});

// The header's tabs: Testuary's, with the extensions' titles, and the moved
// and added tabs at their positions. Each pass places every tab whose
// position names no tab or a tab the header shows, so a position may name
// a moved or an added tab, and the tabs no pass can place go last. A
// replaced tab leaves the header once every tab has its place.
function headerTabs(testuaryTabs, added) {
  const titled = testuaryTabs.map((tab) => ({ id: tab.id, title: added.tab_titles[tab.id] || tab.title }));
  const tabs = titled.filter((tab) => !(tab.id in added.tab_positions));
  const moved = titled.filter((tab) => tab.id in added.tab_positions)
    .map((tab) => ({ ...tab, position: added.tab_positions[tab.id] }));
  let pending = [...moved, ...added.tabs];
  while (pending.length) {
    const left = pending.filter((tab) => !placeTab(tabs, tab));
    if (left.length === pending.length) {
      tabs.push(...left);
      break;
    }
    pending = left;
  }
  const gone = new Set([...added.tabs.map((tab) => tab.replaces), ...added.hidden_tabs]);
  return tabs.filter((tab) => !gone.has(tab.id));
}

// Places a tab at its position, unless the position names a tab the header
// does not show yet, and says whether it did.
function placeTab(tabs, tab) {
  const [where, other] = tab.position.split(":");
  const at = tabs.findIndex((item) => item.id === other);
  if (where === "first") tabs.unshift(tab);
  else if (where !== "before" && where !== "after") tabs.push(tab);
  else if (at < 0) return false;
  else tabs.splice(where === "before" ? at : at + 1, 0, tab);
  return true;
}

// The added tab an address opens: the tab of that id, or the one that
// replaces the Testuary tab of that address.
function addedTab(name) {
  return extensions.tabs.find((item) => item.id === name || item.replaces === name);
}

// A tab's title as the header shows it: the added tab's for its address, a
// new title or Testuary's own.
function tabTitle(id) {
  const added = addedTab(id);
  return added ? added.title : extensions.tab_titles[id] || testuaryTitles.get(id) || id;
}

// A tab's title inside a sentence, such as Open test results: its first
// letter in lower case, unless the title starts with an abbreviation.
function tabTitleInText(id) {
  const title = tabTitle(id);
  return /^[A-Z]{2}/.test(title) ? title : title.charAt(0).toLowerCase() + title.slice(1);
}

// Whether the header shows a Testuary tab. A link to a hidden tab is left
// out with the words that belong to it, so the rest reads as whole sentences.
function tabShown(id) {
  return !extensions.hidden_tabs.includes(id);
}

// A heading that names the whole product: the product's name, else Testuary.
const productTitle = () => extensions.product || "Testuary";

// Where the values come from: the product's name, else Testuary, or
// Testuary with its extensions where values come from both.
function productName() {
  const added = extensions.tabs.length + extensions.dir_types.length + extensions.job_kinds.length
    + Object.keys(extensions.tab_titles).length;
  return extensions.product || (added ? "Testuary and its extensions" : "Testuary");
}

// A progress panel's title ("Testuary", "Testuary | demo playback"): the
// product's name, else an extension job's kind.
function panelTitle(title, kind) {
  const name = extensions.product || (kind && kind.title);
  return name ? title.replace(/^Testuary/, name) : title;
}

// The header: the product name, and Testuary's tabs as index.html lists
// them with the added ones. The footer names where the values come from.
function buildHeader() {
  if (extensions.product) {
    document.querySelector(".top .brand").textContent = extensions.product;
    document.title = extensions.product;
  }
  document.querySelector(".foot").textContent =
    `Local only, on 127.0.0.1. Every value shown comes from ${productName()}.`;
  document.querySelector(".top nav").replaceChildren(...headerTabs(testuaryTabs, extensions).map((tab) =>
    el("a", { href: `#/${tab.id}`, "data-view": tab.id, text: tab.title })));
  markCurrentTab();
}

// The view the address opens: its name, its argument, its query and the
// extension's tab that shows it. A hidden tab's address opens the
// Overview, as an unknown address does.
function addressedView() {
  const [path, search] = location.hash.replace(/^#\/?/, "").split("?");
  const [address, arg] = path.split("/");
  const name = tabShown(address) ? address : "";
  return { name, arg, query: new URLSearchParams(search || ""), tab: addedTab(name) };
}

// The header marks the tab of the addressed view as the current page.
function markCurrentTab() {
  const { name, tab } = addressedView();
  const current = tab ? tab.id : name === "job" ? "jobs" : name;
  for (const link of document.querySelectorAll(".top a[data-view]")) {
    if (link.dataset.view === current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

// The label of Testuary's tool check option, which the server defines once
// (testuary.web.TOOL_CHECK_LABEL) and its check report quotes.
let toolCheckLabel = "";

async function loadExtensions() {
  try {
    const state = await call("/api/state");
    extensions = state.extensions || extensions;
    toolCheckLabel = state.tool_check_label;
  } catch (error) { return; }
  buildHeader();
}

// An extension's view: its script, loaded once, registers it. A script or
// view that fails shows its error here, and the other tabs work.
async function extensionView(script, id, query) {
  try {
    if (!extensionScripts.has(script)) extensionScripts.set(script, import(script));
    await extensionScripts.get(script);
    const show = extensionViews.get(id);
    if (!show) throw new Error(`The script registered no view ${id}.`);
    return await show(query);
  } catch (error) {
    return [el("h1", { text: "This view failed" }),
      el("p", { class: "panel error", text: `Its extension's script failed: ${error.message}` })];
  }
}

// A directory an extension's directory type opens: #/dir?type=EXTENSION/TYPE&dir=DIR.
async function dirTypeView(query) {
  const type = extensions.dir_types.find((item) => item.id === query.get("type"));
  if (!type) return [el("h1", { text: "Results" }), el("p", { class: "panel", text: "No such directory type." })];
  return extensionView(type.script, type.id.split("/")[1], query);
}

function refusedExtensions() {
  if (!extensions.refused.length) return null;
  return el("div", { class: "panel notice" }, el("h2", { text: "Extensions not loaded" }),
    el("ul", {}, extensions.refused.map((item) => el("li", { text: `${item.extension}: ${item.reason}.` }))));
}

// Routing

async function render() {
  for (const stop of leaving) stop();
  leaving = [];
  showMessage("");
  const { name, arg, query, tab } = addressedView();
  markCurrentTab();
  try {
    // Without access (no key, or one of an earlier start, where the
    // interface has one) a view would turn each refused request into a part
    // of its own: the server's refusal says why, once.
    await call("/api/state");
    const parts = tab ? await extensionView(tab.script, tab.id, query)
      : name === "dir" ? await dirTypeView(query)
      : name === "demo" ? await demoView() : name === "check" ? await checkView(query)
      : name === "projects" ? await projectsView() : name === "models" ? await modelsView(query)
      : name === "settings" ? await settingsView(query)
      : name === "run" ? await runView(query)
      : name === "results" ? await resultsView(query) : name === "jobs" ? await jobsView()
        : name === "job" && arg ? await jobView(arg) : await home();
    // A view returns nodes, lists of nodes, and null for a part it leaves out.
    view.replaceChildren(...childList(parts));
    // A renamed tab renames its page's heading too.
    const heading = extensions.tab_titles[name] && view.querySelector("h1");
    if (heading && heading.textContent === testuaryTitles.get(name)) heading.textContent = tabTitle(name);
  } catch (error) {
    view.replaceChildren(el("h1", { text: productTitle() }), el("p", { class: "panel", text: error.message }));
  }
  view.focus();
}

async function watchActive() {
  const link = document.getElementById("active");
  try {
    const state = await call("/api/state");
    lastActive = state.active;
    // The guide of the installed version, in a tab of its own.
    const guide = document.getElementById("guide");
    if (guide && state.guide) { guide.href = state.guide; guide.hidden = false; }
    if (state.active) {
      link.hidden = false;
      link.replaceChildren(jobState(state.active).badge, state.active.title);
      link.href = `#/job/${state.active.id}`;
    } else link.hidden = true;
    for (const listener of activeListeners) listener(state.active);
    for (const listener of stateListeners) listener(state);
  } catch (error) { link.hidden = true; }
}

window.addEventListener("hashchange", render);
loadExtensions().finally(render);
watchActive();
setInterval(watchActive, 3000);
