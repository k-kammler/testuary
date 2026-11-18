// The Notes tab and the notes directory type's view: a form that starts the
// Collect notes, Check notes and Play the demonstration jobs, and a notes
// directory's summary as the job wrote it, with its demonstration folded.
const web = globalThis.testuaryWeb;

web.registerView("notes", async (query) => {
  const dir = query.get("dir");
  const parts = [web.el("h1", { text: "Notes" })];
  const input = web.el("input", { type: "text", id: "notes-dir", value: dir || "" });
  // A button that starts a job of the kind for the directory and opens its page.
  const starter = (kind, text, style) => {
    const button = web.el("button", { type: "button", class: style, text });
    button.addEventListener("click", async () => {
      const job = await web.call("/api/ext/jobs", { kind, dir: input.value });
      location.hash = `#/job/${job.id}`;
    });
    return button;
  };
  parts.push(web.panel(web.el("label", { for: "notes-dir", text: "Notes directory" }), input,
    starter("notes/collect", "Collect notes", "primary"), starter("notes/check", "Check the directory", ""),
    starter("notes/demo", "Play the demonstration", "")));
  if (dir) {
    const found = await web.call(`/api/ext/dir?type=notes/notes&dir=${encodeURIComponent(dir)}`);
    parts.push(web.el("h2", { text: `${found.summary.count} notes` }),
      web.table(["Note"], found.summary.notes.map((note) => [note])),
      web.panel(web.resultsLink(dir, `Open this directory in ${web.tabTitle("results")}`)));
    // The demonstration Play the demonstration played, drawn by Testuary.
    if (found.summary.demo) parts.push(web.nestedRun(`${dir}/${found.summary.demo}`, "Testuary's demonstration"));
  }
  return parts;
});
