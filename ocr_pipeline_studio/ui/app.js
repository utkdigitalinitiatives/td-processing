/* OCR Pipeline Studio -- front-end.
 *
 * Plain ES modules-free JavaScript, no framework. The app has two screens and
 * one polling loop, which is well under the size where a framework starts
 * paying for itself.
 *
 * The one external dependency is markdown-it, vendored into ui/vendor/. It is
 * never loaded from a CDN, because the whole app has to work with no internet
 * connection.
 */

(function () {
  "use strict";

  // html: true is essential here rather than a stylistic choice. The pipeline
  // emits an HTML fragment -- <p>, <sup>, <sub> and character entities like
  // &alpha; -- and that markup is the actual product of the OCR work. With
  // html:false markdown-it would escape it all and the preview would show raw
  // tags instead of formatted text.
  var md = window.markdownit({ html: true, linkify: false, breaks: false });

  var state = {
    pending: [],        // {file, pages} staged in the drop zone
    jobId: null,
    fixesOnly: false,   // whether the job on screen was a page-fixes-only run
    poll: null,         // setInterval handle for /status
    documents: [],      // summaries from /status
    current: null,      // full payload for the open document
    dirty: {},          // name -> true when edited but unsaved
    editTimer: null,
    lastSkip: null,     // a single-span click that could not be carried out
    choiceOpen: {}      // "page:index" -> open/closed, for copy pickers the user toggled
  };

  // ---------- tiny DOM helpers ----------
  function $(id) { return document.getElementById(id); }
  function on(el, evt, fn) { el.addEventListener(evt, fn); }
  function show(el, visible) { el.hidden = !visible; }

  function text(value) {
    // Everything user- or pipeline-supplied goes through here before being
    // put on the page, so a stray "<" in OCR output can never become markup.
    var node = document.createElement("span");
    node.textContent = value == null ? "" : String(value);
    return node;
  }

  function api(path, options) {
    return fetch(path, options).then(function (resp) {
      return resp.json().then(function (body) {
        if (!resp.ok) { throw new Error(body.error || ("Request failed: " + resp.status)); }
        return body;
      });
    });
  }

  // =====================================================================
  // Ollama banner + model picker
  // =====================================================================

  function refreshOllama() {
    return api("/ollama").then(function (status) {
      var banner = $("ollama-banner");
      banner.className = "banner " + (status.running ? "ok" : "bad");
      banner.textContent = status.running
        ? "Ollama is running - " + status.models.length + " model(s) available locally."
        : status.error;
      show(banner, true);
      return status;
    });
  }

  function loadModels() {
    return api("/models").then(function (data) {
      var select = $("model-select");
      select.innerHTML = "";

      if (!data.models.length) {
        var none = document.createElement("option");
        none.textContent = "No models found - run: ollama pull qwen2.5vl:3b";
        none.value = "";
        select.appendChild(none);
      }
      data.models.forEach(function (name) {
        var option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        // Preselect the model the OCR script itself defaults to, when present.
        if (name === data.default_model) { option.selected = true; }
        select.appendChild(option);
      });

      var modeSelect = $("mode-select");
      modeSelect.innerHTML = "";
      data.modes.forEach(function (mode) {
        var option = document.createElement("option");
        option.value = mode.id;
        option.textContent = mode.label;
        option.dataset.help = mode.help;
        if (mode.id === data.default_mode) { option.selected = true; }
        modeSelect.appendChild(option);
      });
      updateModeHelp();
      return data;
    });
  }

  function updateModeHelp() {
    var selected = $("mode-select").selectedOptions[0];
    $("mode-help").textContent = selected ? (selected.dataset.help || "") : "";
  }

  // =====================================================================
  // Drop zone
  // =====================================================================

  function setupDropzone() {
    var zone = $("dropzone");
    var input = $("file-input");

    // dragover must be cancelled or the browser opens the file instead.
    ["dragenter", "dragover"].forEach(function (evt) {
      on(zone, evt, function (e) {
        e.preventDefault();
        zone.classList.add("is-over");
      });
    });
    ["dragleave", "drop"].forEach(function (evt) {
      on(zone, evt, function (e) {
        e.preventDefault();
        zone.classList.remove("is-over");
      });
    });

    on(zone, "drop", function (e) {
      addFiles(e.dataTransfer.files);
    });
    on($("browse-btn"), "click", function () { input.click(); });
    on(input, "change", function () { addFiles(input.files); input.value = ""; });
    on($("clear-btn"), "click", function () { state.pending = []; renderPending(); });
  }

  function addFiles(fileList) {
    Array.prototype.forEach.call(fileList, function (file) {
      if (!/\.pdf$/i.test(file.name)) { return; }
      // Skip a file already staged, so dropping the same batch twice does not
      // upload it twice.
      var already = state.pending.some(function (entry) {
        return entry.file.name === file.name && entry.file.size === file.size;
      });
      if (!already) { state.pending.push({ file: file, pages: "" }); }
    });
    renderPending();
  }

  function renderPending() {
    var list = $("pending-list");
    list.innerHTML = "";
    var fixesOnly = isFixesOnly();
    show($("pages-help"), state.pending.length > 0 && !fixesOnly);

    state.pending.forEach(function (entry, index) {
      var li = document.createElement("li");
      var name = text(entry.file.name);
      name.className = "file-name";
      li.appendChild(name);

      // Abstract pages set by hand. Not offered for a page-fixes-only run,
      // which never looks for an abstract.
      if (!fixesOnly) {
        var pages = pagesInput(entry.pages);
        on(pages, "input", function () { entry.pages = pages.value; });
        li.appendChild(pages);
      }

      var remove = document.createElement("button");
      remove.className = "linkish";
      remove.textContent = "remove";
      on(remove, "click", function () {
        state.pending.splice(index, 1);
        renderPending();
      });
      li.appendChild(remove);
      list.appendChild(li);
    });
    $("run-btn").disabled = state.pending.length === 0;
    $("clear-btn").disabled = state.pending.length === 0;
  }

  // =====================================================================
  // Running the pipeline
  // =====================================================================

  function isFixesOnly() {
    return $("run-kind").value === "fixes";
  }

  // Same forms the server accepts: "5-8", "5", or blank for automatic.
  function validPages(value) {
    if (!(value || "").trim()) { return true; }
    var m = /^\s*(\d+)\s*(?:-\s*(\d+))?\s*$/.exec(value);
    return !!m && +m[1] >= 1 && +(m[2] || m[1]) >= +m[1];
  }

  function pagesInput(value) {
    var input = document.createElement("input");
    input.type = "text";
    input.className = "pages-input";
    input.placeholder = "abstract pages: auto";
    input.title = "Abstract pages as numbered in this PDF, e.g. 5-8. "
      + "Blank finds them automatically.";
    input.value = value || "";
    function check() { input.classList.toggle("is-invalid", !validPages(input.value)); }
    on(input, "input", check);
    check();
    return input;
  }

  function updateRunKind() {
    var fixesOnly = isFixesOnly();
    // The model and VLM pass only matter to the OCR step a fixes-only run skips.
    $("model-select").disabled = fixesOnly;
    $("mode-select").disabled = fixesOnly;
    show($("mode-help"), !fixesOnly);
    show($("kind-help"), fixesOnly);
    renderPending();
  }

  function showRunError(message) {
    var box = $("run-error");
    box.textContent = message;
    show(box, true);
  }

  function badPagesMessage(names) {
    return "Check the abstract pages for " + names.join(", ") + " - use a form like 5-8 or 5.";
  }

  function startRun() {
    var fixesOnly = isFixesOnly();
    var overrides = {};
    var bad = [];
    state.pending.forEach(function (entry) {
      if (fixesOnly || !entry.pages.trim()) { return; }
      if (!validPages(entry.pages)) { bad.push(entry.file.name); }
      overrides[entry.file.name] = entry.pages.trim();
    });
    if (bad.length) {
      showRunError(badPagesMessage(bad));
      return;
    }

    var form = new FormData();
    state.pending.forEach(function (entry) { form.append("files", entry.file); });
    form.append("model", $("model-select").value);
    form.append("mode", $("mode-select").value);
    form.append("fixes_only", fixesOnly ? "1" : "0");
    form.append("overrides", JSON.stringify(overrides));

    $("run-btn").disabled = true;
    show($("run-error"), false);

    api("/upload", { method: "POST", body: form })
      .then(function (data) {
        state.pending = [];
        renderPending();
        beginJob(data.job_id);
      })
      .catch(function (err) {
        // The most common failure here is the Ollama preflight, whose message
        // is already written for a human -- show it as-is.
        var box = $("run-error");
        box.textContent = err.message;
        show(box, true);
        $("run-btn").disabled = state.pending.length === 0;
      });
  }

  // Point the screen at a newly started job. The review screen showed the
  // previous job, so it is closed until this one finishes.
  function beginJob(jobId) {
    state.jobId = jobId;
    state.documents = [];
    state.current = null;
    state.dirty = {};
    $("tab-review").disabled = true;
    show($("progress-panel"), true);
    switchScreen("run");
    startPolling();
  }

  function startPolling() {
    if (state.poll) { clearInterval(state.poll); }
    tick();
    // 1.2s: fast enough to feel live on a per-page pass, slow enough that it
    // costs nothing next to a multi-minute OCR run.
    state.poll = setInterval(tick, 1200);
  }

  function tick() {
    if (!state.jobId) { return; }
    api("/status/" + state.jobId)
      .then(renderProgress)
      .catch(function () { /* a dropped poll is not worth interrupting for */ });
  }

  function renderProgress(status) {
    $("progress-label").textContent = status.label;
    // The full path, so the folder can be found even without the button.
    $("job-folder").textContent = "Working folder: " + (status.folder || "");
    show($("job-folder"), !!status.folder);

    var count = "";
    if (status.file_total) {
      count = "file " + Math.max(status.file_index, 1) + " of " + status.file_total;
    }
    $("progress-count").textContent = count;

    // Progress is approximated from pages within the current file plus how
    // many files are done. There is no honest total up front -- the OCR script
    // only reveals a document's abstract page range once it starts reading it.
    var fraction = 0;
    if (status.file_total) {
      var done = Math.max(status.file_index - 1, 0) / status.file_total;
      var within = status.page_total
        ? (status.page_current / status.page_total) / status.file_total
        : 0;
      fraction = Math.min(done + within, 0.98);
    }
    if (status.finished) { fraction = 1; }
    $("progress-fill").style.width = (fraction * 100).toFixed(1) + "%";

    renderQueue(status);
    $("log-output").textContent = (status.log || []).join("\n");

    if (status.finished) {
      clearInterval(state.poll);
      state.poll = null;
      $("run-btn").disabled = state.pending.length === 0;

      if (status.status === "error") {
        var box = $("run-error");
        box.textContent = status.error || "The pipeline failed.";
        show(box, true);
        return;
      }
      state.documents = status.documents || [];
      state.fixesOnly = !!status.fixes_only;
      // A fixes-only job has no text to save; its product is the PDFs, which
      // are already on disk.
      show($("save-btn"), !state.fixesOnly);
      $("save-status").textContent = state.fixesOnly
        ? "Corrected PDFs are in the fixed folder - use Open folder."
        : "";
      if (state.documents.length) {
        $("tab-review").disabled = false;
        renderDocList();
        switchScreen("review");
        selectDocument(state.documents[0].name);
      }
    }
  }

  function renderQueue(status) {
    var list = $("queue-list");
    list.innerHTML = "";
    (status.files || []).forEach(function (name) {
      var info = (status.file_states || {})[name] || { state: "queued", detail: "" };
      var li = document.createElement("li");
      li.appendChild(text(name));
      var span = document.createElement("span");
      span.className = "state " + info.state;
      span.textContent = info.detail ? info.state + " - " + info.detail : info.state;
      li.appendChild(span);
      // Once the job is over, any file can be run again: with its abstract
      // pages set by hand, or as page fixes only when it has no abstract.
      if (status.finished && status.status === "done") {
        li.appendChild(buildRerun(status.job_id, name));
      }
      list.appendChild(li);
    });
  }

  function buildRerun(jobId, name) {
    var row = document.createElement("div");
    row.className = "rerun";

    var pages = pagesInput("");
    row.appendChild(pages);

    var again = document.createElement("button");
    again.className = "ghost";
    again.textContent = "Rerun abstract";
    again.title = "Run the full pipeline on this file again, using the pages typed here.";
    on(again, "click", function () {
      if (!validPages(pages.value)) {
        showRunError(badPagesMessage([name]));
        return;
      }
      var overrides = {};
      if (pages.value.trim()) { overrides[name] = pages.value.trim(); }
      rerunFiles(jobId, [name], false, overrides);
    });
    row.appendChild(again);

    var fixes = document.createElement("button");
    fixes.className = "ghost";
    fixes.textContent = "Page fixes only";
    fixes.title = "Remove repeats and turn sideways pages, then download the "
      + "corrected PDF. No OCR.";
    on(fixes, "click", function () { rerunFiles(jobId, [name], true, {}); });
    row.appendChild(fixes);

    return row;
  }

  function rerunFiles(jobId, files, fixesOnly, overrides) {
    var unsaved = Object.keys(state.dirty).length;
    if (unsaved && !window.confirm("You have unsaved edits in " + unsaved
        + " document(s). Rerunning moves on to a new run - continue?")) {
      return;
    }
    show($("run-error"), false);
    api("/rerun/" + jobId, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        files: files,
        model: $("model-select").value,
        mode: $("mode-select").value,
        fixes_only: fixesOnly,
        overrides: overrides
      })
    }).then(function (data) {
      syncRunControls(data);
      beginJob(data.job_id);
    }).catch(function (err) {
      showRunError(err.message);
    });
  }

  // Make the Run screen's dropdowns show what a rerun actually started with,
  // so they never describe a different kind of run from the one in progress.
  // Only done once the server has accepted the rerun: a refused one changes
  // nothing.
  function syncRunControls(run) {
    $("run-kind").value = run.fixes_only ? "fixes" : "full";
    selectIfPresent($("model-select"), run.model);
    selectIfPresent($("mode-select"), run.mode);
    updateModeHelp();
    updateRunKind();
  }

  function selectIfPresent(select, value) {
    // Setting a value the list does not have would blank the dropdown.
    var has = Array.prototype.some.call(select.options, function (option) {
      return option.value === value;
    });
    if (has) { select.value = value; }
  }

  // =====================================================================
  // Review screen
  // =====================================================================

  function switchScreen(name) {
    show($("screen-run"), name === "run");
    show($("screen-review"), name === "review");
    document.querySelectorAll(".topbar .tab").forEach(function (tab) {
      tab.classList.toggle("is-active", tab.dataset.screen === name);
    });
  }

  function renderDocList() {
    var list = $("doc-list");
    list.innerHTML = "";
    $("doc-count").textContent = state.documents.length + " total";

    state.documents.forEach(function (doc) {
      var li = document.createElement("li");
      var button = document.createElement("button");
      button.dataset.name = doc.name;

      var name = document.createElement("span");
      name.className = "doc-name";
      name.textContent = doc.name;
      button.appendChild(name);

      var chips = document.createElement("span");
      chips.className = "doc-flags";

      // Flag chips come straight from the pipeline's own per-page reasons.
      // Pages the script flagged as low-confidence are shown first, since
      // those are the ones worth looking at before anything else.
      var pagesByKind = {};
      (doc.flags || []).forEach(function (flag) {
        (pagesByKind[flag.kind] = pagesByKind[flag.kind] || []).push(flag);
      });
      // Page fixes come first: they happened before OCR, to the whole PDF.
      if (doc.duplicates_removed) {
        chips.appendChild(fixChip("dedupe", doc.duplicates_removed + " repeat(s) removed",
          "Repeated pages taken out before OCR - see the Page fixes tab."));
      }
      if (doc.pages_rotated) {
        chips.appendChild(fixChip("rotate", doc.pages_rotated + " rotated",
          "Sideways pages turned upright before OCR - see the Page fixes tab."));
      }
      if (doc.pages_for_review) {
        chips.appendChild(fixChip("rotate-review", doc.pages_for_review + " rotation check",
          "Pages that may be sideways but were left alone - check them on the Page fixes tab."));
      }

      ["low_confidence", "equation", "diff"].forEach(function (kind) {
        var flags = pagesByKind[kind];
        if (!flags) { return; }
        var chip = document.createElement("span");
        chip.className = "chip " + kind;
        // The chip names the actual pages, not just a count, so the sidebar
        // answers "where do I look first?" without a click.
        var pages = flags.map(function (f) { return f.page; });
        chip.textContent = kindLabel(kind) + " p" + pages.join(",");
        // Hovering gives the pipeline's own wording for why each was flagged.
        chip.title = flags.map(function (f) {
          return "Page " + f.page + ": " + f.labels.join("; ");
        }).join("\n");
        chips.appendChild(chip);
      });
      if (state.dirty[doc.name]) {
        var dirty = document.createElement("span");
        dirty.className = "chip dirty";
        dirty.textContent = "unsaved";
        chips.appendChild(dirty);
      }
      button.appendChild(chips);

      if (state.current && state.current.name === doc.name) {
        button.classList.add("is-active");
      }
      on(button, "click", function () { selectDocument(doc.name); });
      li.appendChild(button);
      list.appendChild(li);
    });
  }

  function fixChip(kind, label, title) {
    var chip = document.createElement("span");
    chip.className = "chip " + kind;
    chip.textContent = label;
    chip.title = title;
    return chip;
  }

  function kindLabel(kind) {
    if (kind === "low_confidence") { return "low-conf"; }
    if (kind === "equation") { return "equation"; }
    return "diff";
  }

  function selectDocument(name) {
    api("/document/" + state.jobId + "/" + encodeURIComponent(name))
      .then(function (doc) {
        state.current = doc;
        state.choiceOpen = {};
        $("doc-title").textContent = doc.name;

        var meta = [];
        if (doc.fixes_only) { meta.push("page fixes only"); }
        if (doc.abstract_pages) {
          meta.push("abstract pages " + doc.abstract_pages.requested + " set by hand");
        }
        if (doc.page_count != null) { meta.push(doc.page_count + " pages after fixes"); }
        if (doc.duplicates_removed) { meta.push(doc.duplicates_removed + " repeated page(s) removed"); }
        if (doc.pages_rotated) { meta.push(doc.pages_rotated + " sideways page(s) rotated"); }
        if (doc.model_used) { meta.push("model: " + doc.model_used); }
        $("doc-meta").textContent = meta.join(" - ");

        $("editor").value = doc.text;
        renderPreview();

        // Tabs are disabled rather than hidden when a mode produced no such
        // data, so the UI never shows a control that silently does nothing.
        $("tab-edit").disabled = !!doc.fixes_only;
        $("tab-diff").disabled = !(doc.diffs && doc.diffs.length);
        $("tab-recovery").disabled = !(doc.recovery && doc.recovery.length);
        // Always open for a fixes-only document: the fixes are all it has,
        // and "none were needed" is itself an answer.
        $("tab-fixes").disabled = !(doc.fixes_only
          || (doc.duplicates && doc.duplicates.length)
          || (doc.rotations && doc.rotations.length));

        renderDiff(doc);
        renderRecovery(doc);
        renderFixes(doc);
        renderDocList();
        switchView(doc.fixes_only ? "fixes" : "edit");
      });
  }

  function renderPreview() {
    // The preview is markdown-it output of content the pipeline produced and
    // the user edits locally. Nothing here crosses a trust boundary: it is a
    // single-user local tool rendering that same user's own document.
    $("preview").innerHTML = md.render($("editor").value || "");
  }

  function switchView(view) {
    show($("view-edit"), view === "edit");
    show($("view-diff"), view === "diff");
    show($("view-recovery"), view === "recovery");
    show($("view-fixes"), view === "fixes");
    document.querySelectorAll(".view-toggle .tab").forEach(function (tab) {
      tab.classList.toggle("is-active", tab.dataset.view === view);
    });
  }

  function renderDiff(doc) {
    var body = $("diff-body");
    body.innerHTML = "";

    if (!doc.diffs || !doc.diffs.length) {
      $("diff-intro").textContent = "";
      var empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "This run produced no OCR/VLM comparison. "
        + "Choose one of the diff modes on the Run screen to generate one.";
      body.appendChild(empty);
      return;
    }

    $("diff-intro").textContent =
      "Left is what PaddleOCR read; right is what the vision model read on the "
      + "same page. Click a side to put it into the document - you do not have "
      + "to find the text in the editor yourself.";

    // Bulk actions. These sweep every span that is not already on the chosen
    // side, which is the whole point: the flagged ones are scattered through
    // the document and hunting for each by hand is the tedious part.
    var bulk = document.createElement("div");
    bulk.className = "bulk-actions";

    var remaining = countOnSide(doc.diffs, "ocr");
    var applyAll = document.createElement("button");
    applyAll.className = "primary";
    applyAll.textContent = "Apply all remaining VLM fixes (" + remaining + ")";
    applyAll.disabled = remaining === 0;
    on(applyAll, "click", function () { applyDiff({ all: true, direction: "vlm" }); });
    bulk.appendChild(applyAll);

    var revertAll = document.createElement("button");
    revertAll.className = "ghost";
    revertAll.textContent = "Revert all to OCR";
    on(revertAll, "click", function () { applyDiff({ all: true, direction: "ocr" }); });
    bulk.appendChild(revertAll);

    var note = document.createElement("span");
    note.className = "muted small";
    note.id = "diff-status";
    bulk.appendChild(note);
    body.appendChild(bulk);

    doc.diffs.forEach(function (page) {
      var section = document.createElement("section");
      section.className = "diff-page";

      var header = document.createElement("header");
      header.textContent = "Page " + page.page + " - " + page.spans.length + " difference(s)";
      section.appendChild(header);

      page.spans.forEach(function (span, index) {
        section.appendChild(buildDiffRow(page.page, index, span));
      });
      body.appendChild(section);
    });
  }

  function countOnSide(diffs, side) {
    var total = 0;
    (diffs || []).forEach(function (page) {
      page.spans.forEach(function (span) {
        if (span.state === side) { total += 1; }
      });
    });
    return total;
  }

  function buildDiffRow(pageNo, index, span) {
    var row = document.createElement("div");
    var label = (span.label || "DIFF").toLowerCase().replace(/\s+/g, "-");
    row.className = "diff-row " + label + " state-" + (span.state || "unclear");

    var labelCell = document.createElement("div");
    labelCell.className = "label";
    labelCell.textContent = span.label || "diff";

    // "unclear" means neither side was found verbatim in the document -- the
    // script's own text fixups rewrote this passage after the diff was taken
    // (rejoining a hyphenated line break, say). Saying so up front is better
    // than letting the user click a button that can only report failure.
    if (span.state === "unclear") {
      var mark = document.createElement("span");
      mark.className = "unclear-mark";
      mark.textContent = "manual";
      mark.title = "Neither reading appears verbatim in the document, so this "
        + "one cannot be applied automatically. Edit it on the Edit tab.";
      labelCell.appendChild(document.createElement("br"));
      labelCell.appendChild(mark);
    }
    row.appendChild(labelCell);

    // If the last click on this exact row could not be carried out, say so
    // on the row itself. The summary line lives at the top of a long list,
    // so on its own it is easy to miss and the click looks like it did
    // nothing at all.
    var skip = state.lastSkip;
    if (skip && skip.page === pageNo && skip.index === index) {
      row.classList.add("did-nothing");
      var why = document.createElement("div");
      why.className = "row-note";
      why.textContent = "Not changed - " + describeSkips([skip]) + ".";
      row.appendChild(why);
    }

    // Each side is a button. Clicking it puts that reading into the document,
    // so "merge this one" and "put it back" are the same single gesture.
    row.appendChild(buildSideButton(pageNo, index, span, "ocr"));
    row.appendChild(buildSideButton(pageNo, index, span, "vlm"));

    if (span.occurrences) {
      row.classList.add("needs-choice");
      row.appendChild(buildChoices(pageNo, index, span));
    }

    return row;
  }

  // The words this difference is about appear more than once, and nothing
  // says which copy is its own. Rather than guess -- and quietly change the
  // wrong sentence -- each copy is listed in its surrounding words for the
  // user to pick. The list is a dropdown that stays on the row: open until a
  // copy is picked, then collapsed to show which one, so a wrong pick can be
  // reopened and moved to the right copy.
  function buildChoices(pageNo, index, span) {
    var picked = span.selected != null;

    // Every apply redraws all the rows, so a picker keeps whatever the user
    // last did with it -- one they folded away stays folded while they work
    // on other rows. Untouched, it is open until a copy is picked.
    var key = pageNo + ":" + index;
    var box = document.createElement("details");
    box.className = "choices" + (picked ? " is-picked" : "");
    box.open = key in state.choiceOpen ? state.choiceOpen[key] : !picked;
    on(box, "toggle", function () { state.choiceOpen[key] = box.open; });

    var summary = document.createElement("summary");
    if (picked) {
      var chosen = span.occurrences[span.selected];
      summary.appendChild(text("Changed copy " + (span.selected + 1) + " of "
        + span.occurrences.length + ": "));
      // Shown with the reading the document now has at that copy.
      summary.appendChild(copyInContext(chosen, span[span.state]));
      var change = document.createElement("span");
      change.className = "choices-change";
      change.textContent = "change";
      summary.appendChild(change);
    } else {
      summary.textContent = "This appears " + span.occurrences.length
        + " times - pick which one to change";
    }
    box.appendChild(summary);

    var list = document.createElement("div");
    list.className = "choices-list";
    span.occurrences.forEach(function (occurrence, i) {
      var isSelected = i === span.selected;
      var button = document.createElement("button");
      button.className = "choice"
        + (isSelected ? " is-selected" : "")
        + (i === span.suggested ? " is-suggested" : "");
      button.title = isSelected ? "This is the copy currently changed."
        : picked ? "Move the change to this copy instead."
        : "Change this copy only.";
      button.appendChild(copyInContext(occurrence, occurrence.match));

      var tagText = isSelected ? "current" : (i === span.suggested ? "likely" : "");
      if (tagText) {
        var tag = document.createElement("span");
        tag.className = "likely";
        tag.textContent = tagText;
        button.appendChild(tag);
      }

      on(button, "click", function () {
        if (isSelected) { box.open = false; return; }
        // Picking folds this picker away on the redraw that follows.
        state.choiceOpen[key] = false;
        applyDiff({ page: pageNo, index: index, direction: span.choose_direction,
                    at_offset: occurrence.offset });
      });
      list.appendChild(button);
    });
    box.appendChild(list);
    return box;
  }

  // "...words before [match] words after..." with the match in bold.
  function copyInContext(occurrence, matchText) {
    var wrap = document.createElement("span");
    wrap.appendChild(text(occurrence.before ? "…" + occurrence.before + " " : ""));
    var match = document.createElement("strong");
    match.textContent = matchText;
    wrap.appendChild(match);
    wrap.appendChild(text(occurrence.after ? " " + occurrence.after + "…" : ""));
    return wrap;
  }

  function buildSideButton(pageNo, index, span, side) {
    var button = document.createElement("button");
    button.className = "side " + side + (span.state === side ? " is-current" : "");
    button.textContent = span[side];

    if (span.state === side) {
      // Already what the document says -- shown as the active side rather
      // than as a button that would do nothing.
      button.title = "This is what the document currently says.";
      button.disabled = true;
    } else if (span.occurrences && span.selected == null && side === span.choose_direction) {
      button.title = "This text appears more than once - pick which copy below.";
      on(button, "click", function () {
        var choices = button.parentNode.querySelector(".choices");
        if (!choices) { return; }
        choices.open = true;
        choices.scrollIntoView({ block: "nearest" });
        choices.classList.remove("flash");
        void choices.offsetWidth;  // restart the animation on a repeat click
        choices.classList.add("flash");
      });
    } else {
      button.title = "Put this reading into the document.";
      on(button, "click", function () {
        applyDiff({ page: pageNo, index: index, direction: side });
      });
    }
    return button;
  }

  function applyDiff(request) {
    if (!state.current) { return; }
    api("/apply/" + state.jobId + "/" + encodeURIComponent(state.current.name), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request)
    }).then(function (result) {
      // Remember a single-span click that achieved nothing, so the row can
      // say why rather than appearing inert.
      state.lastSkip = (!request.all && result.skipped.length)
        ? result.skipped[0]
        : null;
      // A pick that could not be made leaves its picker open to pick again.
      if (request.at_offset != null && result.skipped.length) {
        delete state.choiceOpen[request.page + ":" + request.index];
      }

      // The server returns the whole document, so the editor and preview stay
      // in step with what the diff view just did.
      state.current.text = result.text;
      state.current.diffs = result.diffs;
      $("editor").value = result.text;
      renderPreview();
      state.dirty[state.current.name] = true;

      renderDiff(state.current);
      renderDocList();

      var status = $("diff-status");
      if (status) { status.textContent = describeApply(result); }
    }).catch(function (err) {
      var status = $("diff-status");
      if (status) { status.textContent = "Could not apply: " + err.message; }
    });
  }

  function describeApply(result) {
    var counts = result.counts || {};
    var parts = [];
    if (counts.applied) { parts.push(counts.applied + " applied"); }
    if (counts.unchanged) { parts.push(counts.unchanged + " already set"); }

    // Anything the server refused to place is reported rather than hidden --
    // a silently skipped span would leave the user believing it was applied.
    var refused = (counts.not_found || 0) + (counts.ambiguous || 0)
      + (counts.unplaceable || 0) + (counts.moved || 0);
    if (refused) {
      parts.push(refused + " left alone (" + describeSkips(result.skipped) + ")");
    }
    return parts.length ? parts.join(", ") + "." : "Nothing to change.";
  }

  function describeSkips(skipped) {
    var reasons = {
      ambiguous: "text appears more than once - pick the copy on its row",
      moved: "the text changed since the copies were listed - pick again",
      not_found: "text not found in the document",
      unplaceable: "nothing to match against"
    };
    var seen = [];
    (skipped || []).forEach(function (item) {
      var reason = reasons[item.status] || item.status;
      if (seen.indexOf(reason) === -1) { seen.push(reason); }
    });
    return seen.join("; ");
  }

  function renderRecovery(doc) {
    var body = $("recovery-body");
    body.innerHTML = "";

    if (!doc.recovery || !doc.recovery.length) {
      $("recovery-intro").textContent = "";
      var empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "No pages were sent to the vision model for recovery in this run.";
      body.appendChild(empty);
      return;
    }

    $("recovery-intro").textContent =
      "The vision model's own transcription of pages the OCR pass flagged. "
      + "It is shown for comparison only - it was never spliced into the draft.";

    doc.recovery.forEach(function (block) {
      var section = document.createElement("section");
      section.className = "recovery-page";

      var header = document.createElement("header");
      header.textContent = "Page " + block.page + " - " + block.reason;
      section.appendChild(header);

      var content = document.createElement("div");
      content.className = "recovery-body";
      // The block is HTML the pipeline built from the model's transcription,
      // already escaped by the script's own escape_user_content().
      content.innerHTML = block.html;
      section.appendChild(content);

      body.appendChild(section);
    });
  }

  // ---------- page fixes: dedupe + rotation ----------

  function renderFixes(doc) {
    var body = $("fixes-body");
    body.innerHTML = "";

    var duplicates = doc.duplicates || [];
    var rotations = doc.rotations || [];
    var applied = rotations.filter(function (r) { return r.applied; });
    var review = rotations.filter(function (r) { return !r.applied; });

    if (!duplicates.length && !rotations.length) {
      var empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "No repeated or sideways pages were found in this document.";
      body.appendChild(empty);
      return;
    }

    // Page numbers everywhere below are the uploaded PDF's own, so they match
    // what the user sees when they open their file.
    if (duplicates.length) {
      body.appendChild(fixSection(
        "Repeated pages removed (" + duplicates.length + ")",
        "Each of these pages matched an earlier page and was taken out before OCR.",
        duplicates.map(function (d) {
          return fixCard(
            [
              pageFigure(doc, "original", d.dupe_idx, "Page " + d.duplicate_page + " - removed", 0),
              pageFigure(doc, "original", d.orig_idx, "Page " + d.original_page + " - kept", 0)
            ],
            d.score + "% text match, printed page number: " + d.folio
          );
        })
      ));
    }

    if (applied.length) {
      body.appendChild(fixSection(
        "Sideways pages rotated (" + applied.length + ")",
        "Turned upright before OCR. Only the page's rotation setting changed; "
          + "the scan itself is untouched.",
        applied.map(function (r) {
          return fixCard(
            [
              pageFigure(doc, "original", r.original_idx, "Page " + r.original_page + " - as scanned", 0),
              pageFigure(doc, "fixed", r.fixed_idx, "Turned " + r.rotation + "°", 0)
            ],
            r.confidence + " confidence: " + r.why
          );
        })
      ));
    }

    if (review.length) {
      body.appendChild(fixSection(
        "Possibly sideways - left alone (" + review.length + ")",
        "The evidence was too weak to rotate these automatically. The right-hand "
          + "image is only a preview of the suggested turn; the PDF was not changed.",
        review.map(function (r) {
          return fixCard(
            [
              pageFigure(doc, "original", r.original_idx, "Page " + r.original_page + " - as scanned", 0),
              pageFigure(doc, "original", r.original_idx, "If turned " + r.rotation + "°", r.rotation)
            ],
            r.why
          );
        })
      ));
    }
  }

  function fixSection(title, hint, cards) {
    var section = document.createElement("section");
    section.className = "fix-section";

    var header = document.createElement("h3");
    header.textContent = title;
    section.appendChild(header);

    var note = document.createElement("p");
    note.className = "hint";
    note.textContent = hint;
    section.appendChild(note);

    var grid = document.createElement("div");
    grid.className = "fix-grid";
    cards.forEach(function (card) { grid.appendChild(card); });
    section.appendChild(grid);
    return section;
  }

  function fixCard(figures, caption) {
    var card = document.createElement("div");
    card.className = "fix-card";

    var pair = document.createElement("div");
    pair.className = "fix-pair";
    figures.forEach(function (figure) { pair.appendChild(figure); });
    card.appendChild(pair);

    var note = document.createElement("p");
    note.className = "fix-caption";
    note.textContent = caption;
    card.appendChild(note);
    return card;
  }

  function pageFigure(doc, source, pageIdx, label, previewTurn) {
    var figure = document.createElement("figure");
    figure.className = "page-thumb";

    // A square frame, so a portrait page and the same page turned landscape
    // take up the same space and the card does not jump around.
    var frame = document.createElement("div");
    frame.className = "thumb-frame";

    var img = document.createElement("img");
    img.loading = "lazy";
    img.alt = label;
    img.src = "/page-image/" + state.jobId + "/" + source + "/" + pageIdx + "/"
      + encodeURIComponent(doc.name);
    // /Rotate turns a page clockwise, and so does a positive CSS rotate, so
    // the preview shows exactly what applying the suggestion would give.
    if (previewTurn) { img.style.transform = "rotate(" + previewTurn + "deg)"; }
    frame.appendChild(img);
    figure.appendChild(frame);

    var caption = document.createElement("figcaption");
    caption.textContent = label;
    figure.appendChild(caption);
    return figure;
  }

  // ---------- working folder ----------

  function openFolder() {
    // The job on screen, or all of workdir/ before there is one.
    var path = state.jobId ? "/open-folder/" + encodeURIComponent(state.jobId) : "/open-folder";
    api(path, { method: "POST" }).catch(function (err) {
      // Said wherever the user is looking; the message includes the path.
      if ($("screen-run").hidden) {
        $("save-status").textContent = err.message;
      } else {
        showRunError(err.message);
      }
    });
  }

  // ---------- editing and saving ----------

  function onEditorInput() {
    renderPreview();
    if (!state.current) { return; }
    state.dirty[state.current.name] = true;

    // Debounced: the preview updates on every keystroke locally, but the
    // server only needs the text when the user pauses.
    clearTimeout(state.editTimer);
    state.editTimer = setTimeout(function () {
      api("/edit/" + state.jobId + "/" + encodeURIComponent(state.current.name), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: $("editor").value })
      }).then(renderDocList).catch(function () { /* retried on next keystroke */ });
    }, 400);
  }

  function saveEdits() {
    api("/save/" + state.jobId, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    }).then(function (result) {
      state.dirty = {};
      renderDocList();
      $("save-status").textContent = result.written.length
        ? "Saved " + result.written.length + " file(s) - use Open folder to find them in output."
        : "Nothing to save.";
    });
  }

  // =====================================================================
  // Wiring
  // =====================================================================

  function init() {
    setupDropzone();
    on($("run-btn"), "click", startRun);
    on($("mode-select"), "change", updateModeHelp);
    on($("run-kind"), "change", updateRunKind);
    updateRunKind();
    on($("editor"), "input", onEditorInput);
    on($("save-btn"), "click", saveEdits);
    on($("open-folder-btn"), "click", openFolder);

    document.querySelectorAll(".topbar .tab").forEach(function (tab) {
      on(tab, "click", function () {
        if (!tab.disabled) { switchScreen(tab.dataset.screen); }
      });
    });
    document.querySelectorAll(".view-toggle .tab").forEach(function (tab) {
      on(tab, "click", function () {
        if (!tab.disabled) { switchView(tab.dataset.view); }
      });
    });

    refreshOllama();
    loadModels();
    // Re-check Ollama periodically so starting it while the app is open
    // clears the banner without a restart.
    setInterval(refreshOllama, 15000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
