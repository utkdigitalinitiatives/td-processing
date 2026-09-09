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
    pending: [],        // File objects staged in the drop zone
    jobId: null,
    poll: null,         // setInterval handle for /status
    documents: [],      // summaries from /status
    current: null,      // full payload for the open document
    dirty: {},          // name -> true when edited but unsaved
    editTimer: null
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
      var already = state.pending.some(function (f) {
        return f.name === file.name && f.size === file.size;
      });
      if (!already) { state.pending.push(file); }
    });
    renderPending();
  }

  function renderPending() {
    var list = $("pending-list");
    list.innerHTML = "";
    state.pending.forEach(function (file, index) {
      var li = document.createElement("li");
      li.appendChild(text(file.name));
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

  function startRun() {
    var form = new FormData();
    state.pending.forEach(function (file) { form.append("files", file); });
    form.append("model", $("model-select").value);
    form.append("mode", $("mode-select").value);

    $("run-btn").disabled = true;
    show($("run-error"), false);

    api("/upload", { method: "POST", body: form })
      .then(function (data) {
        state.jobId = data.job_id;
        state.pending = [];
        renderPending();
        show($("progress-panel"), true);
        startPolling();
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
      list.appendChild(li);
    });
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

  function kindLabel(kind) {
    if (kind === "low_confidence") { return "low-conf"; }
    if (kind === "equation") { return "equation"; }
    return "diff";
  }

  function selectDocument(name) {
    api("/document/" + state.jobId + "/" + encodeURIComponent(name))
      .then(function (doc) {
        state.current = doc;
        $("doc-title").textContent = doc.name;

        var meta = [];
        if (doc.page_count != null) { meta.push(doc.page_count + " pages after dedupe"); }
        if (doc.duplicates_removed) { meta.push(doc.duplicates_removed + " repeated page(s) removed"); }
        meta.push("model: " + doc.model_used);
        $("doc-meta").textContent = meta.join(" - ");

        $("editor").value = doc.text;
        renderPreview();

        // Tabs are disabled rather than hidden when a mode produced no such
        // data, so the UI never shows a control that silently does nothing.
        $("tab-diff").disabled = !(doc.diffs && doc.diffs.length);
        $("tab-recovery").disabled = !(doc.recovery && doc.recovery.length);

        renderDiff(doc);
        renderRecovery(doc);
        renderDocList();
        switchView("edit");
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
      + "same page. Spans marked MERGED were applied to the draft automatically; "
      + "FLAGGED ones were not and still need your eye.";

    doc.diffs.forEach(function (page) {
      var section = document.createElement("section");
      section.className = "diff-page";

      var header = document.createElement("header");
      header.textContent = "Page " + page.page + " - " + page.spans.length + " difference(s)";
      section.appendChild(header);

      page.spans.forEach(function (span) {
        var row = document.createElement("div");
        var label = (span.label || "DIFF").toLowerCase().replace(/\s+/g, "-");
        row.className = "diff-row " + label;

        var labelCell = document.createElement("div");
        labelCell.className = "label";
        labelCell.textContent = span.label || "diff";
        row.appendChild(labelCell);

        var ocr = document.createElement("div");
        ocr.className = "ocr";
        ocr.textContent = span.ocr;
        row.appendChild(ocr);

        var vlm = document.createElement("div");
        vlm.className = "vlm";
        vlm.textContent = span.vlm;
        row.appendChild(vlm);

        section.appendChild(row);
      });
      body.appendChild(section);
    });
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

  // ---------- editing, saving, exporting ----------

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
        ? "Saved " + result.written.length + " file(s) to workdir."
        : "Nothing to save.";
    });
  }

  function exportDocs() {
    // Fetched into a blob and handed to a synthetic <a download> rather than
    // navigating to the route: /export takes a JSON body, which a plain link
    // or form submission cannot send.
    fetch("/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: state.jobId })
    }).then(function (resp) {
      if (!resp.ok) { return resp.json().then(function (b) { throw new Error(b.error); }); }
      var disposition = resp.headers.get("Content-Disposition") || "";
      var match = /filename="?([^"]+)"?/.exec(disposition);
      var filename = match ? match[1] : "export.md";
      return resp.blob().then(function (blob) {
        var url = URL.createObjectURL(blob);
        var link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        $("save-status").textContent = "Exported " + filename + ".";
      });
    }).catch(function (err) {
      $("save-status").textContent = "Export failed: " + err.message;
    });
  }

  // =====================================================================
  // Wiring
  // =====================================================================

  function init() {
    setupDropzone();
    on($("run-btn"), "click", startRun);
    on($("mode-select"), "change", updateModeHelp);
    on($("editor"), "input", onEditorInput);
    on($("save-btn"), "click", saveEdits);
    on($("export-btn"), "click", exportDocs);

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
