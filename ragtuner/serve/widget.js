/* ragtuner chat widget. Usage:
   <script src="https://your-host/widget.js" data-api="https://your-host" data-title="Ask the docs"></script>
   Posts {query} to <data-api>/ask and renders the answer with numbered citations. */
(function () {
  var script = document.currentScript;
  var api = (script && script.getAttribute("data-api")) || "";
  var title = (script && script.getAttribute("data-title")) || "Ask the docs";
  var css = [
    ".rt-btn{position:fixed;right:16px;bottom:16px;z-index:2147483000;border:0;border-radius:24px;padding:12px 18px;",
    "background:#1f2937;color:#fff;font:600 14px system-ui,sans-serif;cursor:pointer}",
    ".rt-panel{position:fixed;right:16px;bottom:72px;z-index:2147483000;width:min(380px,calc(100vw - 32px));max-height:70vh;",
    "display:none;flex-direction:column;background:#fff;color:#111;border:1px solid #d1d5db;border-radius:12px;",
    "box-shadow:0 8px 24px rgba(0,0,0,.12);font:14px/1.45 system-ui,sans-serif}",
    ".rt-panel.open{display:flex}.rt-head{padding:12px 14px;font-weight:600;border-bottom:1px solid #e5e7eb}",
    ".rt-log{padding:12px 14px;overflow:auto;flex:1}.rt-q{font-weight:600;margin:10px 0 4px}",
    ".rt-a{white-space:pre-wrap}.rt-cite{font-size:12px;color:#4b5563;margin-top:6px}",
    ".rt-form{display:flex;gap:8px;padding:10px;border-top:1px solid #e5e7eb}",
    ".rt-form input{flex:1;padding:8px 10px;border:1px solid #d1d5db;border-radius:8px;font:inherit}",
    ".rt-form button{padding:8px 12px;border:0;border-radius:8px;background:#1f2937;color:#fff;font:inherit;cursor:pointer}",
    "@media (prefers-color-scheme: dark){.rt-panel{background:#111827;color:#f3f4f6;border-color:#374151}",
    ".rt-head,.rt-form{border-color:#374151}.rt-form input{background:#1f2937;color:#f3f4f6;border-color:#374151}",
    ".rt-cite{color:#9ca3af}}"
  ].join("");
  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);
  var btn = document.createElement("button");
  btn.className = "rt-btn"; btn.type = "button"; btn.textContent = title;
  btn.setAttribute("aria-expanded", "false");
  var panel = document.createElement("div");
  panel.className = "rt-panel"; panel.setAttribute("role", "dialog"); panel.setAttribute("aria-label", title);
  panel.innerHTML = '<div class="rt-head"></div><div class="rt-log" aria-live="polite"></div>' +
    '<form class="rt-form"><input name="q" autocomplete="off" aria-label="Your question" placeholder="Type a question">' +
    '<button type="submit">Ask</button></form>';
  panel.querySelector(".rt-head").textContent = title;
  document.body.appendChild(btn); document.body.appendChild(panel);
  var log = panel.querySelector(".rt-log"), form = panel.querySelector("form"), input = form.querySelector("input");
  btn.addEventListener("click", function () {
    var open = panel.classList.toggle("open");
    btn.setAttribute("aria-expanded", String(open));
    if (open) input.focus();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && panel.classList.contains("open")) { panel.classList.remove("open"); btn.setAttribute("aria-expanded", "false"); btn.focus(); }
  });
  function add(cls, text) { var d = document.createElement("div"); d.className = cls; d.textContent = text; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var q = input.value.trim(); if (!q) return;
    input.value = ""; add("rt-q", q); var a = add("rt-a", "Searching...");
    fetch(api + "/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: q }) })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.status); return j; }); })
      .then(function (j) {
        a.textContent = j.answer;
        (j.citations || []).forEach(function (c) { add("rt-cite", "[" + c.n + "] " + c.doc_id + (c.page ? ", page " + c.page : "")); });
      })
      .catch(function (err) { a.textContent = "Could not get an answer: " + err.message; });
  });
})();
