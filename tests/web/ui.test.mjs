import { test } from "node:test";
import assert from "node:assert/strict";
import { esc, plateChip } from "../../web/js/ui.js";

test("esc neutralises HTML", () => {
  assert.equal(esc(`<b a="1">&'`), "&lt;b a=&quot;1&quot;&gt;&amp;&#39;");
  assert.equal(esc(null), "");
});

test("plateChip renders a plate, or a dash when nothing was read", () => {
  assert.equal(plateChip("VW2468"), '<span class="plate">VW 2468</span>');
  assert.match(plateChip(""), /class="plate plate-empty"[^>]*>—</);
});
