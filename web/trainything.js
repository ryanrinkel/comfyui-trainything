// TrainyThing's frontend touch, loaded by ComfyUI from the pack's WEB_DIRECTORY:
//  - the five nodes come coloured, so the "send to a phone" part of a graph
//    reads as one box (a colour saved in a workflow, or picked from the node's
//    right-click menu, wins: this only fills in a node that has none);
//  - a fresh node shows the essentials only. Inputs the schema marks
//    `advanced` fold behind a "more settings" switch on the classic canvas
//    (the Vue node renderer has its own "Show advanced inputs" for the same
//    flag). The switch is not an input: it never reaches the prompt.
import { app } from "../../scripts/app.js";

const PREFIX = "TrainyThing";
const COLOR = "#283356";   // the title bar
const BGCOLOR = "#344275"; // the body
const SWITCH = "more settings";

function defaults(nodeData) {
  const out = {};
  for (const section of ["required", "optional"]) {
    for (const [name, spec] of Object.entries(nodeData.input?.[section] ?? {})) {
      out[name] = spec?.[1]?.default;
    }
  }
  return out;
}

function fold(node, show) {
  const adv = node.widgets?.filter((w) => w.options?.advanced) ?? [];
  for (const w of adv) w.hidden = !show;
  const [width] = node.size;
  const [, height] = node.computeSize();
  node.setSize([width, height]);
  // During a wait the QR sits below the widgets: keep its room.
  if (node.imgs?.length && typeof node.setSizeForImage === "function") node.setSizeForImage();
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "trainything.nodes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData?.name?.startsWith(PREFIX)) return;
    const defs = defaults(nodeData);

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      if (!this.color) this.color = COLOR;
      if (!this.bgcolor) this.bgcolor = BGCOLOR;
      const adv = this.widgets?.filter((w) => w.options?.advanced) ?? [];
      if (adv.length && !this.widgets.some((w) => w.name === SWITCH)) {
        const sw = this.addWidget("toggle", SWITCH, false, (v) => fold(this, v), { serialize: false, on: "shown", off: "hidden" });
        // Sit right after the essentials: the schema lists those first.
        this.widgets.splice(this.widgets.indexOf(sw), 1);
        this.widgets.splice(this.widgets.indexOf(adv[0]), 0, sw);
        fold(this, false);
      }
      return r;
    };

    // A loaded workflow restores the saved size and values after creation:
    // fold again, as it was saved. A workflow from before the switch (none
    // saved) unfolds when something advanced was changed from its default,
    // so a setting never hides.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure?.apply(this, arguments);
      const sw = this.widgets?.find((w) => w.name === SWITCH);
      if (sw) {
        let saved = info?.widgets_values_named?.[SWITCH];
        if (typeof saved !== "boolean" && Array.isArray(info?.widgets_values) && info.widgets_values.length === this.widgets.length) {
          saved = info.widgets_values[this.widgets.indexOf(sw)];
        }
        if (typeof saved === "boolean") {
          sw.value = saved;
        } else {
          sw.value = this.widgets.some((w) => w.options?.advanced && defs[w.name] !== undefined && w.value !== defs[w.name]);
        }
        fold(this, sw.value);
      }
      return r;
    };
  },
});
