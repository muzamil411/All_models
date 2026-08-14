/*
 * Deterministic animation engine.
 *
 * Scenes never use CSS animations or requestAnimationFrame — the renderer
 * calls SCENE.seek(t) for each frame and expects the DOM to be a pure
 * function of t. That is what makes an offline render frame-accurate.
 */

// Easing curves. expoOut matches the settle measured in the reference video
// (displacement decaying ~(-22,-26) -> (-12,-15) -> (-3,-3) -> 0 over ~0.8s).
const ease = {
  linear: (x) => x,
  expoOut: (x) => (x >= 1 ? 1 : 1 - Math.pow(2, -10 * x)),
  cubicOut: (x) => 1 - Math.pow(1 - x, 3),
  quintOut: (x) => 1 - Math.pow(1 - x, 5),
  backOut: (x) => {
    const c = 1.70158 + 1;
    return 1 + c * Math.pow(x - 1, 3) + 1.70158 * Math.pow(x - 1, 2);
  },
  cubicInOut: (x) =>
    x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2,
};

/** Normalised progress of t through [start, start+dur], clamped to 0..1. */
function span(t, start, dur) {
  if (dur <= 0) return t >= start ? 1 : 0;
  return Math.min(1, Math.max(0, (t - start) / dur));
}

/** Eased progress through a window. */
function anim(t, start, dur, curve = "expoOut") {
  return ease[curve](span(t, start, dur));
}

/** Map eased progress onto a value range. */
function tween(t, start, dur, from, to, curve = "expoOut") {
  return from + (to - from) * anim(t, start, dur, curve);
}

/** Scene params arrive as base64 JSON on the URL so one file serves many instances. */
function params(defaults = {}) {
  const raw = new URLSearchParams(location.search).get("p");
  if (!raw) return defaults;
  const decoded = JSON.parse(decodeURIComponent(escape(atob(raw))));
  return { ...defaults, ...decoded };
}

/** Register the scene the renderer will drive. */
function defineScene(scene) {
  window.SCENE = scene;
  window.SCENE_READY = document.fonts.ready.then(() => {
    scene.seek(0);
    return true;
  });
}
