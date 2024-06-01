const el = document.querySelector(".draggable");
const movableEl = el.parentNode;
let state = {
    eventToCoordinates(event) { return {x: event.clientX, y: event.clientY}; },
    dragging: false,
    _pos: {x: 0, y: 0},
    get pos() { return this._pos },
    set pos(p) {
        this._pos = p;
        movableEl.style.transform = //*
            `translate(${this._pos.x}px,${this._pos.y}px)`;
    },
}
state.pos = {x: 850, y: 230};
makeDraggable(state, el);
function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

function makeDraggable(state, el) {
    // from https://www.redblobgames.com/making-of/draggable/
    function start(event) {
        if (event.button !== 0) return; // left button only
        let {x, y} = state.eventToCoordinates(event);
        state.dragging = {dx: state.pos.x - x, dy: state.pos.y - y};
        el.classList.add('dragging');
        el.setPointerCapture(event.pointerId);
        el.style.userSelect = 'none'; // if there's text
        el.style.webkitUserSelect = 'none'; // safari
    }

    function end(_event) {
        state.dragging = null;
        el.classList.remove('dragging');
        el.style.userSelect = ''; // if there's text
        el.style.webkitUserSelect = ''; // safari
    }

    function move(event) {
        if (!state.dragging) return;
        let {x, y} = state.eventToCoordinates(event);
        state.pos = {x: x + state.dragging.dx, y: y + state.dragging.dy};
    }

    el.addEventListener('pointerdown', start);
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('pointermove', move)
    el.addEventListener('touchstart', (e) => e.preventDefault());
    el.addEventListener('dragstart', (e) => e.preventDefault());
}