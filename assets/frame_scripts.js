const el = document.querySelector("svg circle");
let state = {
    eventToCoordinates: eventToSvgCoordinates,
    dragging: null,
    _pos: undefined,
    get pos() {
        return this._pos;
    },
    set pos(p) { 
        this._pos = {x: p.x, y: p.y};
        el.setAttribute('cx', this._pos.x);
        el.setAttribute('cy', this._pos.y);
    },
};
state.pos = {x: 0, y: 0};
makeDraggable(state, el);

function makeDraggable(state, el) {
    // from https://www.redblobgames.com/making-of/draggable/
    function start(event) {
        if (event.button !== 0) return; // left button only
        let {x, y} = state.eventToCoordinates(event);
        state.dragging = {dx: state.pos.x - x, dy: state.pos.y - y};
        el.classList.add('dragging');
        el.setPointerCapture(event.pointerId);
    }

    function end(_event) {
        state.dragging = null;
        el.classList.remove('dragging');
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
}

function eventToSvgCoordinates(event, el=event.currentTarget) {
    const svg = el.ownerSVGElement;
    let p = svg.createSVGPoint();
    p.x = event.clientX;
    p.y = event.clientY;
    p = p.matrixTransform(svg.getScreenCTM().inverse());
    return p;
}