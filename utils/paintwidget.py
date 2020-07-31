from .labwidget import Widget, Property, minify


class PaintWidget(Widget):
    def __init__(self,
                 width=256, height=256,
                 image='', mask='', brushsize=10.0, oneshot=False, disabled=False,
                 vanishing=True, opacity=0.7,
                 **kwargs):
        super().__init__(**kwargs)
        self.mask = Property(mask)
        self.image = Property(image)
        self.vanishing = Property(vanishing)
        self.brushsize = Property(brushsize)
        self.erase = Property(False)
        self.oneshot = Property(oneshot)
        self.disabled = Property(disabled)
        self.width = Property(width)
        self.height = Property(height)
        self.opacity = Property(opacity)
        self.startpos = Property(None)
        self.dragpos = Property(None)
        self.dragging = Property(False)

    def widget_js(self):
        return minify(f'''
      {PAINT_WIDGET_JS}
      var pw = new PaintWidget(element, model);
    ''')

    def widget_html(self):
        v = self.view_id()
        return minify(f'''
    <style>
    #{v} {{ position: relative; display: inline-block; }}
    #{v} .paintmask {{
      position: absolute; top:0; left: 0; z-index: 1;
      opacity: { self.opacity } }}
    #{v} .paintmask.vanishing {{
      opacity: 0; transition: opacity .1s ease-in-out; }}
    #{v} .paintmask.vanishing:hover {{ opacity: { self.opacity }; }}
    </style>
    <div id="{v}"></div>
    ''')


PAINT_WIDGET_JS = """
class PaintWidget {
  constructor(el, model) {
    this.el = el;
    this.model = model;
    this.size_changed();
    this.model.on('mask', this.mask_changed.bind(this));
    this.model.on('image', this.image_changed.bind(this));
    this.model.on('vanishing', this.mask_changed.bind(this));
    this.model.on('width', this.size_changed.bind(this));
    this.model.on('height', this.size_changed.bind(this));
  }
  mouse_stroke(first_event) {
    var self = this;
    if (first_event.which === 3 || first_event.button === 2) {
        first_event.preventDefault();
        self.mask_canvas.style.pointerEvents = 'none';
        setTimeout(() => {
            self.mask_canvas.style.pointerEvents = 'all';
        }, 3000);
        return;
    }
    if (self.model.get('disabled')) { return; }
    if (self.model.get('oneshot')) {
        var canvas = self.mask_canvas;
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    function track_mouse(evt) {
      if (evt.type == 'keydown' || self.model.get('disabled')) {
        if (self.model.get('disabled') || evt.key === "Escape") {
          window.removeEventListener('mousemove', track_mouse);
          window.removeEventListener('mouseup', track_mouse);
          window.removeEventListener('keydown', track_mouse, true);
          if (self.model.get('dragging')) {
            self.model.set('dragging', false);
          }
          self.mask_changed();
        }
        return;
      }
      if (evt.type == 'mouseup' ||
        (typeof evt.buttons != 'undefined' && evt.buttons == 0)) {
        window.removeEventListener('mousemove', track_mouse);
        window.removeEventListener('mouseup', track_mouse);
        window.removeEventListener('keydown', track_mouse, true);
        self.model.set('dragging', false);
        self.model.set('mask', self.mask_canvas.toDataURL());
        return;
      }
      var p = self.cursor_position(evt);
      var d = self.model.get('dragging');
      var e = self.model.get('erase') ^ (evt.ctrlKey);
      if (!d) { self.model.set('startpos', [p.x, p.y]); }
      self.model.set('dragpos', [p.x, p.y]);
      if (!d) { self.model.set('dragging', true); }
      self.fill_circle(p.x, p.y,
          self.model.get('brushsize'),
          e);
    }
    this.mask_canvas.focus();
    window.addEventListener('mousemove', track_mouse);
    window.addEventListener('mouseup', track_mouse);
    window.addEventListener('keydown', track_mouse, true);
    track_mouse(first_event);
  }
  mask_changed() {
    this.mask_canvas.classList.toggle("vanishing", this.model.get('vanishing'));
    this.draw_data_url(this.mask_canvas, this.model.get('mask'));
  }
  image_changed() {
    this.image.src = this.model.get('image');
  }
  size_changed() {
    this.mask_canvas = document.createElement('canvas');
    this.image = document.createElement('img');
    this.mask_canvas.className = "paintmask";
    this.image.className = "paintimage";
    for (var attr of ['width', 'height']) {
      this.mask_canvas[attr] = this.model.get(attr);
      this.image[attr] = this.model.get(attr);
    }

    this.el.innerHTML = '';
    this.el.appendChild(this.image);
    this.el.appendChild(this.mask_canvas);
    this.mask_canvas.addEventListener('mousedown',
        this.mouse_stroke.bind(this));
    this.mask_changed();
    this.image_changed();
  }

  cursor_position(evt) {
    const rect = this.mask_canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    return {x: x, y: y};
  }

  fill_circle(x, y, r, erase, blur) {
    var ctx = this.mask_canvas.getContext('2d');
    ctx.save();
    if (blur) {
        ctx.filter = 'blur(' + blur + 'px)';
    }
    ctx.globalCompositeOperation = (
        erase ? "destination-out" : 'source-over');
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore()
  }

  draw_data_url(canvas, durl) {
    var ctx = canvas.getContext('2d');
    var img = new Image;
    canvas.pendingImg = img;
    function imgdone() {
      if (canvas.pendingImg == img) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        canvas.pendingImg = null;
      }
    }
    img.addEventListener('load', imgdone);
    img.addEventListener('error', imgdone);
    img.src = durl;
  }
}
"""
