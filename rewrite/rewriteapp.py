import copy
import os
import torch
import json
from utils import renormalize, imgviz
from utils import show, labwidget, paintwidget
from collections import OrderedDict


##########################################################################
# UI
##########################################################################

class GanRewriteApp(labwidget.Widget):

    def __init__(self, gw, mask_dir=None, size=256, num_canvases=9):
        super().__init__(style=dict(border="0", padding="0",
                                    display="inline-block", width="1000px",
                                    left="0", margin="0"
                                    ), className='rwa')
        self.gw = gw
        self.size = size
        self.savedir = self.gw.cachedir if mask_dir is None else mask_dir
        self.original_model = copy.deepcopy(gw.model)
        self.request = {}
        self.imgnum_textbox = labwidget.Textbox('0-%d' % (num_canvases - 1)
                                                ).on('value', self.change_numbers)
        self.msg_out = labwidget.Div()
        self.loss_out = labwidget.Div()
        self.query_out = labwidget.Div()
        self.copy_canvas = paintwidget.PaintWidget(
            image='', width=self.size * 0.75, height=self.size * 0.75
        ).on('mask', self.change_copy_mask)
        self.paste_canvas = paintwidget.PaintWidget(
            image='', width=self.size * 0.75, height=self.size * 0.75,
            opacity=0.0, oneshot=True,
        ).on('mask', self.change_paste_mask)
        self.object_out = labwidget.Div(
            style={'display': 'inline-block',
                   'vertical-align': 'top',
                   'width': '%spx' % size,
                   'height': '%spx' % size})
        self.target_out = labwidget.Div(
            style={'display': 'inline-block',
                   'vertical-align': 'top',
                   'width': '%spx' % size,
                   'height': '%spx' % size})
        self.context_out = labwidget.Div(
            style={'display': 'inline-block',
                   'vertical-align': 'top',
                   'text-align': 'left',
                   'width': '%spx' % ((size + 2) * 3 // 2),
                   'height': '%spx' % (size * 3 // 8 + 20),
                   'white-space': 'nowrap',
                   'overflow-x': 'scroll'},
            className='ctx_tray')
        self.context_img_array = []
        self.keytray_div = labwidget.Div(style={'display': 'none'})
        self.keytray_menu = labwidget.Menu(
        ).on('selection', self.repaint_key_tray)
        self.keytray_removebtn = labwidget.Button('remove'
                                                  ).on('click', self.keytray_remove)
        self.keytray_showbtn = labwidget.Button('show'
                                                ).on('click', self.keytray_show)
        self.keytray_querybtn = labwidget.Button('query'
                                                 ).on('click', self.keytray_query)
        self.keytray_zerobtn = labwidget.Button('zero'
                                                ).on('click', self.keytray_zero)
        self.keytray_canvas = paintwidget.PaintWidget(
            width=self.size, height=self.size,
            vanishing=False, disabled=True)
        self.keytray_div.show([
            [
                self.keytray_menu,
                self.keytray_removebtn,
                self.keytray_showbtn,
                self.keytray_zerobtn,
                self.keytray_querybtn,
            ],
            [self.keytray_canvas]])
        inline = dict(display='inline')
        self.query_btn = labwidget.Button('Match Sel', style=inline
                                          ).on('click', self.query)
        self.context_querybtn = labwidget.Button('Search', style=inline
                                                 ).on('click', self.keytray_query)
        self.highlight_btn = labwidget.Button('Show Context Matches', style=inline
                                              ).on('click', self.toggle_highlight)
        self.original_btn = labwidget.Button('Toggle Original', style=inline
                                             ).on('click', self.toggle_original)
        self.object_btn = labwidget.Button('Copy', style=inline
                                           ).on('click', self.pick_object)
        self.key_btn = labwidget.Button('Add to Context', style=inline
                                        ).on('click', self.key_add)
        self.paste_btn = labwidget.Button('Paste', style=inline
                                          ).on('click', self.paste)
        self.snap_btn = labwidget.Button('Snap'
                                         ).on('click', self.snapshot_images)
        self.brushsize_textbox = labwidget.Textbox(10, desc='brush: ', size=3
                                                   ).on('value', self.change_brushsize)
        self.rank_textbox = labwidget.Textbox(
            '1', desc='rank: ', size=4, style=inline)
        self.paste_niter_textbox = labwidget.Textbox(
            '2001', desc='paste niter: ', size=8)
        self.paste_piter_textbox = labwidget.Textbox(
            '10', desc='proj every: ', size=4)
        self.paste_lr_textbox = labwidget.Textbox(
            '0.05', desc='paste lr: ', size=8)
        self.erase_btn = labwidget.Button('Erase').on('click', self.exec_erase)
        self.exec_btn = labwidget.Button('Execute Change',
                                         style=dict(display='inline', background='darkgreen')
                                         ).on('click', self.exec_request)
        self.overfit_btn = labwidget.Button('Overfit').on(
            'click', self.exec_overfit)
        self.revert_btn = labwidget.Button('Revert', style=inline
                                           ).on('click', self.revert)
        self.saved_list = labwidget.Datalist(choices=self.saved_names(),
                                             style=inline)
        self.load_btn = labwidget.Button('Load', style=inline
                                         ).on('click', self.tryload)
        self.save_btn = labwidget.Button('Save', style=inline
                                         ).on('click', self.save)
        self.sel = list(range(num_canvases))
        self.overwriting = True
        self.obj_acts = None
        self.query_key = None
        self.query_vis = False
        self.show_original = False
        self.query_rq = None
        self.query_key_valid = True
        self.clipped_activations = None
        self.canvas_array = []
        self.snap_image_array = []
        for i in range(num_canvases):
            self.canvas_array.append(paintwidget.PaintWidget(
                image=renormalize.as_url(
                    self.gw.render_image(i)),
                # width=self.size * 3 // 4, height=self.size * 3 // 4
                width=self.size, height=self.size
            ).on('mask', self.change_mask))
            self.canvas_array[-1].index = i
            self.snap_image_array.append(
                labwidget.Image(style={'margin-top': 0,
                                       'max-width': '%dpx' % self.size,
                                       'max-height': '%dpx' % self.size}))
            self.snap_image_array[-1].index = i
        self.current_mask_item = None

    def change_brushsize(self):
        brushsize = int(self.brushsize_textbox.value)
        for c in self.canvas_array:
            c.brushsize = brushsize

    def old_repaint_key_tray(self):
        if 'key' not in self.request or len(self.request['key']) == 0:
            self.keytray_div.style = {'display': 'none'}
            return
        keymasks = OrderedDict([
            (imgnum, mask) for imgnum, mask in self.request['key']])
        self.keytray_menu.choices = list(keymasks.keys())
        self.keytray_div.style = {'display': 'block'}
        if self.keytray_menu.selection is None or (
                int(self.keytray_menu.selection) not in keymasks):
            sel = int(self.keytray_menu.choices[-1])
            self.keytray_menu.selection = sel
        else:
            sel = int(self.keytray_menu.selection)
        self.keytray_canvas.image = renormalize.as_url(
            self.gw.render_image(sel, None))
        self.keytray_canvas.mask = keymasks[sel]

    def repaint_key_tray(self):
        if 'key' not in self.request:
            keymasks = {}
        else:
            keymasks = OrderedDict([
                (imgnum, mask) for imgnum, mask in self.request['key']])
        if len(self.context_img_array) < len(keymasks):
            while len(self.context_img_array) < len(keymasks):
                self.context_img_array.append(
                    labwidget.Image(style=dict(
                        maxWidth='%spx' % int(self.size * 3 // 8),
                        maxHeight='%spx' % int(self.size * 3 // 8),
                        border='1 px solid white')).on(
                        'click', self.click_context_img))
            self.context_out.show(*[[imgw] for imgw in self.context_img_array])
        for i, (imgnum, mask) in enumerate(keymasks.items()):
            imgw = self.context_img_array[i]
            area = (renormalize.from_url(mask, target='pt',
                                         size=self.gw.x_shape[2:])[0] > 0.25)
            imgw.render(self.gw.render_image(imgnum, mask=area,
                                             thickness=0, outside_bright=1.0, inside_color=[255, 255, 255]))
            imgw.imgnum = imgnum
        for i in range(len(keymasks), len(self.context_img_array)):
            self.context_img_array[i].src = ''
            self.context_img_array[i].imgnum = None

    def click_context_img(self, evt):
        if not evt.target or evt.target.imgnum is None:
            return
        imgnum = evt.target.imgnum
        index = [i for i, _ in self.request['key']].index(imgnum)
        if index >= 0:
            self.request['key'].pop(index)
            if len(self.request['key']) == 0:
                del self.request['key']
            self.repaint_key_tray()

    def keytray_remove(self):
        if self.keytray_menu.selection is None:
            return
        if 'key' not in self.request or len(self.request['key']) == 0:
            return
        sel = int(self.keytray_menu.selection)
        if sel is None:
            return True
        index = [imgnum for imgnum, _ in self.request['key']].index(sel)
        if index >= 0:
            self.request['key'].pop(index)
            if len(self.request['key']) == 0:
                del self.request['key']
            self.keytray_menu.selection = None
            self.repaint_key_tray()

    def keytray_show(self):
        if 'key' not in self.request or len(self.request['key']) == 0:
            return
        self.sel = [i for i, _ in self.request['key']]
        self.imgnum_textbox.value = ','.join(str(i) for i in self.sel)
        self.repaint_canvas_array()

    def keytray_query(self):
        if 'key' not in self.request or len(self.request['key']) == 0:
            return
        mkey = self.gw.multi_key_from_selection(self.request['key'], rank=1)
        self.exec_query(key=mkey[0])

    def keytray_zero(self):
        if 'key' not in self.request or len(self.request['key']) == 0:
            return
        rank = int(self.rank_textbox.value)
        mkey = self.gw.multi_key_from_selection(self.request['key'], rank=rank)
        self.gw.zero(mkey)
        self.repaint_canvas_array()
        self.show_msg(f'zeroed key value from model')

    def key_add(self):
        if (self.current_mask_item is None
                or not self.canvas_array[self.current_mask_item].mask):
            return
        imgnum = self.sel[self.current_mask_item]
        mask = self.canvas_array[self.current_mask_item].mask
        if 'key' in self.request:
            keymasks = OrderedDict([
                (imgnum, mask) for imgnum, mask in self.request['key']])
        else:
            keymasks = {}
        keymasks[imgnum] = mask
        self.request['key'] = list(keymasks.items())
        self.query_key_valid = False
        self.keytray_menu.selection = imgnum
        self.repaint_key_tray()

    def repaint_canvas_array(self):
        level = (self.query_rq.quantiles(0.999)[0]
                 if (self.query_vis and self.query_rq) else None)
        if self.show_original:
            saved_state_dict = copy.deepcopy(self.gw.model.state_dict())
            with torch.no_grad():
                self.gw.model.load_state_dict(self.original_model.state_dict())
        images = self.gw.render_image_batch(self.sel,
                                            self.query_key if self.query_vis else None, level,
                                            border_color=[255, 255, 255])
        if self.show_original:
            with torch.no_grad():
                self.gw.model.load_state_dict(saved_state_dict)

        for i, canvas in enumerate(self.canvas_array):
            canvas.mask = ''
            size = (canvas.height, canvas.width) if canvas.height else None
            if i < len(self.sel):
                canvas.image = renormalize.as_url(images[i], size=size)
            else:
                canvas.image = ''

    def snapshot_images(self):
        for canvas, snap in zip(self.canvas_array, self.snap_image_array):
            snap.src = canvas.image

    def clear_images(self):
        for snap in self.snap_image_array:
            snap.src = ''

    def change_numbers(self, c):
        sel = []
        for p in [r.split('-') for r in self.imgnum_textbox.value.split(',')]:
            try:
                bottom = int(p[0])
                top = int(p[1]) if len(p) > 1 else bottom
                sel.extend([i for i in range(bottom, top + 1)
                            if 0 <= i < len(self.gw.zds)])
            except Exception as e:
                print('Exception {}'.format(e))
                pass
            if len(sel) >= len(self.canvas_array):
                sel = sel[:len(self.canvas_array)]
        self.sel = sel
        self.repaint_canvas_array()
        self.clear_images()

    def change_paste_mask(self):
        if 'paste' not in self.request:
            return
        imgnum, oldmask = self.request['paste']
        mask = self.paste_canvas.mask
        if self.gw.is_empty_mask(mask):
            return
        self.request['paste'] = (imgnum, mask)
        self.exec_paste()

    def change_copy_mask(self, ev):
        if 'object' not in self.request:
            return
        imgnum, oldmask = self.request['object']
        mask = self.copy_canvas.mask
        if self.gw.is_empty_mask(mask):
            self.request.pop('object')
            self.copy_canvas.image = ''
        else:
            self.request['object'] = (imgnum, mask)
            self.exec_object()

    def change_mask(self, ev):
        i = ev.target.index
        for j in range(len(self.canvas_array)):
            if j != i:
                self.canvas_array[j].mask = ''
        self.current_mask_item = i

    def set_mask(self, imgnum, mask):
        assert imgnum in self.sel
        for i in range(len(self.canvas_array)):
            if self.sel[i] != imgnum:
                self.canvas_array[i].mask = ''
            else:
                self.canvas_array[i].mask = mask
                self.current_mask_item = i

    def toggle_highlight(self):
        self.query_vis = not self.query_vis
        if self.query_vis and not self.query_key_valid:
            self.update_query_key()
        self.repaint_canvas_array()

    def toggle_original(self):
        self.show_original = not self.show_original
        if self.show_original:
            self.original_btn.label = 'Toggle Changed'
        else:
            self.original_btn.label = 'Toggle Original'
        self.repaint_canvas_array()

    def query(self):
        if self.overwriting or 'query' not in self.request:
            if self.current_mask_item is None or not self.canvas_array[self.current_mask_item].mask:
                return
            imgnum = self.sel[self.current_mask_item]
            mask = self.canvas_array[self.current_mask_item].mask
            self.request['query'] = (imgnum, mask)
        self.exec_query()

    def update_query_key(self, key=None, no_repaint=False):
        if self.query_key_valid:
            return
        if key is None:
            if 'query' in self.request:
                imgnum, mask = self.request['query']
            else:
                if self.current_mask_item is None or not self.canvas_array[self.current_mask_item].mask:
                    return
                imgnum = self.sel[self.current_mask_item]
                mask = self.canvas_array[self.current_mask_item].mask
                self.request['query'] = (imgnum, mask)
            key = self.gw.query_key_from_selection(imgnum, mask)
        sel, rq = self.gw.ranking_for_key(key, k=len(self.canvas_array))
        self.query_rq = rq
        self.query_key = key
        self.query_key_valid = True

    def exec_query(self, key=None, no_repaint=False):
        if key is None:
            imgnum, mask = self.request['query']
            key = self.gw.query_key_from_selection(imgnum, mask)
        sel, rq = self.gw.ranking_for_key(key, k=len(self.canvas_array))
        self.sel = sel.tolist()
        self.query_rq = rq
        self.imgnum_textbox.value = ','.join(str(i) for i in self.sel)
        self.query_key = key
        self.query_key_units = key.sort(descending=True)[1][:10].tolist()
        self.query_out.clear()
        self.query_out.print('query: ', self.query_key_units)
        if not no_repaint:
            self.repaint_canvas_array()
            self.clear_images()
        self.show_msg('query done')

    def pick_object(self):
        if self.overwriting or 'object' not in self.request:
            if self.current_mask_item is None or not self.canvas_array[self.current_mask_item].mask:
                return
            imgnum = self.sel[self.current_mask_item]
            mask = self.canvas_array[self.current_mask_item].mask
            self.request['object'] = (imgnum, mask)
        self.exec_object()

    def exec_object(self, obj_acts=None, obj_area=None, obj_output=None,
                    bounds=None):
        if obj_acts is None:
            imgnum, mask = self.request['object']
            obj_acts, obj_output, obj_area, bounds = (
                self.gw.object_from_selection(imgnum, mask))
        if obj_area is None:
            obj_area = torch.ones(obj_acts.shape[2:], device=obj_acts.device)
            mask = None
        self.obj_acts, self.obj_area = obj_acts, obj_area
        cropped_out = self.gw.merge_target_output(obj_output, obj_acts, bounds)
        imgout = self.gw.render_object(cropped_out, obj_area)
        imgviz.ImageVisualizer((imgout.height, imgout.width))
        self.copy_canvas.image = renormalize.as_url(
            self.request_mask(thickness=3))
        self.copy_canvas.mask = mask
        self.show_msg('picked object')

    def request_mask(self, field='object', index=None, **kwargs):
        # For generating high-resolution figures: directly visualize a mask.
        if field not in self.request:
            print(f'No {field} selected')
            return
        if field == 'key':
            if index >= len(self.request[field]):
                print(f'No {index}th entry in key')
                return
            imgnum, mask = self.request[field][index]
        else:
            imgnum, mask = self.request[field]
        area = (renormalize.from_url(mask, target='pt',
                                     size=self.gw.x_shape[2:])[0] > 0.25)
        imgout = self.gw.render_image(imgnum, mask=area, **kwargs)
        return imgout

    def revert(self):
        with torch.no_grad():
            self.gw.model.load_state_dict(self.original_model.state_dict())
        self.repaint_canvas_array()
        # self.saved_list.value = ''
        self.show_msg('reverted to original')

    def paste(self):
        if self.obj_acts is None:
            self.show_msg('ERR: no activations')
            return
        if self.overwriting or 'paste' not in self.request:
            if self.current_mask_item is None:
                self.show_msg('ERR: no selected item')
                return
            if not self.canvas_array[self.current_mask_item].mask:
                self.show_msg(f'ERR: no mask for selected item at index {self.current_mask_item} (imgnum {self.sel[self.current_mask_item]})')
                return
            imgnum = self.sel[self.current_mask_item]
            mask = self.canvas_array[self.current_mask_item].mask
            self.request['paste'] = (imgnum, mask)
        self.exec_paste()

    def exec_paste(self):
        imgnum, mask = self.request['paste']
        goal_in, goal_out, viz_out, bounds = self.gw.paste_from_selection(
            imgnum, mask, self.obj_acts, self.obj_area)
        self.paste_canvas.image = renormalize.as_url(
            self.gw.render_object(viz_out, box=bounds))

    def exec_erase(self):
        if 'paste' not in self.request:
            self.show_msg('no paste goal selected...')
            return
        if 'key' not in self.request:
            self.show_msg('no context key selected...')
            return
        self.show_msg('erasing from model...')
        rank = int(self.rank_textbox.value)
        niter = int(self.paste_niter_textbox.value)
        piter = int(self.paste_piter_textbox.value)
        lr = float(self.paste_lr_textbox.value)

        def update_callback(it, loss):
            if it % 50 == 0 or it == niter - 1:
                loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                             f'\tloss {loss.item():.4f}')
                self.loss_out.print(loss_info, replace=True)
        self.gw.apply_erase(self.request,
                            rank=rank, drank=30, niter=niter, piter=piter, lr=lr,
                            update_callback=update_callback)
        self.repaint_canvas_array()
        self.show_msg(f'erased from model')

    def exec_request(self):
        if 'object' not in self.request:
            self.show_msg('no copy object selected...')
            return
        if 'paste' not in self.request:
            self.show_msg('no paste goal selected...')
            return
        self.show_msg('modifying model...')
        rank = int(self.rank_textbox.value)
        niter = int(self.paste_niter_textbox.value)
        piter = int(self.paste_piter_textbox.value)
        lr = float(self.paste_lr_textbox.value)

        def update_callback(it, loss):
            if it % 50 == 0 or it == niter - 1:
                loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                             f'\tloss {loss.item():.4f}')
                self.loss_out.print(loss_info, replace=True)
        self.gw.apply_edit(self.request,
                           rank=rank, niter=niter, piter=piter, lr=lr,
                           update_callback=update_callback)
        self.show_original = False
        self.repaint_canvas_array()
        self.show_msg(f'pasted into model')

    def exec_overfit(self):
        if 'object' not in self.request:
            self.show_msg('no copy object selected...')
            return
        if 'paste' not in self.request:
            self.show_msg('no paste goal selected...')
            return
        self.show_msg('overfitting model...')
        niter = int(self.paste_niter_textbox.value)
        lr = float(self.paste_lr_textbox.value)

        def update_callback(it, loss):
            if it % 50 == 0 or it == niter - 1:
                loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                             f'\tloss {loss.item():.4f}')
                self.loss_out.print(loss_info, replace=True)
        self.gw.apply_overfit(self.request,
                              niter=niter, lr=lr, update_callback=update_callback)
        self.repaint_canvas_array()
        self.show_msg(f'overfitted into model')

    def save(self):
        if not self.saved_list.value:
            return
        self.save_as_name(self.saved_list.value)
        self.saved_list.choices = self.saved_names()
        self.show_msg('saved as ' + self.saved_list.value)

    def tryload(self):
        if not self.saved_list.value:
            return
        if self.saved_list.value in self.saved_list.choices:
            self.show_msg('loading edit...')
            self.load_from_name(self.saved_list.value)
            self.show_msg('loaded from ' + self.saved_list.value + '; exec to execute model change.')

    def saved_names(self):
        os.makedirs(self.savedir, exist_ok=True)
        return sorted([name[:-5] for name in os.listdir(self.savedir)
                       if name.endswith('.json')])

    def save_as_name(self, name):
        data = self.request
        os.makedirs(self.savedir, exist_ok=True)
        with open(os.path.join(self.savedir, '%s.json' % name), 'w') as f:
            json.dump(data, f, indent=1)

    def load_from_name(self, name):
        os.makedirs(self.savedir, exist_ok=True)
        with open(os.path.join(self.savedir, '%s.json' % name), 'r') as f:
            self.request = json.load(f)
            self.repaint_key_tray()
            # if 'query' in self.request:
            #     self.exec_query()
            if 'object' in self.request:
                self.exec_object()
            # self.set_mask(*self.request['paste'])
            if 'paste' in self.request:
                self.exec_paste()

    def show_msg(self, msg):
        # self.msg_out.clear()
        self.msg_out.print(msg, replace=True)

    def widget_html(self):
        def h(w):
            return w._repr_html_()
        return f'''<div {self.std_attrs()}>
        <div>
        <style>
        .rwa input[type=button] {{ background: dimgray; color: white;
          border: 0; border-radius: 8px; padding: 5px 10px; font-size: 18px; }}
        .rwa-tray input[type=button] {{ background: #45009E; }}
        </style>
        <center><span style="font-size:24px;margin-right:24px;vertical-align:bottom;"
        >Rewriting a Deep Generative Model</span>
        {h(self.context_querybtn)}
        {h(self.original_btn)}
        {h(self.exec_btn)}
        </center>
        <div style="margin-top: 8px; margin-bottom: 8px;"><!-- middle row -->
        <hr style="border:1px solid gray; background-color: white">
        <div style="
          width:{(self.size + 2) * 1.5 + 20}px;
          display:inline-block;
          vertical-align:top"
          class="rwa-tray"><!-- context tray -->
        <div style="display:block;
          width:{(self.size + 2) * 1.5}px;
          vertical-align:top;
          "><!-- bottom of left -->
        <div style="display:block;
          padding-top:20px;
          padding-bottom:20px;
          vertical-align:top;
          text-align:center">{h(self.key_btn)} &nbsp; {h(self.highlight_btn)}
        </div>
        <div style="display:inline-block; background: #f2f2f2;">
        {h(self.context_out)}
        </div>
        </div><!-- contxt tray -->
        <hr style="border: 1px solid gray">
        <div style="display:block;
          text-align:center;vertical-align:top;"><!-- copy-and-paste tray -->
        <div style="display:inline-block; width:{self.size * 3 // 4 +2}px;
          padding-bottom:20px;
          text-align:center">
        {h(self.object_btn)}
        </div>
        <div style="display:inline-block; width:{self.size * 3 // 4 +2}px;
          padding-bottom:20px;
          text-align:center">
        {h(self.paste_btn)}
        </div>
        <div style="display:inline-block;
          width:{self.size * 0.75}px;
          height:{self.size * 0.75}px;
          vertical-align:top;
          text-align:center;
          background:#f2f2f2">{h(self.copy_canvas)}</div>
        <div style="display:inline-block;
          width:{self.size * 0.75}px;
          height:{self.size * 0.75}px;
          vertical-align:top;
          text-align:center;
          background:#f2f2f2">{h(self.paste_canvas)}</div>
        </div> <!-- copy-and-paste tray -->
        </div><!--left-->
        <!--right-->
        <div style="height:{(self.size + 2) * 2 + 50}px; width:{(self.size + 2) * 2 + 22 + 32}px;
          display: inline-block;
          vertical-align: top;
          border-left: 4px dashed gray;
          padding-left: 5px;
          margin-left: 5px;
          margin-top:8px; overflow-y: scroll">
        {show.html([[c] for c in self.canvas_array])}
        </div>
        </div>

        <div style="width:100%;">
        <hr style="border:1px solid gray; background-color: white">
        </div>

        <div style="width:100%; text-align: center;
           margin-top:8px;padding-top:30px;">
        Images {h(self.imgnum_textbox)}
        {h(self.query_btn)}
        {h(self.rank_textbox)}
        {h(self.paste_lr_textbox)}
        {h(self.revert_btn)}
        &nbsp;
        {h(self.saved_list)}
        {h(self.load_btn)}
        {h(self.save_btn)}
        </div>

        {h(self.loss_out)}
        {h(self.msg_out)}
        </div>'''


##########################################################################
# Utility functions
##########################################################################

def positive_bounding_box(data):
    pos = (data > 0)
    v, h = pos.sum(0).nonzero(), pos.sum(1).nonzero()
    left, right = v.min().item(), v.max().item()
    top, bottom = h.min().item(), h.max().item()
    return top, left, bottom + 1, right + 1


def centered_location(data):
    t, l, b, r = positive_bounding_box(data)
    return (t + b) // 2, (l + r) // 2


def paste_clip_at_center(source, clip, center, area=None):
    target = source.clone()
    # clip = clip[:,:,:target.shape[2],:target.shape[3]]
    t, l = (max(0, min(e - s, c - s // 2))
            for s, c, e in zip(clip.shape[2:], center, source.shape[2:]))
    b, r = t + clip.shape[2], l + clip.shape[3]
    # TODO: consider copying over a subset of channels.
    target[:, :, t:b, l:r] = clip if area is None else (
        (1 - area)[None, None, :, :].to(target.device) * target[:, :, t:b, l:r] + area[None, None, :, :].to(target.device) * clip)
    return target, (t, l, b, r)


def crop_clip_to_bounds(source, target, bounds):
    t, l, b, r = bounds
    vr, hr = [ts // ss for ts, ss in zip(target.shape[2:], source.shape[2:])]
    st, sl, sb, sr = t // vr, l // hr, -(-b // vr), -(-r // hr)
    tt, tl, tb, tr = st * vr, sl * hr, sb * vr, sr * hr
    cs, ct = source[:, :, st:sb, sl:sr], target[:, :, tt:tb, tl:tr]
    return cs, ct, (st, sl, sb, sr), (tt, tl, tb, tr)


def projected_conv(weight, direction):
    if len(weight.shape) == 5:
        cosine_map = torch.einsum('goiyx, di -> godyx', weight, direction)
        result = torch.einsum('godyx, di -> goiyx', cosine_map, direction)
    else:
        cosine_map = torch.einsum('oiyx, di -> odyx', weight, direction)
        result = torch.einsum('odyx, di -> oiyx', cosine_map, direction)
    return result


def rank_one_conv(weight, direction):
    cosine_map = (weight * direction[None, :, None, None]).sum(1, keepdim=True)
    return cosine_map * direction[None, :, None, None]


def zca_from_cov(cov):
    evals, evecs = torch.symeig(cov.double(), eigenvectors=True)
    zca = torch.mm(torch.mm(evecs, torch.diag
                            (evals.sqrt().clamp(1e-20).reciprocal())),
                   evecs.t()).to(cov.dtype)
    return zca
