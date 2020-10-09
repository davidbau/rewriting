"""
labwidget by David Bau.

Base class for a lightweight javascript notebook widget framework
that is portable across Google colab and Jupyter notebooks.
No use of requirejs: the design uses all inline javascript.

Defines Model, Widget, Trigger, and Property, which set up data binding
using the communication channels available in either google colab
environment or jupyter notebook.

This module also defines Label, Textbox, Range, Choice, and Div
widgets; the code for these are good examples of usage of Widget,
Trigger, and Property objects.

Within HTML widgets, user interaction should update the javascript
model using model.set('propname', value); this will propagate to
the python model and notify any registered python listeners; similarly
model.on('propname', callback) will listen for property changes
that come from python.

MIT LICENSE

Copyright 2020 David Bau

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

"""

import json
import html
import re
from inspect import signature


class Model(object):
    '''
    Abstract base class that supports data binding.  Within __init__,
    a model subclass defines databound events and properties using:

       self.evtname = Trigger()
       self.propname = Property(initval)

    Any Trigger or Property member can be watched by registering a
    listener with `model.on('propname', callback)`.

    An event can be triggered by `model.evtname.trigger(value)`.
    A property can be read with `model.propname`, and can be set by
    `model.propname = value`; this also triggers notifications.
    In both these cases, any registered listeners will be called
    with the given value.
    '''

    def on(self, name, cb, internal=False):
        '''
        Registers a listener for named events and properties.
        A space-separated list of names can be provided as `name`.
        '''
        for n in name.split():
            self.prop(n).on(cb, internal=internal)
        return self

    def off(self, name, cb=None):
        '''
        Unregisters a listener for named events and properties.
        A space-separated list of names can be provided as `name`.
        '''
        for n in name.split():
            self.prop(n).off(cb)
        return self

    def prop(self, name):
        '''
        Returns the underlying Trigger or Property object for a
        property, rather than its held value.
        '''
        curvalue = super().__getattribute__(name)
        if not isinstance(curvalue, Trigger):
            raise AttributeError('%s not a property or trigger but %s'
                                 % (name, str(type(curvalue))))
        return curvalue

    def _initprop_(self, name, value):
        '''
        To be overridden in base classes.  Handles initialization of
        a new Trigger or Property member.
        '''
        value.name = name
        value.target = self
        return

    def __setattr__(self, name, value):
        '''
        When a member is an Trigger or Property, then assignment notation
        is delegated to the Trigger or Property so that notifications
        and reparenting can be handled.  That is, `model.name = value`
        turns into `prop(name).set(value)`.
        '''
        if hasattr(self, name):
            curvalue = super().__getattribute__(name)
            if isinstance(curvalue, Trigger):
                # Delegte "set" to the underlying Property.
                curvalue.set(value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)
            if isinstance(value, Trigger):
                self._initprop_(name, value)

    def __getattribute__(self, name):
        '''
        When a member is a Property, then property getter
        notation is delegated to the peoperty object.
        '''
        curvalue = super().__getattribute__(name)
        if isinstance(curvalue, Property):
            return curvalue.value
        return curvalue


class Widget(Model):
    '''
    Base class for an HTML widget that uses a Javascript model object
    to syncrhonize HTML view state with the backend Python model state.
    Each widget subclass overrides widget_js to provide Javascript code
    that defines the widget's behavior.  This javascript will be wrapped
    in an immediately-invoked function and included in the widget's HTML
    representation (_repr_html_) when the widget is viewed.

    A widget's javascript is provided with two local variables:

       element - the widget's root HTML element.  By default this is
                 a <div> but can be overridden in widget_html.
       model   - the object representing the data model for the widget.
                 within javascript.

    The model object provides the following javascript API:

       model.get('propname') obtains a current property value.
       model.set('propname', 'value') requests a change in value.
       model.on('propname', callback) listens for property changes.
       model.trigger('evtname', value) triggers an event.

    Note that model.set just requests a change but does not change the
    value immediately: model.get will not reflect the change until the
    python backend has handled it and notified the javascript of the new
    value, which will trigger any callbacks previously registered using
    .on('propname', callback).  Thus Widget impelements a V-shaped
    notification protocol:

    User entry ->                 |              -> User-visible feedback
        js model.set ->           |        -> js.model.on  callback
          python prop.trigger ->  |   -> python prop.notify
                         python prop.handle

    Finally, all widgets provide standard databinding for style and data
    properties, which are write-only (python-to-js) properties that
    let python directly control CSS styles and HTML dataset attributes
    for the top-level widget element.
    '''

    def __init__(self, style=None, data=None, className=None):
        # In the jupyter case, there can be some delay between js injection
        # and comm creation, so we need to queue some initial messages.
        if WIDGET_ENV == 'jupyter':
            self._comms = []
            self._queue = []
        # Each call to _repr_html_ creates a unique view instance.
        self._viewcount = 0
        # Python notification is handled by Property objects.

        def handle_remote_set(name, value):
            with capture_output(self):  # make errors visible.
                self.prop(name).trigger(value)
        self._recv_from_js_(handle_remote_set)
        # The style and data properties come standard, and are used to
        # control the style and data attributes on the toplevel element.
        self.style = Property(style)
        self.className = Property(className)
        self.data = Property(data)
        # Each widget has a "write" event that is used to insert
        # html before the widget.
        self.write = Trigger()

    def set_style(self, k, v=None):
        d = self.style or {}
        if v is None and isinstance(k, dict):
            d.update(k)
        else:
            d[k] = v
        self.style = d

    def widget_js(self):
        '''
        Override to define the javascript logic for the widget.  Should
        render the initial view based on the current model state (if not
        already rendered using widget_html) and set up listeners to keep
        the model and the view synchronized.
        '''
        return ''

    def widget_html(self):
        '''
        Override to define the initial HTML view of the widget.  Should
        define an element with id given by view_id().
        '''
        return f'<div {self.std_attrs()}></div>'

    def view_id(self):
        '''
        Returns an HTML element id for the view currently being rendered.
        Note that each time _repr_html_ is called, this id will change.
        '''
        return f"_{id(self)}_{self._viewcount}"

    def std_attrs(self):
        '''
        Returns id and (if applicable) style attributes, escaped and
        formatted for use within the top-level element of widget HTML.
        '''
        return (f'id="{self.view_id()}"' +
                style_attr(self.style) +
                class_attr(self.className) +
                data_attrs(self.data))

    def _repr_html_(self):
        '''
        Returns the HTML code for the widget.
        '''
        self._viewcount += 1
        json_data = json.dumps({
            k: v.value for k, v in vars(self).items()
            if isinstance(v, Property)})
        json_data = re.sub('</', '<\\/', json_data)

        std_widget_js = minify(f'''
          var model = new Model("{id(self)}", {json_data});
          var element = document.getElementById("{self.view_id()}");
          model.on('write', (ev) => {{
            var dummy = document.createElement('div');
            dummy.innerHTML = ev.value.trim();
            dummy.childNodes.forEach((item) => {{
              element.parentNode.insertBefore(item, element);
            }});
          }});
          function upd(a) {{ return (e) => {{ for (k in e.value) {{
            element[a][k] = e.value[k];
          }}}}}}
          model.on('style', upd('style'));
          model.on('style', upd('style'));
          model.on('data', upd('dataset'));
        ''')

        return ''.join([
            self.widget_html(),
            '<script>(function() {',
            WIDGET_MODEL_JS,
            std_widget_js,
            self.widget_js(),
            '})();</script>'
        ])

    def _initprop_(self, name, value):
        if not hasattr(self, '_viewcount'):
            raise ValueError('base Model __init__ must be called')
        super()._initprop_(name, value)

        def notify_js(event):
            self._send_to_js_(id(self), name, event.value)
        if isinstance(value, Trigger):
            value.on(notify_js, internal=True)

    def _send_to_js_(self, *args):
        if self._viewcount > 0:
            if WIDGET_ENV == 'colab':
                colab_output.eval_js(minify(f"""
                (window.send_{id(self)} = window.send_{id(self)} ||
                new BroadcastChannel("channel_{id(self)}")
                ).postMessage({json.dumps(args)});
                """), ignore_result=True)
            elif WIDGET_ENV == 'jupyter':
                if not self._comms:
                    self._queue.append(args)
                    return
                for comm in self._comms:
                    comm.send(args)

    def _recv_from_js_(self, fn):
        if WIDGET_ENV == 'colab':
            colab_output.register_callback(f"invoke_{id(self)}", fn)
        elif WIDGET_ENV == 'jupyter':
            def handle_comm(msg):
                fn(*(msg['content']['data']))
                # TODO: handle closing also.

            def handle_close(close_msg):
                comm_id = close_msg['content']['comm_id']
                self._comms = [c for c in self._comms if c.comm_id != comm_id]

            def open_comm(comm, open_msg):
                self._comms.append(comm)
                comm.on_msg(handle_comm)
                comm.on_close(handle_close)
                comm.send('ok')
                if self._queue:
                    for args in self._queue:
                        comm.send(args)
                    self._queue.clear()
                if open_msg['content']['data']:
                    handle_comm(open_msg)
            cname = "comm_" + str(id(self))
            COMM_MANAGER.register_target(cname, open_comm)

    def display(self):
        from IPython.core.display import display
        display(self)
        return self


class Trigger(object):
    """
    Trigger is the base class for Property and other data-bound
    field objects.  Trigger holds a list of listeners that need to
    be notified about the event.

    Multple Trigger objects can be tied (typically a parent Model can
    have Triggers that are triggered by children models).  To support
    this, each Trigger can have a parent.

    Trigger objects provide a notification protocol where view
    interactions trigger events at a leaf that are sent up to the
    root Trigger to be handled.  By default, the root handler accepts
    events by notifying all listeners and children in the tree.
    """

    def __init__(self):
        self._listeners = []
        self.parent = None
        # name and target are set in Model._initprop_.
        self.name = None
        self.target = None

    def handle(self, value):
        '''
        Method to override; called at the root when an event has been
        triggered, and on a child when the parent has notified.  By
        default notifies all listeners.
        '''
        # Allow direct delegation to a handler
        if isinstance(value, Event):
            value = value.value
        self.notify(value)

    def trigger(self, value=None):
        '''
        Triggers an event to be handled by the root.  By default, the root
        handler will accept the event so all the listeners will be notified.
        '''
        if self.parent is not None:
            self.parent.trigger(value)
        else:
            self.handle(value)

    def set(self, value):
        '''
        Sets the parent Trigger.  Child Triggers trigger events by
        triggering parents, and in turn they handle notifications
        that come from parents.
        '''
        if self.parent is not None:
            self.parent.off(self.handle)
            self.parent = None
        if isinstance(value, Trigger):
            ancestor = value.parent
            while ancestor is not None:
                if ancestor == self:
                    raise ValueError('bound properties should not form a loop')
                ancestor = ancestor.parent
            self.parent = value
            self.parent.on(self.handle, internal=True)
        elif not isinstance(self, Property):
            raise ValueError('only properties can be set to a value')

    def notify(self, value=None):
        '''
        Notifies listeners and children.  If a listener accepts an argument,
        the value will be passed as a single argument.
        '''
        for cb, internal in self._listeners:
            with block_events(self.name, cb, internal) as ctx:
                if ctx.silence:
                    # do not notify recursively...
                    # print(f'silenced recursive {self.name} {cb.__name__}')
                    pass
                elif len(signature(cb).parameters) == 0:
                    cb()  # no-parameter callback.
                else:
                    cb(Event(value, self.name, self.target))

    def on(self, cb, internal=False):
        '''
        Registers a listener.  Calling multiple times registers
        multiple listeners.
        '''
        self._listeners.append((cb, internal))

    def off(self, cb=None):
        '''
        Unregisters a listener.
        '''
        self._listeners = [(c, i) for c, i in self._listeners
                           if c != cb and cb is not None]


class Property(Trigger):
    """
    A Property is just an Trigger that remembers its last value.
    """

    def __init__(self, value=None):
        '''
        Can be initialized with a starting value.
        '''
        super().__init__()
        self.set(value)

    def handle(self, value):
        '''
        The default handling for a Property is to store the value,
        then notify listeners.  This method can be overridden,
        for example to validate values.
        '''
        # Allow direct delegation to a handler
        if isinstance(value, Event):
            value = value.value
        self.value = value
        self.notify(value)

    def set(self, value):
        '''
        When a Property value is set to an ordinary value, it
        triggers an event which causes a notification to be
        sent to update all linked Properties.  A Property set
        to another Property becomes a child of the value.
        '''
        # Handle setting a parent Property
        if isinstance(value, Property):
            super().set(value)
            self.handle(value.value)
        elif isinstance(value, Trigger):
            raise ValueError('Cannot set a Property to an Trigger')
        else:
            self.trigger(value)


class Event(object):
    def __init__(self, value, name, target, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.value = value
        self.name = name
        self.target = target


entered_handler_stack = []

class block_events(object):
    def __init__(self, name='block_events', cb=None, internal=False):
        global entered_handler_stack
        self.internal = internal
        self.cb = cb
        self.name = name
        if not internal:
            self.silence = any(
                (not s.internal)
                for s in entered_handler_stack)
        else:
            self.silence = any(
                (s.cb == cb and s.name == name)
                for s in entered_handler_stack)

    def __enter__(self):
        global entered_handler_stack
        entered_handler_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        global entered_handler_stack
        entered_handler_stack.pop()


class capture_output(object):
    """Context manager for capturing stdout/stderr.  This is used,
    by default, to wrap handler code that is invoked by a triggering
    event coming from javascript.  Any stdout/stderr or exceptions
    that are thrown are formatted and written above the relevant widget."""

    def __init__(self, widget):
        from io import StringIO
        self.widget = widget
        self.buffer = StringIO()

    def __enter__(self):
        import sys
        self.saved = dict(stdout=sys.stdout, stderr=sys.stderr)
        sys.stdout = self.buffer
        sys.stderr = self.buffer

    def __exit__(self, exc_type, exc_value, exc_tb):
        import sys
        import traceback
        captured = self.buffer.getvalue()
        if len(captured):
            self.widget.write.trigger(f'<pre>{html.escape(captured)}</pre>')
        if exc_type:
            import traceback
            tbtxt = ''.join(
                    traceback.format_exception(exc_type, exc_value, exc_tb))
            self.widget.write.trigger(
                f'<pre style="color:red;text-align:left">{tbtxt}</pre>')
        sys.stdout = self.saved['stdout']
        sys.stderr = self.saved['stderr']


##########################################################################
# Specific widgets
##########################################################################

class Button(Widget):
    def __init__(self, label='button', style=None, **kwargs):
        super().__init__(style=defaulted(style, display='block'), **kwargs)
        self.click = Trigger()
        self.label = Property(label)

    def widget_js(self):
        return minify('''
          element.addEventListener('click', (e) => {
            model.trigger('click');
          })
          model.on('label', (ev) => {
            element.value = ev.value;
          })
        ''')

    def widget_html(self):
        return f'''<input {self.std_attrs()} type="button" value="{
            html.escape(str(self.label))}">'''


class Label(Widget):
    def __init__(self, value='', **kwargs):
        super().__init__(**kwargs)
        # databinding is defined using Property objects.
        self.value = Property(value)

    def widget_js(self):
        # Both "model" and "element" objects are defined within the scope
        # where the js is run.    "element" looks for the element with id
        # self.view_id(); if widget_html is overridden, this id should be used.
        return minify('''
            model.on('value', (ev) => {
                element.innerText = model.get('value');
            });
        ''')

    def widget_html(self):
        return f'''<label {self.std_attrs()}>{
            html.escape(str(self.value))}</label>'''


class Textbox(Widget):
    def __init__(self, value='', size=20, style=None, desc=None, **kwargs):
        super().__init__(style=defaulted(style, display='inline-block'), **kwargs)
        # databinding is defined using Property objects.
        self.value = Property(value)
        self.size = Property(size)
        self.desc = Property(desc)

    def widget_js(self):
        # Both "model" and "element" objects are defined within the scope
        # where the js is run.    "element" looks for the element with id
        # self.view_id(); if widget_html is overridden, this id should be used.
        return minify('''
          element.value = model.get('value');
          element.size = model.get('size');
          element.addEventListener('keydown', (e) => {
            if (e.code == 'Enter') {
              model.set('value', element.value);
            }
          });
          element.addEventListener('blur', (e) => {
            model.set('value', element.value);
          });
          model.on('value', (ev) => {
            element.value = model.get('value');
          });
          model.on('size', (ev) => {
            element.size = model.get('size');
          });
        ''')

    def widget_html(self):

        html_str = f'''<input {self.std_attrs()} value="{
            html.escape(str(self.value))}" size="{self.size}">'''
        if self.desc is not None:
            html_str = f"""<span>{self.desc}</span>{html_str}"""
        return html_str


class Textarea(Widget):
    def __init__(self, value='', rows=2, cols=20, **kwargs):
        super().__init__(**kwargs)
        # databinding is defined using Property objects.
        self.value = Property(value)
        self.rows = Property(rows)
        self.cols = Property(cols)

    def widget_js(self):
        # Both "model" and "element" objects are defined within the scope
        # where the js is run.    "element" looks for the element with id
        # self.view_id(); if widget_html is overridden, this id should be used.
        return minify('''
          element.value = model.get('value');
          element.rows = model.get('rows');
          element.cols = model.get('cols');
          element.addEventListener('blur', (e) => {
            model.set('value', element.value);
          });
          model.on('value', (ev) => {
            element.value = model.get('value');
          });
          model.on('rows', (ev) => {
            element.rows = model.get('rows');
          });
          model.on('cols', (ev) => {
            element.cols = model.get('cols');
          });
        ''')

    def widget_html(self):
        return f'''<textarea {self.std_attrs()} rows="{
            self.rows}" cols="{self.cols}">{
            html.escape(str(self.value))}"</textarea>'''

class Range(Widget):
    def __init__(self, value=50, min=0, max=100, **kwargs):
        super().__init__(**kwargs)
        # databinding is defined using Property objects.
        self.value = Property(value)
        self.min = Property(min)
        self.max = Property(max)

    def widget_js(self):
        # Note that the 'input' event would enable during-drag feedback,
        # but this is pretty slow on google colab.
        return minify('''
          element.addEventListener('change', (e) => {
            model.set('value', element.value);
          });
          model.on('value', (e) => {
            if (!element.matches(':active')) {
              element.value = e.value;
            }
          })
        ''')

    def widget_html(self):
        return f'''<input {self.std_attrs()} type="range" value="{
            self.value}" min="{self.min}" max="{self.max}">'''


class Choice(Widget):
    """
    A set of radio button choices.
    """

    def __init__(self, choices=None, selection=None, horizontal=False,
                 **kwargs):
        super().__init__(**kwargs)
        if choices is None:
            choices = []
        self.choices = Property(choices)
        self.horizontal = Property(horizontal)
        self.selection = Property(selection)

    def widget_js(self):
        # Note that the 'input' event would enable during-drag feedback,
        # but this is pretty slow on google colab.
        return minify('''
          function esc(unsafe) {
            return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;")
                   .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
          }
          function render() {
            var lines = model.get('choices').map((c) => {
              return '<label><input type="radio" name="choice" value="' +
                 esc(c) + '">' + esc(c) + '</label>'
            });
            element.innerHTML = lines.join(model.get('horizontal')?' ':'<br>');
          }
          model.on('choices horizontal', render);
          model.on('selection', (ev) => {
            [...element.querySelectorAll('input')].forEach((e) => {
              e.checked = (e.value == ev.value);
            })
          });
          element.addEventListener('change', (e) => {
            model.set('selection', element.choice.value);
          });
        ''')

    def widget_html(self):
        radios = [
            f"""<label><input name="choice" type="radio" {
            'checked' if value == self.selection else ''
            } value="{html.escape(value)}">{html.escape(value)}</label>"""
            for value in self.choices]
        sep = " " if self.horizontal else "<br>"
        return f'<form {self.std_attrs()}>{sep.join(radios)}</form>'


class Menu(Widget):
    """
    A dropdown choice.
    """

    def __init__(self, choices=None, selection=None, **kwargs):
        super().__init__(**kwargs)
        if choices is None:
            choices = []
        self.choices = Property(choices)
        self.selection = Property(selection)

    def widget_js(self):
        return minify('''
          function esc(unsafe) {
            return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;")
                   .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
          }
          function render() {
            var selection = model.get('selection');
            var lines = model.get('choices').map((c) => {
              return '<option value="' + esc(''+c) + '"' +
                     (c == selection ? ' selected' : '') +
                     '>' + esc(''+c) + '</option>';
            });
            element.menu.innerHTML = lines.join('\\n');
          }
          model.on('choices horizontal', render);
          model.on('selection', (ev) => {
            [...element.querySelectorAll('option')].forEach((e) => {
              e.selected = (e.value == ev.value);
            })
          });
          element.addEventListener('change', (e) => {
            model.set('selection', element.menu.value);
          });
        ''')

    def widget_html(self):
        options = [
            f"""<option value="{html.escape(str(value))}" {
            'selected' if value == self.selection else ''
            }>{html.escape(str(value))}</option>"""
            for value in self.choices]
        sep = "\n"
        return f'''<form {self.std_attrs()}"><select name="menu">{
             sep.join(options)}</select></form>'''


class Datalist(Widget):
    """
    An input with a dropdown choice.
    """

    def __init__(self, choices=None, value=None, **kwargs):
        super().__init__(**kwargs)
        if choices is None:
            choices = []
        self.choices = Property(choices)
        self.value = Property(value)

    def datalist_id(self):
        return self.view_id() + '-dl'

    def widget_js(self):
        # The mousedown/mouseleave dance defeats the prefix-matching behavior
        # of the built-in datalist by erasing value momentarily on mousedown.
        return minify('''
          function esc(unsafe) {
            return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;")
                   .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
          }
          function render() {
            var lines = model.get('choices').map((c) => {
              return '<option value="' + esc(''+c) + '">';
            });
            element.inp.list.innerHTML = lines.join('\\n');
          }
          model.on('choices', render);
          model.on('value', (ev) => {
            element.inp.value = ev.value;
          });
          function restoreValue() {
            var inp = element.inp;
            if (inp.value == '') {
              inp.value = inp.placeholder;
              inp.placeholder = '';
            }
          }
          element.inp.addEventListener('mousedown', (e) => {
            var inp = element.inp;
            if (inp.value != '') {
              inp.placeholder = inp.value;
              inp.value = '';
              if (e.clientX < inp.getBoundingClientRect().right - 25) {
                setTimeout(restoreValue, 0);
              }
            }
          });
          element.inp.addEventListener('mouseleave', restoreValue)
          element.inp.addEventListener('change', (e) => {
            model.set('value', element.inp.value);
          });
        ''')

    def widget_html(self):
        options = [
            f"""<option value="{html.escape(str(value))}">"""
            for value in self.choices]
        return ''.join([
            f'<form {self.std_attrs()} onsubmit="return false;">',
            f'<input name="inp" list="{self.datalist_id()}" autocomplete="off">',
            f'<datalist id="{self.datalist_id()}">',
            ''.join(options),
            f'</datalist></form>'])


class Div(Widget):
    """
    Just an empty DIV element.  Use the innerHTML property to
    change its contents, or use the clear() and print() method.
    """

    def __init__(self, innerHTML='', **kwargs):
        super().__init__(**kwargs)
        # TODO: unify more closely with the show() library.
        self.innerHTML = Property(innerHTML)

    def clear(self):
        """Clears the contents of the div."""
        self.innerHTML = ''

    def show(self, *args):
        from . import show
        self.innerHTML = show.html(args)

    def print(self, *args, replace=False):
        """Appends plain text (as a pre) into the div."""
        newHTML = '<pre>%s</pre>' % ' '.join(
            html.escape(str(text)) for text in args)
        if replace:
            self.innerHTML = newHTML
        else:
            self.innerHTML += newHTML

    def widget_js(self):
        # Note that if we want innerHTML to support script execution,
        # we need to do it explicitly, like this.
        return minify('''
          model.on('innerHTML', (ev) => {
            element.innerHTML = ev.value;
            Array.from(element.querySelectorAll("script")).forEach(old=>{
              const newScript = document.createElement("script");
              Array.from(old.attributes).forEach(attr =>
                 newScript.setAttribute(attr.name, attr.value));
              newScript.appendChild(document.createTextNode(old.innerHTML));
              old.parentNode.replaceChild(newScript, old);
            });
          });
        ''')

    def widget_html(self):
        return f'''<div {self.std_attrs()}>{self.innerHTML}</div>'''


class ClickDiv(Div):
    '''
    A Div that triggers click events when anything inside them is clicked.
    If a clicked element contains a data-click value, then that value is
    sent as the click event value.
    '''

    def __init__(self, innerHTML='', **kwargs):
        super().__init__(innerHTML, **kwargs)
        self.click = Trigger()

    def widget_js(self):
        return super().widget_js() + minify('''
          element.addEventListener('click', (ev) => {
            var target = ev.target;
            while (target && target != element && !target.dataset.click) {
              target = target.parentElement;
            }
            var value = target.dataset.click;
            model.trigger('click', value);
          });
        ''')


class Image(Widget):
    """
    Just a IMG element.  Use the src property to change its contents by url,
    or use the clear() and render(imgdata) methods to convert PIL or
    tensor image data to a url to display.
    """

    def __init__(self, src='', style=None, **kwargs):
        super().__init__(style=defaulted(style, margin=0), **kwargs)
        self.src = Property(src)
        self.click = Trigger()

    def clear(self):
        """Clears the image."""
        self.src = ''

    def render(self, imgdata, source=None):
        """Converts a pil image or some tensor to a url to show inline."""
        from . import renormalize
        self.src = renormalize.as_url(imgdata, source=source)

    def widget_js(self):
        return minify('''
          model.on('src', (ev) => { element.src = ev.value; });
          element.addEventListener('click', (ev) => {
            model.trigger('click');
          });
        ''')

    def widget_html(self):
        return f'''<img {self.std_attrs()} src="{html.escape(self.src)}">'''

##########################################################################
# Utils
##########################################################################


def minify(t):
    # TODO: plug in some more real minification.
    return re.sub(r'\n\s*', '\n', t)


def style_attr(d):
    if not d:
        return ''
    return ' style="%s"' % html.escape(css_style_from_dict(d))


def class_attr(d):
    if not d:
        return ''
    return ' class="%s"' % d


def data_attrs(d):
    if not d:
        return ''
    return ''.join([
        ' data-%s="%s"' % (k, html.escape(str(v))) for k, v in d.items()])


def css_style_from_dict(d):
    # escape punctuation.  (but not #, which is used in colors)
    return ';'.join(
        re.sub('([A-Z]+)', r'-\1', k).lower() + ':' +
        re.sub('([][\\!"$%&\'()*+,./:;<=>?@^`{|}~])', r'\\\1', str(v))
        for k, v in d.items())


def defaulted(d, **kwargs):
    if d is None:
        return kwargs
    result = dict(kwargs)
    result.update(d)
    return result

##########################################################################
# Implementation Details
##########################################################################


WIDGET_ENV = None
if WIDGET_ENV is None:
    try:
        from google.colab import output as colab_output
        WIDGET_ENV = 'colab'
    except Exception as e:
        pass
if WIDGET_ENV is None:
    try:
        from ipykernel.comm import Comm as jupyter_comm
        COMM_MANAGER = get_ipython().kernel.comm_manager
        WIDGET_ENV = 'jupyter'
    except Exception as e:
        pass
if WIDGET_ENV is None:
    print('Neither colab nor jupyter environment found.')

SEND_RECV_JS = """
function recvFromPython(obj_id, fn) {
  var recvname = "recv_" + obj_id;
  if (window[recvname] === undefined) {
    window[recvname] = new BroadcastChannel("channel_" + obj_id);
  }
  window[recvname].addEventListener("message", (ev) => {
    if (ev.data == 'ok') {
      window[recvname].ok = true;
      return;
    }
    fn.apply(null, ev.data.slice(1));
  });
}
function sendToPython(obj_id, ...args) {
  google.colab.kernel.invokeFunction('invoke_' + obj_id, args, {})
}
""" if WIDGET_ENV == 'colab' else """
function getChan(obj_id) {
  var cname = "comm_" + obj_id;
  if (!window[cname]) { window[cname] = []; }
  var chan = window[cname];
  if (!chan.comm && Jupyter.notebook.kernel) {
    chan.comm = Jupyter.notebook.kernel.comm_manager.new_comm(cname, {});
    chan.comm.on_msg((ev) => {
      if (chan.retry) { clearInterval(chan.retry); chan.retry = null; }
      if (ev.content.data == 'ok') { return; }
      var args = ev.content.data.slice(1);
      for (fn of chan) { fn.apply(null, args); }
    });
    chan.retries = 5;
    chan.retry = setInterval(() => {
      if (chan.retries) { chan.retries -= 1; chan.comm.open(); }
      else { clearInterval(chan.retry); chan.retry = null; }
    }, 2000);
  }
  return chan;
}
function recvFromPython(obj_id, fn) {
  getChan(obj_id).push(fn);
}
function sendToPython(obj_id, ...args) {
  var comm = getChan(obj_id).comm;
  if (comm) { comm.send(args); }
}
"""


WIDGET_MODEL_JS = minify(SEND_RECV_JS + """
class Model {
  constructor(obj_id, init) {
    this._id = obj_id;
    this._listeners = {};
    this._data = Object.assign({}, init)
    recvFromPython(this._id, (name, value) => {
      this._data[name] = value;
      var e = new Event(name); e.value = value;
      if (this._listeners.hasOwnProperty(name)) {
        this._listeners[name].forEach((fn) => { fn(e); });
      }
    })
  }
  trigger(name, value) {
    sendToPython(this._id, name, value);
  }
  get(name) {
    return this._data[name];
  }
  set(name, value) {
    this.trigger(name, value);
  }
  on(name, fn) {
    name.split(/\s+/).forEach((n) => {
      if (!this._listeners.hasOwnProperty(n)) {
        this._listeners[n] = [];
      }
      this._listeners[n].push(fn);
    });
  }
  off(name, fn) {
    name.split(/\s+/).forEach((n) => {
      if (!fn) {
        delete this._listeners[n];
      } else if (this._listeners.hasOwnProperty(n)) {
        this._listeners[n] = this._listeners[n].filter(
            (e) => { return e !== fn; });
      }
    });
  }
}
""")
