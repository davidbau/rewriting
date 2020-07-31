'''
Utilities for showing progress bars, controlling default verbosity, etc.
'''

# If the tqdm package is not available, then do not show progress bars;
# just connect print_progress to print.
import sys
import types
import builtins
try:
    from tqdm import tqdm
    try:
        from tqdm.notebook import tqdm as tqdm_nb
    except:
        from tqdm import tqdm_notebook as tqdm_nb
except:
    tqdm = None

default_verbosity = True
next_description = None
python_print = builtins.print


def post(**kwargs):
    '''
    When within a progress loop, pbar.post(k=str) will display
    the given k=str status on the right-hand-side of the progress
    status bar.  If not within a visible progress bar, does nothing.
    '''
    innermost = innermost_tqdm()
    if innermost is not None:
        innermost.set_postfix(**kwargs)


def desc(desc):
    '''
    When within a progress loop, pbar.desc(str) changes the
    left-hand-side description of the loop toe the given description.
    '''
    innermost = innermost_tqdm()
    if innermost is not None:
        innermost.set_description(str(desc))


def descnext(desc):
    '''
    Called before starting a progress loop, pbar.descnext(str)
    sets the description text that will be used in the following loop.
    '''
    global next_description
    if not default_verbosity or tqdm is None:
        return
    next_description = desc


def print(*args):
    '''
    When within a progress loop, will print above the progress loop.
    '''
    global next_description
    next_description = None
    if default_verbosity:
        msg = ' '.join(str(s) for s in args)
        if tqdm is None:
            python_print(msg)
        else:
            tqdm.write(msg)


def tqdm_terminal(it, *args, **kwargs):
    '''
    Some settings for tqdm that make it run better in resizable terminals.
    '''
    return tqdm(it, *args, dynamic_ncols=True, ascii=True,
                leave=(innermost_tqdm() is not None), **kwargs)


def in_notebook():
    '''
    True if running inside a Jupyter notebook.
    '''
    # From https://stackoverflow.com/a/39662359/265298
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def innermost_tqdm():
    '''
    Returns the innermost active tqdm progress loop on the stack.
    '''
    if hasattr(tqdm, '_instances') and len(tqdm._instances) > 0:
        return max(tqdm._instances, key=lambda x: x.pos)
    else:
        return None


def reporthook(*args, **kwargs):
    '''
    For use with urllib.request.urlretrieve.

    with pbar.reporthook() as hook:
        urllib.request.urlretrieve(url, filename, reporthook=hook)
    '''
    kwargs2 = dict(unit_scale=True, miniters=1)
    kwargs2.update(kwargs)
    bar = __call__(None, *args, **kwargs2)

    class ReportHook(object):
        def __init__(self, t):
            self.t = t

        def __call__(self, b=1, bsize=1, tsize=None):
            if hasattr(self.t, 'total'):
                if tsize is not None:
                    self.t.total = tsize
            if hasattr(self.t, 'update'):
                self.t.update(b * bsize - self.t.n)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            if hasattr(self.t, '__exit__'):
                self.t.__exit__(*exc)
    return ReportHook(bar)


def __call__(x, *args, **kwargs):
    '''
    Invokes a progress function that can wrap iterators to print
    progress messages, if verbose is True.

    If verbose is False or tqdm is unavailable, then a quiet
    non-printing identity function is used.

    verbose can also be set to a spefific progress function rather
    than True, and that function will be used.
    '''
    global default_verbosity, next_description
    if not default_verbosity or tqdm is None:
        return x
    if default_verbosity == True:
        fn = tqdm_nb if in_notebook() else tqdm_terminal
    else:
        fn = default_verbosity
    if next_description is not None:
        kwargs = dict(kwargs)
        kwargs['desc'] = next_description
        next_description = None
    return fn(x, *args, **kwargs)


class VerboseContextManager():
    def __init__(self, v, entered=False):
        self.v, self.entered, self.saved = v, False, []
        if entered:
            self.__enter__()
            self.entered = True

    def __enter__(self):
        global default_verbosity
        if self.entered:
            self.entered = False
        else:
            self.saved.append(default_verbosity)
            default_verbosity = self.v
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global default_verbosity
        default_verbosity = self.saved.pop()

    def __call__(self, v=True):
        '''
        Calling the context manager makes a new context that is
        pre-entered, so it works as both a plain function and as a
        factory for a context manager.
        '''
        new_v = v if self.v else not v
        cm = VerboseContextManager(new_v, entered=True)
        default_verbosity = new_v
        return cm


# Use as either "with pbar.verbose:" or "pbar.verbose(False)", or also
# "with pbar.verbose(False):"
verbose = VerboseContextManager(True)

# Use as either "with @pbar.quiet" or "pbar.quiet(True)". or also
# "with pbar.quiet(True):"
quiet = VerboseContextManager(False)


class CallableModule(types.ModuleType):
    def __init__(self):
        # or super().__init__(__name__) for Python 3
        types.ModuleType.__init__(self, __name__)
        self.__dict__.update(sys.modules[__name__].__dict__)

    def __call__(self, x, *args, **kwargs):
        return __call__(x, *args, **kwargs)


sys.modules[__name__] = CallableModule()
