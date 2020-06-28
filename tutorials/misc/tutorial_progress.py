class TutorialBar(object):
    def __init__(self, max_iter):
        self.current_iter = 0
        self.max_iter = max_iter
        self.bar = _progress_start(max_iter)

    def update(self):
        self.current_iter += 1
        if self.bar:
            self.bar.update(self.current_iter)


def _progress_start(max_iter):
    """ _progress
    
    Creates a progress bar, if progressbar is installed.
    
    Arguments:
    ----------
        max_iter: integer.
            Maximum number of iterations.
    Returns:
    --------
        iterator, either unchanged or replaced by the progress bar iterator.
    """
    try:
        import progressbar

        widgets = [
            " [",
            progressbar.Timer(),
            "] ",
            progressbar.Bar(),
            " (",
            progressbar.ETA(),
            ") ",
        ]
        prog = progressbar.ProgressBar(widgets=widgets, maxval=max_iter)
        prog.start()
        return prog
    except ImportError:
        pass
    return None
