import warnings
from functools import wraps


def warning_format(message, category=None, filename=None, lineno=None, line=None):
    # Cleaner way of printing the warning
    # Sets the warning message and does not print the line where the warning occurred
    return f"{message} \n"


def warn_known_bug(original_function=None, gitlab_issue=None):
    # Function copied from: https://stackoverflow.com/a/24617244

    def _decorate(function):
        @wraps(function)
        def wrapped_function(*args, **kwargs):

            if gitlab_issue is not None:
                add_msg = (
                    f"\n\n See gitlab issue #{gitlab_issue} "
                    f"https://gitlab.com/datafold-dev/datafold/issues/{gitlab_issue} \n"
                )
            else:
                add_msg = ""

            warnings.formatwarning = warning_format
            warnings.warn(
                f"Function '{function.__name__}' has a known bug. "
                f"Use with caution {add_msg}"
            )
            return function(*args, **kwargs)

        return wrapped_function

    if original_function:
        return _decorate(original_function)

    return _decorate


def warn_experimental_function(original_function):
    def func_wrapper(*args, **kwargs):
        warnings.formatwarning = warning_format
        warnings.warn(
            f"Class '{original_function.__name__}' is marked as experimental. This means "
            f"the intended functionality may not be complete and there is no sufficient "
            f"testing. Use function with caution!"
        )
        return original_function(*args, **kwargs)

    return func_wrapper


def warn_experimental_class(cls):
    # From ( adapted example from "Creating Singletons"
    # https://realpython.com/primer-on-python-decorators/#decorating-classes

    @wraps(cls)
    def wrapper_class(*args, **kwargs):
        warnings.formatwarning = warning_format
        warnings.warn(
            f"Class '{cls.__name__}' is marked as experimental. This means "
            f"the intended functionality may not be complete and there is no sufficient "
            f"testing. Use class with caution!"
        )
        return cls(*args, **kwargs)

    return wrapper_class


if __name__ == "__main__":

    @warn_experimental_function
    def test(a):
        print(f"a={a}")
        return 1

    print(test(1))
