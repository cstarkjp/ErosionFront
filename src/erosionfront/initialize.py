"""
Configure to run in `IPython`_.

---------------------------------------------------------------------

Sets up `IPython`_ environment if e.g. we're running
in a `Jupyter notebook`_ or `Jupyter QtConsole`_.

 - prepares Matplotlib to display inline and (for Macs)
   at a 'retina' resolution -- if this
   is not available, a benign error report (currently disabled)
   is made and progress continues
 - enables automatic reloading (in case the code has been modded) when
   a notebook is re-run in-situ
"""

__all__ = [
    "initialize",
]

def initialize() -> None:
    try:
        from IPython import get_ipython #type: ignore
            
        def check_is_ipython() -> bool:
            """Check if we are running an IPython kernel from Jupyter etc."""
            try:
                if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
                    return False
            except ImportError:
                return False
            except AttributeError:
                return False
            return True


        is_python: bool = check_is_ipython()

        if is_python:
            try:
                get_ipython().run_line_magic(
                    "config", 
                    "InlineBackend.figure_format = 'retina'",
                )
            except NameError:
                pass

            try:
                get_ipython().run_line_magic("matplotlib", "inline",)
            except NameError:
                pass

            try:
                get_ipython().run_line_magic("load_ext", "autoreload",)
                get_ipython().run_line_magic("autoreload", "2",)
            except NameError as error:
                print(
                    "Error trying to invoke get_ipython(), "
                    + "possibly because not running IPython:",
                    error,
                )
    except:
        pass

initialize()