"""
Export plots to image (PNG, JPEG) or PDF files.
"""
import warnings
import logging
from os.path import exists, join, realpath
from os import mkdir
from typing import Any

warnings.filterwarnings("ignore")

__all__ = [
    "create_dir",
    "create_directories",
    "export_plots",
    "export_plot",
]

def create_directories(
    results_path: tuple[str, str] = ("..", "Results",), 
    results_dir: str = "Demo",
) -> str:
    """
    Create results parent and target directory.

    Parameters
    ----------
    results_path: tuple[str, str] = ("..", "Results",)
        Path to parent results directory (to be created if necessary)
    results_dir: str = "Demo"
        Target results directory (to be created)

    Returns
    -------
    str: 
        Path to target results directory
    """
    results_path_: list = ["."] + list(results_path)
    create_dir(join(*results_path_))
    results_dir_: list = results_path_ + [results_dir]
    return create_dir(join(*results_dir_))


def create_dir(dir: str) -> str:
    """
    Try to create an output directory if one doesn't exist.

    Throws an exception if the directory cannot be created.
    Returns quietly if the directory already exists.

    Parameters
    ----------
    dir: str
        Name of directory.

    Returns
    -------
    str: 
        Path to directory.
    """
    try:
        if not exists(dir):
            mkdir(dir)
        else:
            return dir
    except OSError:
        print(f'Cannot create directory "{realpath(dir)}"')
        raise
    except Exception:
        print(Exception)
        raise
    return dir

def export_plots(
    fig_dict: dict,
    results_dir: str,
    file_types: list[str] | tuple[str] | str = "pdf",
    suffix: str = "",
    dpi: int | None = None,
) -> None:
    """
    Export plots to PDF or other format files.

    Parameters
    ----------
    fig_dict: dict
        Dictionary of figures
    results_dir: str
        Name of output directory
    file_types: list[str] | tuple[str] | str = "pdf"
        File format (or list of file formats)
    suffix: str = ""
        Filename suffix
    dpi: int | None = None
        DPI of output image.
    """
    results_path: str = realpath(results_dir)
    logging.info(
        "erosionfront.misc.save.export_plots:\n   " 
        + f'Writing to dir: "{results_path}"'
    )
    file_types_mod: list[str] = (
        file_types if isinstance(file_types, list) else [str(file_types)]
    )
    for file_type_ in file_types_mod:
        # logging.info(f'Image file type: "{file_type}"')
        for fig_dict_item_ in list(fig_dict.items()):
            export_plot(
                *fig_dict_item_,
                results_path,
                file_type=file_type_,
                suffix=suffix,
                dpi=dpi,
            )

def export_plot(
    fig_name: str,
    fig: Any,
    results_dir: str,
    file_type: str = "pdf",
    suffix: str = "",
    dpi: int | None = None,
) -> None:
    """
    Export plot to PDF or other format file.

    Parameters
    ----------
    fig_name: str
        Name to be used for file (extension auto-appended).
    fig: Any
        Figure object.
    results_dir: str
        Name of output directory.
    file_type: str = "pdf"
        File format.
    suffix: str = ""
        Filename suffix.
    dpi: int | None = None
        DPI of output image.
    """
    fig_name_mod: str = f"{fig_name}{suffix}.{file_type.lower()}"
    try:
        # logging.info(f'dpi={dpi}')
        fig.savefig(
            join(results_dir, fig_name_mod),
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=dpi,
            format=file_type,
        )
        logging.info(
            f'erosionfront.misc.save.export_plot: Exported "{fig_name_mod}"'
        )
    except OSError:
        logging.info(
            f'erosionfront.misc.save.export_plot: '
            + f'Failed to export figure "{fig_name_mod}"'
        )
        raise
    # except:
    #     raise
