"""
Export data to HDF5 and metadata to JSON files.
"""

import warnings
import os  # switch to pathlib?
import json  # look into orjson?
from io import TextIOWrapper

import numpy as np
import h5py
from h5py import File as hd5File

from erosionfront.misc.utils import is_serializable, make_serializable

warnings.filterwarnings("ignore")

__all__ = ["ExportMixin"]

class ExportMixin:
    """
    Mixin class providing data/metadata export tools for by level-set solver.
    
    Attributes
    ----------
    T_slices: dict
        Dictionary of time T slices of evolving surface and its properties.
    """
    T_slices: dict
    
    def export_data(
            self,
            dir: str, 
            filename: str="T_slices",
        ) -> None:
        """
        Tool to export time T slice data to HDF5 file.

        Parameters
        ----------
        dir: str
            Path to directory to write HDF5 file into.
        filename: str="T_slices"
            Name of HDF file.
        """
        file_path: str = os.path.join(dir, f"{filename}.hdf5",)
        file_: hd5File
        with h5py.File(file_path, "w",) as file_:
            i_: int
            for i_ in self.T_slices:
                T_group_ = file_.create_group(f"{self.T_slices[i_]['T']}")
                T_group_.create_dataset("x", data=self.T_slices[i_]["x"])
                T_group_.create_dataset("z", data=self.T_slices[i_]["z"])
                T_group_.create_dataset("β", data=self.T_slices[i_]["β"])

    def export_metadata(
            self, 
            dir: str, 
            filename: str="metadata",
        ) -> dict:
        """
        Tool to simulation metadata to JSON file.

        Parameters
        ----------
        dir: str
            Path to directory to write JSON file into.
        filename: str="metadata"
            Name of JSON file.
        """
        file_path: str = os.path.join(dir, f"{filename}.json",)
        metadata: dict = {}
        for group_ in (".", "domain", "surface", "substrate", "model", "gma",):
            if group_==".":
                object_ = self
                group_ = "sim"
            else:
                object_ = getattr(self, group_)
            metadata[group_] = {}
            for attr_ in object_.__dict__:
                value_ = object_.__dict__[attr_]
                type_ = type(value_)
                if is_serializable(value_):
                    if type_ is tuple and not isinstance(value_[0], np.ndarray):
                        metadata[group_].update({
                            attr_: [
                                make_serializable(element_)
                                for element_ in value_
                            ]
                        })
                    else:
                        metadata[group_].update({
                            attr_: make_serializable(value_)
                        })
                else:
                    if type_ is np.ndarray and value_.size<100:
                        metadata[group_].update({
                            attr_: make_serializable(value_)
                        })
        file_: TextIOWrapper
        with open(file_path, "w",) as file_:
            json.dump(
                metadata, 
                file_, 
                indent=4,
                ensure_ascii=False,
            )
        return metadata