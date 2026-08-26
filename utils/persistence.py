import io
import os
import shutil
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.pipeline import Pipeline


class Persistence:
    run: int
    path_prefix: str
    __pipeline: "Pipeline"

    def __init__(self, pipeline: "Pipeline") -> None:
        self.run = 1
        self.__pipeline = pipeline
        # If the path already exists, find the next run number
        if os.path.exists(f"./.output/{self.__pipeline}"):
            # Infer the run number from the existing directories
            for dir in os.listdir(f"./.output/{self.__pipeline}"):
                if dir.isdigit():
                    self.run = max(self.run, int(dir) + 1)
        # Create the path prefix for the output directory
        self.path_prefix = f"./.output/{self.__pipeline}/{self.run}"
        os.makedirs(self.path_prefix, exist_ok=True)  # & create the directory

    def log_path(self):
        # Log the run number
        self.__pipeline.logger.info(
            f"Output directory: {os.path.abspath(self.path_prefix)}"
        )

    def resolve_asset_path(self, name: str, tag: str | None = None) -> str:
        """Resolve the path to an asset file
        :param name: The name of the asset file
        :return: The resolved path to the asset file"""
        return os.path.join(self.path_prefix, tag or "", name)

    def read_file(self, name: str, tag: str | None = None) -> str:
        """Read a file from the record directory
        :param name: The name of the file to read
        :return: The contents of the file"""
        with open(
            os.path.join(self.path_prefix, tag or "", name), "r", encoding="utf-8"
        ) as file:
            return file.read()

    def save_file(self, name: str, data: str, tag: str | None = None) -> str:
        """Save data to a file in the record directory
        :param name: The name of the file to save
        :param data: The data to save
        :return: The path to the saved file"""
        # Save the data to the file
        with open(
            os.path.join(self.path_prefix, tag or "", name), "w", encoding="utf-8"
        ) as file:
            _ = file.write(data)
        # Return the path to the file
        return os.path.join(self.path_prefix, tag or "", name)

    def capture_stdout_log(
        self,
        name: str,
        func: Callable[[io.TextIOWrapper], None],
        tag: str | None = None,
    ) -> None:
        """Capture the stdout of a function and save it to a file
        :param name: The name of the log file
        :param func: The function to capture the stdout of"""
        # Save the original stdout
        orig_stdout = sys.stdout
        # Begin writing to the record file
        with open(
            os.path.join(self.path_prefix, tag or "", f"{name}.log"),
            "w",
            encoding="utf-8",
        ) as file:
            func(file)
            sys.stdout = orig_stdout

    def move_from_folder(self, src_path: str, folder_name: str, tag: str | None = None):
        new_folder_path = self.resolve_asset_path(folder_name, tag=tag)
        assert not os.path.exists(new_folder_path), "Folder already exists"
        shutil.copytree(src_path, new_folder_path)
        shutil.rmtree(src_path)
