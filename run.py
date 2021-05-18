r""" A script that is used to run an experiment defined by a configuration provided as a *.yaml* file.
TODO:
  - Find a better solution for modifying sys.path as required
"""
# built-in modules
import argparse
import importlib
import logging
import os
import shutil
import sys

import yaml

from pprint import pformat

sys.path.append(".")

from src.utils.basic.general import get_timestamp, key_in_dict

# set logging configuration

log_filename = "logs/logs" + get_timestamp() + ".log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-4s %(levelname)-4s %(message)s",
    filename=log_filename,
    filemode="w",
)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


class ExperimentRunner:
    r"""Fairly generic experiment runner class.
    Attributes
    ----------
    config_path : str
        The path to the configuration yaml file.
    config_dict_orig : dict
        The unparsed configuration dictionary.
    config_dict : dict
        The parsed configuration dictionary defining the pipeline that is supposed to be run.
    timestamp : str
        The formatted timestamp of the initialization date of the class instance.
    timestamped_output_dir : str
        The path to the directory, where all of the outputs obtained by running the pipeline are stored in addition
        to the configuration yaml.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.config_dict_orig = self._load_config(self.config_path)
        self.config_dict = self._parse_config(self.config_dict_orig)

        self.timestamp = get_timestamp()
        self.timestamped_output_dir = (
            self.config_dict["output_dir"] + "/" + self.timestamp
        )
        self._make_timestamped_output_dir()

        # pprint.PrettyPrinter(indent=4).pprint(self.config_dict)
        logging.debug(pformat(self.config_path))
        logging.debug("###" * 20)
        logging.debug("###" * 20)

    def _load_config(self, config_path):
        r""" Method to load a configuration defined in a *.yaml* file.
        Parameters
        ----------
        config_path : str
            Path to the configuration.yaml file.
        Returns
        -------
        config_dict : dict
            The raw configuration dictionary representing the information defined in the configuration file.
        """
        with open(self.config_path, "r") as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)

        return config_dict

    def _parse_config(self, config_dict):
        r""" Method to parse the raw configuration dictionary by replacing e.g. the defined callable elements with
        actual respective instances.
        Parameters
        ----------
        config_dict : dict
        Returns
        -------
        config_dict : dict
            The parsed configuration dictionary
        """
        assert key_in_dict(
            ["module", "class", "run_params", "output_dir"], config_dict
        ), "Malformed configuration file"

        config_dict["module"] = importlib.import_module(config_dict["module"])
        config_dict["class"] = getattr(config_dict["module"], config_dict["class"])

        # TODO: Make sure that we get full absolute path;
        config_dict["output_dir"] = os.path.abspath(config_dict["output_dir"])

        return config_dict

    def _make_timestamped_output_dir(self):
        r"""Creates a timestamped output directory."""
        if not os.path.exists(self.timestamped_output_dir):
            os.makedirs(self.timestamped_output_dir)

    def _save_config(self, output_dir):
        r"""Saves the config file in the timestamped output dir."""
        config_fname = self.config_path.split("/")[-1]
        shutil.copy(src=self.config_path, dst=output_dir + "/" + config_fname)

    def run(self):
        r""" Method to run the pipeline configuration defined in :py:attr:`config_dict`.
        TODO
        -----
          - Move replacement of "[fun] ..." strings to appropriate place
          - Get rid of code duplication when instantiating exp class
          - Check arguments of alternative constructor; change s.t. this works generally
        """
        for key in self.config_dict["run_params"].keys():
            value = self.config_dict["run_params"][key]
            if type(value) is str:
                fun_id = value.find("[fun]")
                if fun_id > -1:
                    self.config_dict["run_params"][key] = eval(
                        value[fun_id + len("[fun] ") :]
                    )

        # Create instance of the experiment class
        if key_in_dict("constructor", self.config_dict):
            constructor = getattr(
                self.config_dict["class"], self.config_dict["constructor"]
            )
            try:  # Try to pass name of output_dir in case experiment class expects that
                exp_instance = constructor(
                    self.config_dict["run_params"],
                    output_dir=self.timestamped_output_dir,
                )
            except TypeError:
                exp_instance = constructor(self.config_dict["run_params"])
        else:
            try:
                exp_instance = self.config_dict["class"](
                    **self.config_dict["run_params"],
                    output_dir=self.timestamped_output_dir
                )
            except TypeError as exception:
                print(exception)
                exp_instance = self.config_dict["class"](
                    **(self.config_dict["run_params"])
                )

        self._make_timestamped_output_dir()
        try:
            exp_instance.output(self.timestamped_output_dir)
        except Exception:
            pass
        self._save_config(self.timestamped_output_dir)

        # Run all the steps in the pipeline
        for step in self.config_dict["pipeline"]:
            instance_method = getattr(exp_instance, step["method"])
            if key_in_dict("params", step):
                params = step["params"]
                instance_method(**params)
            else:
                instance_method()

        # Copy the log-file to the output directory
        shutil.copy(src=log_filename, dst=self.timestamped_output_dir + "/")


if __name__ == "__main__":
    r"""Main routine.
    Arguments
    ---------
    config_file : str
        Path to the configuration yaml file.
    """
    # TODO Find a solution for adjusting sys.path properly.
    sys.path += [".."]

    arg_parser = argparse.ArgumentParser(description="Pipeline runner.")

    arg_parser.add_argument(
        "--config_file", metavar="CONFIG_FILE", help="Configuration file", required=True
    )
    arg_parser.add_argument(
        "--debug",
        help="Disables the exception catching to allow for online debugging",
        required=False,
        dest="debug",
        action="store_true",
    )
    args = arg_parser.parse_args()

    if os.path.isfile(args.config_file):
        config_path = os.path.abspath(args.config_file)
    else:
        logging.critical("There is no file with name {}.".format(args.config_file))
        sys.exit()

    exp_runner = ExperimentRunner(config_path)

    if args.debug:
        exp_runner.run()
    else:
        try:
            exp_runner.run()
        except Exception:
            logging.error("", exc_info=True)
            shutil.copy(src=log_filename, dst=exp_runner.timestamped_output_dir + "/")
            sys.exit(1)

    sys.exit()
