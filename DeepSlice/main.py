import os
import tempfile
from glob import glob
from typing import Union
from .coord_post_processing import spacing_and_indexing, angle_methods, horizontal
from .read_and_write import QuickNII_functions
from .neural_network import neural_network
from .metadata import metadata_loader


ORIENTATIONS = ("coronal", "horizontal")


class DSModel:
    def __init__(self, species, orientation="coronal"):
        """Initialises a DeepSlice model for a given species
        :param species: the species of the brain to be processed, must be one of "mouse", "rat"
        :type species: str
        :param orientation: the cutting plane, "coronal" (default) or "horizontal" (mouse only;
            sections tilted up to ~30 degrees from horizontal, imaged with anterior at the top)
        :type orientation: str
        """
        if orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {ORIENTATIONS}")
        if orientation == "horizontal" and species != "mouse":
            raise ValueError("horizontal models are only available for mouse")
        self.species = species
        self.orientation = orientation
        self.model_key = "mouse_horizontal" if orientation == "horizontal" else species

        self.config, self.metadata_path = metadata_loader.load_config()
        xception_weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"]["xception_imagenet"], self.metadata_path
        )
        weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"][self.model_key]["primary"],
            self.metadata_path,
        )
        self.model = neural_network.initialise_network(
            xception_weights, weights, self.species
        )

    def predict(
        self,
        image_directory: str = None,
        ensemble: bool = None,
        section_numbers: bool = True,
        legacy_section_numbers=False,
        image_list=None,
        use_secondary_model=False,
        pixel_size_um=None,
    ):
        """predicts the atlas position for a folder full of histological brain sections

        :param image_directory: the directory containing the brain sections
        :type image_directory: str
        :param ensemble: whether to use multiple models, this will default to True when available, defaults to None
        :type ensemble: bool, optional
        :param section_numbers: whether this dataset includes section numbers (as the last three digits of the filename) , defaults to True
        :type section_numbers: bool, optional
        :param legacy_section_numbers: a legacy setting which parses section numbers how old DeepSlice used to, defaults to False
        :type legacy_section_numbers: bool, optional
        :param pixel_size_um: image pixel size in microns (a number, or a dict from file name to number); required by models with a fixed physical field of view
        :type pixel_size_um: float or dict, optional
        """

        # We set this to false as predict is the entry point for a new brain and therefore we need to reset all values which may persist from a previous animal.
        self.bad_sections_present = False
        # Different species may or may not have an ensemble model, so we need to check for this before defaulting to True
        if ensemble == None:
            ensemble = self.config["ensemble_status"][self.model_key]
            ensemble = eval(ensemble)
        # models with a fixed physical field of view see each image as a square canvas of
        # that size around the tissue; predictions are mapped back to the original images
        fov_um = self.config["weight_file_paths"][self.model_key].get("field_of_view_um")
        canvases, canvas_dir, original_list = None, None, image_list
        if fov_um:
            if pixel_size_um is None:
                raise ValueError(f"the {self.model_key} model needs pixel_size_um")
            sources = image_list or [
                p for p in glob(image_directory + "/*")
                if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
            ]
            canvas_dir = tempfile.TemporaryDirectory()
            canvases = horizontal.prepare_canvases(sources, pixel_size_um, fov_um, canvas_dir.name)
            original_list = sources
            image_list = [os.path.join(canvas_dir.name, os.path.basename(p)) for p in sources]
        if image_list:
            image_generator, width, height = neural_network.load_images_from_list(
                image_list
            )
            if image_directory and not canvases:
                print(
                    "WARNING: image_directory is set but image_list is also set. image_directory will be ignored."
                )
        else:
            image_generator, width, height = neural_network.load_images_from_path(
                image_directory
            )
        weight_paths = self.config["weight_file_paths"][self.model_key]
        primary_weights = metadata_loader.get_data_path(
            weight_paths["primary"], self.metadata_path
        )
        if "secondary" in weight_paths:
            secondary_weights = metadata_loader.get_data_path(
                weight_paths["secondary"], self.metadata_path
            )
        else:
            secondary_weights = "None"

        if secondary_weights == "None":
            print(f"ensemble is not available for {self.model_key}")
            if use_secondary_model:
                print(
                    "WARNING: use_secondary_model is set but no secondary model is available. use_secondary_model will be ignored."
                )
                use_secondary_model = False
            ensemble = False
        if use_secondary_model and ensemble:
            print(
                "WARNING: use_secondary_model is set but ensemble is also set. use_secondary_model will be ignored."
            )
            use_secondary_model = False
        if use_secondary_model:
            print("Using secondary model")
            predictions = neural_network.predictions_util(
                self.model,
                image_generator,
                secondary_weights,
                None,
                ensemble,
                self.species,
            )
        else:
            predictions = neural_network.predictions_util(
                self.model,
                image_generator,
                primary_weights,
                secondary_weights,
                ensemble,
                self.species,
            )
        predictions["width"] = width
        predictions["height"] = height
        if canvases:
            cols = horizontal.COLUMNS
            predictions[cols] = [
                horizontal.canvas_to_image_anchoring(a, canvases[f][0], *canvases[f][1])
                for a, f in zip(predictions[cols].values, predictions["Filenames"])
            ]
            predictions["width"] = [canvases[f][1][0] for f in predictions["Filenames"]]
            predictions["height"] = [canvases[f][1][1] for f in predictions["Filenames"]]
            canvas_dir.cleanup()
            image_list = original_list
        if self.orientation == "horizontal":
            # which hemisphere(s) each image shows, and how much tissue it contains
            if image_list:
                by_name = {os.path.basename(p): p for p in image_list}
                paths = [by_name[f] for f in predictions["Filenames"]]
            else:
                paths = [os.path.join(image_directory, f) for f in predictions["Filenames"]]
            predictions = horizontal.add_section_descriptions(predictions, paths)
        if section_numbers:
            predictions["nr"] = spacing_and_indexing.number_sections(
                predictions["Filenames"], legacy_section_numbers
            )
            predictions["nr"] = predictions["nr"].astype(int)
            predictions = predictions.sort_values(by="nr").reset_index(drop=True)
        elif self.orientation == "horizontal":
            predictions["depths"] = [
                horizontal.centre_depth(a) for a in predictions[horizontal.COLUMNS].values
            ]
            predictions = predictions.sort_values(by="depths").reset_index(drop=True)
        else:
            predictions = predictions.sort_values(by="oy").reset_index(drop=True)

        #: pd.DataFrame: Filenames and predicted QuickNII coordinates of the input sections.

        self.predictions = predictions
        self.image_directory = image_directory

    def set_bad_sections(self, bad_sections: list, auto=False):
        """
        sets the bad sections for a given brain. Must be run after predict()

        :param bad_sections: A list of bad sections to ignore when calculating angles and spacing, the list just needs to contain a unique string for each section, for instance the section number. do not use a string which appears in multiple filenames
        :type bad_sections: list
        """
        self.predictions = spacing_and_indexing.set_bad_sections_util(
            self.predictions, bad_sections, auto
        )

    def enforce_index_order(self):
        """
        reorders the section depths (oy, or the dorsoventral depth for horizontal series) in the predictions such that they align with the section indexes
        """
        if self.orientation == "horizontal":
            self.predictions = horizontal.enforce_index_order(
                self.predictions, self._depth_curve()
            )
            return
        self.predictions = spacing_and_indexing.enforce_section_ordering(
            self.predictions
        )

    def enforce_index_spacing(
        self, section_thickness: Union[int, float] = None, suppress=False
    ):
        """
        Space evenly according to the section indexes, if these indexes do not represent the precise order in which the sections were
        cut, this will lead to less accurate predictions. Section indexes must account for missing sections (ie, if section 3 is missing
        indexes must be 1, 2, 4).

        :param section_thickness: the thickness of the sections in microns, defaults to None
        :type section_thickness: Union[int, float], optional
        """
        if self.orientation == "horizontal":
            self.predictions = horizontal.enforce_index_spacing(
                self.predictions,
                section_thickness_um=section_thickness,
                curve=self._depth_curve(),
                verbose=not suppress,
            )
            return
        voxel_size = self.config["target_volumes"][self.species]["voxel_size_microns"]
        self.predictions = spacing_and_indexing.space_according_to_index(
            self.predictions,
            section_thickness=section_thickness,
            voxel_size=voxel_size,
            suppress=suppress,
            species=self.species,
        )

    def adjust_angles(self, ML: Union[int, float], DV: Union[int, float]):
        """
        Adjusts the Mediolateral (ML) and Dorsoventral (DV) angles of all sections to the specified values.

        :param ML: the Mediolateral angle to set
        :param DV: the Dorsoventral angle to set
        :type ML: [int, float]
        :type DV: [int, float]
        """
        if self.orientation == "horizontal":
            raise NotImplementedError("for horizontal series use set_tilt(pitch, roll)")
        self.predictions = angle_methods.set_angles(self.predictions, ML, DV)

    def set_tilt(self, pitch: Union[int, float], roll: Union[int, float]):
        """
        Sets the tilt of all sections in a horizontal series.

        :param pitch: anteroposterior slope in degrees (positive: anterior end higher)
        :param roll: mediolateral slope in degrees (positive: animal's right side higher)
        """
        if self.orientation != "horizontal":
            raise ValueError("set_tilt is for horizontal series; use adjust_angles")
        self.predictions = horizontal.set_series_tilt(self.predictions, pitch, roll)

    def set_hemisphere(self, side: str, sections: list = None, all_sections: bool = False):
        """
        Horizontal series only: move one-hemisphere sections to the hemisphere that was
        imaged. The model assumes sections are seen from the dorsal side, so a hemisphere
        mounted ventral side up is predicted as the other hemisphere.

        :param side: "left" or "right"
        :param sections: optional list of strings; only filenames containing one are changed
        :param all_sections: consider every section, not only those called one-sided; use
            when the whole series is one hemisphere
        """
        if self.orientation != "horizontal":
            raise ValueError("set_hemisphere is only available for horizontal series")
        self.predictions = horizontal.set_hemisphere(
            self.predictions, side, sections, all_sections
        )

    def propagate_scale(self, pixel_size_um: Union[int, float]):
        """
        Horizontal series only: with a known pixel size, give every section the
        series-wide scale (tissue shrinkage) estimated from the reliable sections.

        :param pixel_size_um: pixel size of the images in microns
        """
        if self.orientation != "horizontal":
            raise ValueError("propagate_scale is only available for horizontal series")
        self.predictions = horizontal.propagate_scale(
            self.predictions, pixel_size_um, self._depth_curve()
        )

    def _depth_curve(self):
        return self.config.get("depth_weights", {}).get(self.model_key)

    def propagate_angles(self, method="weighted_mean"):
        """
        Calculates the average Mediolateral and Dorsoventral angles for all sections.
        For horizontal series, the reliability-weighted mean pitch and roll.
        """
        if self.orientation == "horizontal":
            curve = self._depth_curve()
            if method == "mean":
                curve = {"z": [0.0, 320.0], "weight": [1.0, 1.0]}
            self.predictions = horizontal.propagate_tilt(self.predictions, curve)
            return
        ##needs to be run twice as adjusting the angle in one plane bumps the other out slightly.
        for i in range(2):
            self.predictions = angle_methods.propagate_angles(
                self.predictions, method, self.species
            )

    def load_QUINT(self, filename):
        """
        Load a QUINT compatible JSON or XML.

        :param filename: the name of the file to load
        :type filename: str
        """
        if filename.lower().endswith(".json"):
            predictions, target = QuickNII_functions.read_QUINT_JSON(filename)
            if target == "ABA_Mouse_CCFv3_2017_25um.cutlas" and self.species != "mouse":
                self.species = "mouse"
                print("Switching to a mouse model")
            elif target == "WHS_Rat_v4_39um.cutlas" and self.species != "rat":
                self.species = "rat"
                print("switching to a rat model")
        elif filename.lower().endswith(".xml"):
            predictions = QuickNII_functions.read_QuickNII_XML(filename)
        else:
            raise ValueError("File must be a JSON or XML")
        if self.species != "mouse":
            self.orientation = "coronal"
        self.model_key = "mouse_horizontal" if self.orientation == "horizontal" else self.species
        self.predictions = predictions
        xception_weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"]["xception_imagenet"], self.metadata_path
        )
        weights = metadata_loader.get_data_path(
            self.config["weight_file_paths"][self.model_key]["primary"],
            self.metadata_path,
        )
        self.model = neural_network.initialise_network(
            xception_weights, weights, self.species
        )

    def save_predictions(self, filename):
        """
        Save the predictions to a QuickNII compatible JSON file.

        :param filename: the name of the file to save to
        :type filename: str
        """
        target = self.config["target_volumes"][self.species]["name"]
        aligner = self.config["DeepSlice_version"]["prerelease"]
        self.predictions.to_csv(filename + ".csv", index=False)
        QuickNII_functions.write_QUINT_JSON(
            df=self.predictions, filename=filename, aligner=aligner, target=target
        )
