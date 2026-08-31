import numpy as np


class DataService:


    @staticmethod
    def normalise(orig_values):
        """ Normalize data to interval [0,1] """
        range_dict = DataService.calculate_range(orig_values)

        if range_dict["range"] == 0:  # constant data has no range to scale by
            return np.zeros(len(orig_values))

        normalized_array = []
        for y_orig_val in orig_values:
            normalized_array.append( round( abs(y_orig_val - range_dict["min_from_range"]) / range_dict["range"], 2))

        return_numpy_array = np.asarray(normalized_array)
        return return_numpy_array


    @staticmethod
    def calculate_range(orig_values) -> dict:
        range_dict = {"min_from_range": min(orig_values),
                      "max_from_range": max(orig_values),
                      "range": (max(orig_values) - min(orig_values))}
        return range_dict


    @staticmethod
    def denormalise(values_to_denormalise, orig_values) -> dict:
        """ Reverts data back to original values """
        range_dict = DataService.calculate_range(orig_values)
        denormalised_values = []
        y_min = range_dict["min_from_range"]
        for i in range(len(values_to_denormalise)):
            denorm_val = y_min + values_to_denormalise[i] * range_dict["range"]
            denormalised_values.append(denorm_val)
        return_numpy_array = np.asarray(denormalised_values)
        return return_numpy_array
