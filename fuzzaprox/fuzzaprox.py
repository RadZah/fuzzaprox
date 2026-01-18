import numpy as np

from fuzzaprox.services.DataService import DataService
from fuzzaprox.transformation.Transform import Transform
from fuzzaprox.transformation.FuzzySet import FuzzySet


class Fuzzaprox:
    """ Fuzzy approximation interface - facade class """
    
    def __init__(self):
        self.fuzzy_set_instance = None
        self.fuzzy_set_push = None
        self.transformation = None
        self.data_x = None
        self.norm_data_y = None
        self.orig_data_y = None

    def define_fuzzy_set(self, base_start, kernel_start, kernel_end, base_end):
        """ Define FUZZY SET by parameters """
        fuzzy_set_instance = FuzzySet({'kernel_start': int(kernel_start - base_start),
                                       'kernel_end': int(kernel_end - base_start),
                                       'fuzzy_set_width': int(base_end - base_start)})
        self.fuzzy_set_instance = fuzzy_set_instance
        self.set_fuzzy_set_push(fuzzy_set_instance.fuzzy_set_width)

    def set_fuzzy_set(self, fuzzy_set_setting):
        """ Sets FUZZY SET by Dictionary """
        self.fuzzy_set_instance = FuzzySet(fuzzy_set_setting)

    def set_fuzzy_set_with_instance(self, fuzzy_set_instance):
        """ Sets FUZZY SETs with Instance """
        self.set_fuzzy_set({"kernel_start": fuzzy_set_instance.kernel_start,
                            "kernel_end": fuzzy_set_instance.kernel_end,
                            "fuzzy_set_width": fuzzy_set_instance.fuzzy_set_width})
        self.fuzzy_set_instance = fuzzy_set_instance

    def set_fuzzy_set_push(self, fuzzy_set_push):
        """ Sets the distance between fuzzy sets """
        self.fuzzy_set_push = int(fuzzy_set_push/2)

    @staticmethod
    def convert_list_to_array(list_data):
        """ Just converts list to array, if not already in ndarray format """
        if isinstance(list_data, list):
            np_array = np.array(list_data)
        elif isinstance(list_data, np.ndarray):
            np_array = np.array(list_data)  # Return a copy to prevent external modifications
        else:
            raise ValueError("Data should be in list or np.ndarray format")
        return np_array

    @staticmethod
    def generate_x_axis_from_y_list(y_np_array):
        """
        Generate sequential integer x-axis indices corresponding to y data points.
        
        Creates an array of integer values from 0 to len(y_np_array)-1, which serves
        as the x-axis for plotting and processing the y data. Each index corresponds
        to a position in the input array.
        
        Args:
            y_np_array (np.ndarray): Input array of y values
            
        Returns:
            np.ndarray: Array of integer indices [0, 1, 2, ..., len(y_np_array)-1]
        """
        x_axes = np.arange(len(y_np_array), dtype=int)
        return x_axes

    def set_input_data(self, y_values):
        """ Sets input data, which will be approximated """
        # Ensure approximated values are in np array format and save it as input
        data_as_array = self.convert_list_to_array(y_values)
        self.orig_data_y = y_values  # set original data

        # normalize approximated data to interval [0,1] for operations over residuated lattices
        normalized_y_np_array = DataService.normalise(data_as_array)
        self.norm_data_y = normalized_y_np_array

        # Create x-axis corresponding to input y-values
        self.data_x = self.generate_x_axis_from_y_list(normalized_y_np_array)

    def run(self):
        """ Run the approximation """
        # Set necessary input
        self.transformation = Transform(self.data_x,
                                        self.norm_data_y,
                                        self.fuzzy_set_instance,
                                        self.fuzzy_set_push)

        # Run Upper and Bottom Approximation
        self.transformation.upper_bottom_forward_ft()  # Calculates Forward F-transforms
        self.transformation.calculate_inverse_ft()  # Calculates Inverse F-transforms

    def get_approximations(self):
        """ --- """
        # format output
        return_dict = self.transformation.return_input_param_and_approx()
        return return_dict

    def get_fw_approx_upper(self):
        """ Returns UPPER FORWARD approximation """

        fw_x = self.transformation.upper_t_fw_data_x.copy()
        fw_y = self.transformation.upper_t_fw_data_y.copy()
        last_orig_x = self.transformation.original_data_x[-1]
        while fw_x[-1] > last_orig_x:
            fw_x = fw_x[:-1]
            fw_y = fw_y[:-1]
            
        fw_approx_upper = {"fw_x": fw_x, "fw_y": fw_y}
        return fw_approx_upper
    
    def get_inv_approx_upper(self):
        """ Returns UPPER INVERSION approximation """
        inv_approx_upper = self.transformation.upper_t_inv_data_y
        return inv_approx_upper

    def get_fw_approx_bottom(self):
        """ Returns BOTTOM FORWARD approximation """
        
        fw_x = self.transformation.bottom_t_fw_data_x.copy()
        fw_y = self.transformation.bottom_t_fw_data_y.copy()
        last_orig_x = self.transformation.original_data_x[-1]
        while fw_x[-1] > last_orig_x:
            fw_x = fw_x[:-1]
            fw_y = fw_y[:-1]
            
        fw_approx_bottom = {"fw_x": fw_x, "fw_y": fw_y}
        return fw_approx_bottom
    
    def get_inv_approx_bottom(self):
        """ Returns BOTTOM INVERSE approximation """
        inv_approx_bottom = self.transformation.bottom_t_inv_data_y
        return inv_approx_bottom

    def get_normalised_y_vals(self):
        """ Returns normalized-Y values """
        return self.norm_data_y

    def get_x_axes(self):
        """ Return just X-axes range with values of integer numbers """
        return self.data_x
