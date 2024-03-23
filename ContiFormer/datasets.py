"""
The dataset preprocessing is based on DeepAR
https://arxiv.org/pdf/1704.04110.pdf
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sktime.datasets import load_from_arff_to_dataframe
from torch import Tensor
import os, os.path
import urllib.response
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from torch.utils.data import TensorDataset
import pandas as pd
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
#from torchaudio.datasets import SPEECHCOMMANDS
import os
import urllib.request
import tarfile
import shutil
import librosa
import torch.utils.data as data
from scipy import integrate
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler



class SinusoidalDataset(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=1, freq_min=10, freq_max=500, num_classes=100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sinusoidal signals with noise and trend
        frequencies = [i for i in range(self.freq_min, self.freq_max+1, (self.freq_max-self.freq_min)//self.num_classes)]
        freq = frequencies[idx % self.num_classes]

        t = np.linspace(0, 1, self.seq_length)
        signal = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)

        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend) 
        signal += trend

        # Add more complex patterns to the signal
        signal += 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)
        

        # Add more noise to the signal
        #signal += 0.1 * np.random.randn(self.seq_length)

        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features), 'label': label}

        return sample
    
class SinusoidalDatasetLong(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=1, freq_min=10, freq_max=500, num_classes=20):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sinusoidal signals with noise and trend
        frequencies = [i for i in range(self.freq_min, self.freq_max+1, (self.freq_max-self.freq_min)//self.num_classes)]
        freq = frequencies[idx % self.num_classes]

        t = np.linspace(0, 1, self.seq_length)
        signal = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)

        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend) 
        signal += trend

        # Add more complex patterns to the signal
        signal += 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)

        # Add more noise to the signal
        signal += 0.1 * np.random.randn(self.seq_length)

        signal2 = 0.5 * np.sin(2 * np.pi * (freq / 5) * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)
        signal2 += 0.2 * np.sin(4 * np.pi * (freq / 5) * t) + 0.1 * np.sin(8 * np.pi * (freq / 5) * t)
        signal2 += 0.1 * np.random.randn(self.seq_length)

        signal[250:] = signal2[250:]

        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features), 'label': label}

        return sample
    

class LeadLagDataset(Dataset):
    def __init__(self, num_samples, seq_length=100, ll_length=100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.ll_length = ll_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        v1 = np.random.normal(0, 0.1, self.seq_length)
        v2 = np.random.normal(0, 0.1, self.seq_length)
        if idx%2 == 1:
            v2[self.seq_length-self.ll_length :] = v1[:self.ll_length]

        sample = np.stack((v1.cumsum(), v2.cumsum()), axis=1)


        sample = {'input': torch.tensor(sample, dtype=torch.float), 'label': idx%2}
        
        '''
        t = np.linspace(0, 1, self.seq_length)

        v1 = 0.5 * np.sin( np.random.uniform(0, 2*np.pi) * t + np.random.uniform(0, 2*np.pi)) #+ 0.01 * np.random.randn(self.seq_length)
        v2 = 0.5 * np.sin( np.random.uniform(0, 2*np.pi) * t + np.random.uniform(0, 2*np.pi)) #+ 0.01 * np.random.randn(self.seq_length)

        if idx%2 == 1:
            v2[self.seq_length-self.ll_length :] = v1[:self.ll_length]

        sample = np.stack((v1, v2), axis=1)

        sample = {'input': torch.tensor(sample, dtype=torch.float), 'label': idx%2}
        
        '''
           
        return sample
        
    

class EigenWorms():
    def __init__(self):
        pass

    ## taken from https://github.com/jambo6/neuralRDEs/blob/master/get_data/uea.py +
    #  https://github.com/tk-rusch/LEM/blob/main/src/eigenWorms/data.py and changed slightly

    def download(self, raw_data_dir):
        url = 'http://www.timeseriesclassification.com/aeon-toolkit/EigenWorms.zip'
        save_dir = raw_data_dir
        zipname = save_dir + '/eigenWorms.zip'
        ## download zipped data
        urllib.request.urlretrieve(url, zipname)
        ## unzip:
        with zipfile.ZipFile(zipname, 'r') as zip:
            zip.extractall(save_dir)

    def create_torch_data(self, train_file, test_file):
        """Creates torch tensors for test and training from the UCR arff format.

        Args:
            train_file (str): The location of the training data arff file.
            test_file (str): The location of the testing data arff file.

        Returns:
            data_train, data_test, labels_train, labels_test: All as torch tensors.
        """
        # Get arff format
        train_data, train_labels = load_from_arff_to_dataframe(train_file)
        test_data, test_labels = load_from_arff_to_dataframe(test_file)

        def convert_data(data):
            # Expand the series to numpy
            data_expand = data.applymap(lambda x: x.values).values
            # Single array, then to tensor
            data_numpy = np.stack([np.vstack(x).T for x in data_expand])
            return data_numpy

        train_data, test_data = convert_data(train_data), convert_data(test_data)

        # Encode labels as often given as strings
        encoder = LabelEncoder().fit(train_labels)
        train_labels, test_labels = encoder.transform(train_labels), encoder.transform(test_labels)
        train_labels, test_labels = torch.Tensor(train_labels), torch.Tensor(test_labels)

        return train_data, test_data, train_labels, test_labels
    

    def process_data(self, raw_data_dir, processed_data_dir):
        rnd_state = 1234
        train_arff = raw_data_dir + '/EigenWorms_TRAIN.arff'
        test_arff = raw_data_dir + '/EigenWorms_TEST.arff'
        trainx, testx, trainy, testy = self.create_torch_data(train_arff,test_arff)
        datax = np.vstack((trainx, testx))
        datay = np.hstack((trainy, testy))

        train_data, test_data, train_labels, test_labels = model_selection.train_test_split(datax, datay, test_size=0.3, random_state=rnd_state)
        valid_data, test_data, valid_labels, test_labels = model_selection.train_test_split(test_data, test_labels,
                                                                                            test_size=0.5, random_state=rnd_state)

        train_dataset = TensorDataset(Tensor(train_data).float(), Tensor(train_labels).long())
        torch.save(train_dataset, processed_data_dir + '/training.pt')
        test_dataset = TensorDataset(Tensor(test_data).float(), Tensor(test_labels).long())
        torch.save(test_dataset, processed_data_dir + '/test.pt')
        valid_dataset = TensorDataset(Tensor(valid_data).float(), Tensor(valid_labels).long())
        torch.save(valid_dataset, processed_data_dir + '/validation.pt')

    def get_eigenworms(self):
        data_dir = os.getcwd()  + '/data/eigenworms'
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        raw_data_dir = data_dir + '/raw'
        if os.path.isdir(raw_data_dir):
            print("Data already downloaded")
        else:
            os.mkdir(raw_data_dir)
            print("Downloading data")
            self.download(raw_data_dir)
            print("Data download finished")

        processed_data_dir = data_dir + '/processed'
        if os.path.isdir(processed_data_dir):
            print("Data already processed")
        else:
            os.mkdir(processed_data_dir)
            self.process_data(raw_data_dir, processed_data_dir)
            print("Finished processing data")

        train_dataset = torch.load(processed_data_dir + '/training.pt')
        test_dataset = torch.load(processed_data_dir + '/test.pt')
        valid_dataset = torch.load(processed_data_dir + '/validation.pt')

        return train_dataset, test_dataset, valid_dataset


class TsFileParseException(Exception):
    """
    Should be raised when parsing a .ts file and the format is incorrect.
    """
    pass


#Taken from https://github.com/ChangWeiTan/TSRegression/blob/master/utils/data_loader.py and https://github.com/tk-rusch/LEM/blob/main/src/heart_rate/data.py

class HeartRate():
    def __init__(self):
        pass

    def load_from_tsfile_to_dataframe(self, full_file_path_and_name, return_separate_X_and_y=True,
                                    replace_missing_vals_with='NaN'):
        """Loads data from a .ts file into a Pandas DataFrame.
        Parameters
        ----------
        full_file_path_and_name: str
            The full pathname of the .ts file to read.
        return_separate_X_and_y: bool
            true if X and Y values should be returned as separate Data Frames (X) and a numpy array (y), false otherwise.
            This is only relevant for data that
        replace_missing_vals_with: str
        The value that missing values in the text file should be replaced with prior to parsing.
        Returns
        -------
        DataFrame, ndarray
            If return_separate_X_and_y then a tuple containing a DataFrame and a numpy array containing the relevant time-series and corresponding class values.
        DataFrame
            If not return_separate_X_and_y then a single DataFrame containing all time-series and (if relevant) a column "class_vals" the associated class values.
        """

        # Initialize flags and variables used when parsing the file
        metadata_started = False
        data_started = False

        has_problem_name_tag = False
        has_timestamps_tag = False
        has_univariate_tag = False
        has_class_labels_tag = False
        has_target_labels_tag = False
        has_data_tag = False

        previous_timestamp_was_float = None
        previous_timestamp_was_int = None
        previous_timestamp_was_timestamp = None
        num_dimensions = None
        is_first_case = True
        instance_list = []
        class_val_list = []
        line_num = 0

        # Parse the file
        # print(full_file_path_and_name)
        with open(full_file_path_and_name, 'r', encoding='ISO-8859-1') as file:
            for line in tqdm(file):
                # print(".", end='')
                # Strip white space from start/end of line and change to lowercase for use below
                line = line.strip().lower()
                # Empty lines are valid at any point in a file
                if line:
                    # Check if this line contains metadata
                    # Please note that even though metadata is stored in this function it is not currently published externally
                    if line.startswith("@problemname"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")
                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException("problemname tag requires an associated value")

                        problem_name = line[len("@problemname") + 1:]
                        has_problem_name_tag = True
                        metadata_started = True
                    elif line.startswith("@timestamps"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len != 2:
                            raise TsFileParseException("timestamps tag requires an associated Boolean value")
                        elif tokens[1] == "true":
                            timestamps = True
                        elif tokens[1] == "false":
                            timestamps = False
                        else:
                            raise TsFileParseException("invalid timestamps value")
                        has_timestamps_tag = True
                        metadata_started = True
                    elif line.startswith("@univariate"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)
                        if token_len != 2:
                            raise TsFileParseException("univariate tag requires an associated Boolean value")
                        elif tokens[1] == "true":
                            univariate = True
                        elif tokens[1] == "false":
                            univariate = False
                        else:
                            raise TsFileParseException("invalid univariate value")

                        has_univariate_tag = True
                        metadata_started = True
                    elif line.startswith("@classlabel"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException("classlabel tag requires an associated Boolean value")

                        if tokens[1] == "true":
                            class_labels = True
                        elif tokens[1] == "false":
                            class_labels = False
                        else:
                            raise TsFileParseException("invalid classLabel value")

                        # Check if we have any associated class values
                        if token_len == 2 and class_labels:
                            raise TsFileParseException("if the classlabel tag is true then class values must be supplied")

                        has_class_labels_tag = True
                        class_label_list = [token.strip() for token in tokens[2:]]
                        metadata_started = True
                    elif line.startswith("@targetlabel"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException("targetlabel tag requires an associated Boolean value")

                        if tokens[1] == "true":
                            target_labels = True
                        elif tokens[1] == "false":
                            target_labels = False
                        else:
                            raise TsFileParseException("invalid targetLabel value")

                        has_target_labels_tag = True
                        class_val_list = []
                        metadata_started = True
                    # Check if this line contains the start of data
                    elif line.startswith("@data"):
                        if line != "@data":
                            raise TsFileParseException("data tag should not have an associated value")

                        if data_started and not metadata_started:
                            raise TsFileParseException("metadata must come before data")
                        else:
                            has_data_tag = True
                            data_started = True
                    # If the 'data tag has been found then metadata has been parsed and data can be loaded
                    elif data_started:
                        # Check that a full set of metadata has been provided
                        incomplete_regression_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_target_labels_tag or not has_data_tag
                        incomplete_classification_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_class_labels_tag or not has_data_tag
                        if incomplete_regression_meta_data and incomplete_classification_meta_data:
                            raise TsFileParseException("a full set of metadata has not been provided before the data")

                        # Replace any missing values with the value specified
                        line = line.replace("?", replace_missing_vals_with)

                        # Check if we dealing with data that has timestamps
                        if timestamps:
                            # We're dealing with timestamps so cannot just split line on ':' as timestamps may contain one
                            has_another_value = False
                            has_another_dimension = False

                            timestamps_for_dimension = []
                            values_for_dimension = []

                            this_line_num_dimensions = 0
                            line_len = len(line)
                            char_num = 0

                            while char_num < line_len:
                                # Move through any spaces
                                while char_num < line_len and str.isspace(line[char_num]):
                                    char_num += 1

                                # See if there is any more data to read in or if we should validate that read thus far

                                if char_num < line_len:

                                    # See if we have an empty dimension (i.e. no values)
                                    if line[char_num] == ":":
                                        if len(instance_list) < (this_line_num_dimensions + 1):
                                            instance_list.append([])

                                        instance_list[this_line_num_dimensions].append(pd.Series())
                                        this_line_num_dimensions += 1

                                        has_another_value = False
                                        has_another_dimension = True

                                        timestamps_for_dimension = []
                                        values_for_dimension = []

                                        char_num += 1
                                    else:
                                        # Check if we have reached a class label
                                        if line[char_num] != "(" and target_labels:
                                            class_val = line[char_num:].strip()

                                            # if class_val not in class_val_list:
                                            #     raise TsFileParseException(
                                            #         "the class value '" + class_val + "' on line " + str(
                                            #             line_num + 1) + " is not valid")

                                            class_val_list.append(float(class_val))
                                            char_num = line_len

                                            has_another_value = False
                                            has_another_dimension = False

                                            timestamps_for_dimension = []
                                            values_for_dimension = []

                                        else:

                                            # Read in the data contained within the next tuple

                                            if line[char_num] != "(" and not target_labels:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " does not start with a '('")

                                            char_num += 1
                                            tuple_data = ""

                                            while char_num < line_len and line[char_num] != ")":
                                                tuple_data += line[char_num]
                                                char_num += 1

                                            if char_num >= line_len or line[char_num] != ")":
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " does not end with a ')'")

                                            # Read in any spaces immediately after the current tuple

                                            char_num += 1

                                            while char_num < line_len and str.isspace(line[char_num]):
                                                char_num += 1

                                            # Check if there is another value or dimension to process after this tuple

                                            if char_num >= line_len:
                                                has_another_value = False
                                                has_another_dimension = False

                                            elif line[char_num] == ",":
                                                has_another_value = True
                                                has_another_dimension = False

                                            elif line[char_num] == ":":
                                                has_another_value = False
                                                has_another_dimension = True

                                            char_num += 1

                                            # Get the numeric value for the tuple by reading from the end of the tuple data backwards to the last comma

                                            last_comma_index = tuple_data.rfind(',')

                                            if last_comma_index == -1:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains a tuple that has no comma inside of it")

                                            try:
                                                value = tuple_data[last_comma_index + 1:]
                                                value = float(value)

                                            except ValueError:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains a tuple that does not have a valid numeric value")

                                            # Check the type of timestamp that we have

                                            timestamp = tuple_data[0: last_comma_index]

                                            try:
                                                timestamp = int(timestamp)
                                                timestamp_is_int = True
                                                timestamp_is_timestamp = False
                                            except ValueError:
                                                timestamp_is_int = False

                                            if not timestamp_is_int:
                                                try:
                                                    timestamp = float(timestamp)
                                                    timestamp_is_float = True
                                                    timestamp_is_timestamp = False
                                                except ValueError:
                                                    timestamp_is_float = False

                                            if not timestamp_is_int and not timestamp_is_float:
                                                try:
                                                    timestamp = timestamp.strip()
                                                    timestamp_is_timestamp = True
                                                except ValueError:
                                                    timestamp_is_timestamp = False

                                            # Make sure that the timestamps in the file (not just this dimension or case) are consistent

                                            if not timestamp_is_timestamp and not timestamp_is_int and not timestamp_is_float:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains a tuple that has an invalid timestamp '" + timestamp + "'")

                                            if previous_timestamp_was_float is not None and previous_timestamp_was_float and not timestamp_is_float:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                            if previous_timestamp_was_int is not None and previous_timestamp_was_int and not timestamp_is_int:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                            if previous_timestamp_was_timestamp is not None and previous_timestamp_was_timestamp and not timestamp_is_timestamp:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                            # Store the values

                                            timestamps_for_dimension += [timestamp]
                                            values_for_dimension += [value]

                                            #  If this was our first tuple then we store the type of timestamp we had

                                            if previous_timestamp_was_timestamp is None and timestamp_is_timestamp:
                                                previous_timestamp_was_timestamp = True
                                                previous_timestamp_was_int = False
                                                previous_timestamp_was_float = False

                                            if previous_timestamp_was_int is None and timestamp_is_int:
                                                previous_timestamp_was_timestamp = False
                                                previous_timestamp_was_int = True
                                                previous_timestamp_was_float = False

                                            if previous_timestamp_was_float is None and timestamp_is_float:
                                                previous_timestamp_was_timestamp = False
                                                previous_timestamp_was_int = False
                                                previous_timestamp_was_float = True

                                            # See if we should add the data for this dimension

                                            if not has_another_value:
                                                if len(instance_list) < (this_line_num_dimensions + 1):
                                                    instance_list.append([])

                                                if timestamp_is_timestamp:
                                                    timestamps_for_dimension = pd.DatetimeIndex(timestamps_for_dimension)

                                                instance_list[this_line_num_dimensions].append(
                                                    pd.Series(index=timestamps_for_dimension, data=values_for_dimension))
                                                this_line_num_dimensions += 1

                                                timestamps_for_dimension = []
                                                values_for_dimension = []

                                elif has_another_value:
                                    raise TsFileParseException(
                                        "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                            line_num + 1) + " ends with a ',' that is not followed by another tuple")

                                elif has_another_dimension and target_labels:
                                    raise TsFileParseException(
                                        "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                            line_num + 1) + " ends with a ':' while it should list a class value")

                                elif has_another_dimension and not target_labels:
                                    if len(instance_list) < (this_line_num_dimensions + 1):
                                        instance_list.append([])

                                    instance_list[this_line_num_dimensions].append(pd.Series(dtype=np.float32))
                                    this_line_num_dimensions += 1
                                    num_dimensions = this_line_num_dimensions

                                # If this is the 1st line of data we have seen then note the dimensions

                                if not has_another_value and not has_another_dimension:
                                    if num_dimensions is None:
                                        num_dimensions = this_line_num_dimensions

                                    if num_dimensions != this_line_num_dimensions:
                                        raise TsFileParseException("line " + str(
                                            line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                            # Check that we are not expecting some more data, and if not, store that processed above

                            if has_another_value:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ',' that is not followed by another tuple")

                            elif has_another_dimension and target_labels:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ':' while it should list a class value")

                            elif has_another_dimension and not target_labels:
                                if len(instance_list) < (this_line_num_dimensions + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dimensions].append(pd.Series())
                                this_line_num_dimensions += 1
                                num_dimensions = this_line_num_dimensions

                            # If this is the 1st line of data we have seen then note the dimensions

                            if not has_another_value and num_dimensions != this_line_num_dimensions:
                                raise TsFileParseException("line " + str(
                                    line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                            # Check if we should have class values, and if so that they are contained in those listed in the metadata

                            if target_labels and len(class_val_list) == 0:
                                raise TsFileParseException("the cases have no associated class values")
                        else:
                            dimensions = line.split(":")
                            # If first row then note the number of dimensions (that must be the same for all cases)
                            if is_first_case:
                                num_dimensions = len(dimensions)

                                if target_labels:
                                    num_dimensions -= 1

                                for dim in range(0, num_dimensions):
                                    instance_list.append([])
                                is_first_case = False

                            # See how many dimensions that the case whose data in represented in this line has
                            this_line_num_dimensions = len(dimensions)

                            if target_labels:
                                this_line_num_dimensions -= 1

                            # All dimensions should be included for all series, even if they are empty
                            if this_line_num_dimensions != num_dimensions:
                                raise TsFileParseException("inconsistent number of dimensions. Expecting " + str(
                                    num_dimensions) + " but have read " + str(this_line_num_dimensions))

                            # Process the data for each dimension
                            for dim in range(0, num_dimensions):
                                dimension = dimensions[dim].strip()

                                if dimension:
                                    data_series = dimension.split(",")
                                    data_series = [float(i) for i in data_series]
                                    instance_list[dim].append(pd.Series(data_series))
                                else:
                                    instance_list[dim].append(pd.Series())

                            if target_labels:
                                class_val_list.append(float(dimensions[num_dimensions].strip()))

                line_num += 1

        # Check that the file was not empty
        if line_num:
            # Check that the file contained both metadata and data
            complete_regression_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_target_labels_tag and has_data_tag
            complete_classification_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_class_labels_tag and has_data_tag

            if metadata_started and not complete_regression_meta_data and not complete_classification_meta_data:
                raise TsFileParseException("metadata incomplete")
            elif metadata_started and not data_started:
                raise TsFileParseException("file contained metadata but no data")
            elif metadata_started and data_started and len(instance_list) == 0:
                raise TsFileParseException("file contained metadata but no data")

            # Create a DataFrame from the data parsed above
            data = pd.DataFrame(dtype=np.float32)

            for dim in range(0, num_dimensions):
                data['dim_' + str(dim)] = instance_list[dim]

            # Check if we should return any associated class labels separately

            if target_labels:
                if return_separate_X_and_y:
                    return data, np.asarray(class_val_list)
                else:
                    data['class_vals'] = pd.Series(class_val_list)
                    return data
            else:
                return data
        else:
            raise TsFileParseException("empty file")

    def download(self, raw_data_dir):
        ## Unfortunately, all Monash, UEA & UCR Time Series Regression datasets have to be downloaded, although we only need the HR prediction data set
        url = 'https://zenodo.org/record/3902651/files/Monash_UEA_UCR_Regression_Archive.zip?download=1'
        save_dir = raw_data_dir
        zipname = save_dir + '/uea_reg.zip'
        ## download zipped data
        urllib.request.urlretrieve(url, zipname)
        ## unzip:
        with zipfile.ZipFile(zipname, 'r') as zip:
            zip.extractall(save_dir)

    def process_data(self, raw_data_dir, processed_data_dir):
        rnd_state = 123456
        train_ts = raw_data_dir + '/BIDMC32HR/BIDMC32HR_TRAIN.ts'
        test_ts = raw_data_dir + '/BIDMC32HR/BIDMC32HR_TEST.ts'
        X_train, y_train = self.load_from_tsfile_to_dataframe(train_ts)
        X_test, y_test = self.load_from_tsfile_to_dataframe(test_ts)
        all_frames = pd.concat((X_train, X_test))

        ## Bit taken from https://github.com/jambo6/neuralRDEs/blob/master/get_data/tsr.py
        tensor_labels = torch.Tensor(np.concatenate((y_train, y_test)))
        tensor_data = []
        for idx in range(all_frames.shape[0]):
            tensor_data.append(torch.Tensor(pd.concat(all_frames.iloc[idx].values, axis=1).values))
        tensor_data = torch.stack(tensor_data)
        dataset = TensorDataset(tensor_data, tensor_labels)

        train_dataset, valid_test_dataset = torch.utils.data.random_split(dataset, [5565, 2384], generator=torch.Generator().manual_seed(123456))
        test_dataset, valid_dataset = torch.utils.data.random_split(valid_test_dataset, [1192, 1192], generator=torch.Generator().manual_seed(123456))
        torch.save(train_dataset, processed_data_dir + '/training.pt')
        torch.save(test_dataset, processed_data_dir + '/test.pt')
        torch.save(valid_dataset, processed_data_dir + '/validation.pt')

    def get_heart_rate(self):
        data_dir = os.getcwd()  + '/data/heart_rate'
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        raw_data_dir = data_dir + '/raw'
        if os.path.isdir(raw_data_dir):
            print("Data already downloaded")
        else:
            os.mkdir(raw_data_dir)
            print("Downloading data")
            self.download(raw_data_dir)
            print("Data download finished")

        processed_data_dir = data_dir + '/processed'
        if os.path.isdir(processed_data_dir):
            print("Data already processed")
        else:
            os.mkdir(processed_data_dir)
            self.process_data(raw_data_dir, processed_data_dir)
            print("Finished processing data")

        train_dataset = torch.load(processed_data_dir + '/training.pt')
        test_dataset = torch.load(processed_data_dir + '/test.pt')
        valid_dataset = torch.load(processed_data_dir + '/validation.pt')

        return train_dataset, test_dataset, valid_dataset
    

class sMNIST():
    def __init__(self):
        pass
    def get_data(self, bs_train,bs_test):
        train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

        test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                                train=False,
                                                transform=transforms.ToTensor())

        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [57000,3000])

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=bs_train,
                                                shuffle=True)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                batch_size=bs_test,
                                                shuffle=False)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=bs_test,
                                                shuffle=False)

        return train_loader, valid_loader, test_loader
    

class nCIFAR():
    def __init__(self):
        pass
    def get_data(self,bs_train,bs_test):
        train_dataset = torchvision.datasets.CIFAR10(root='../../data/cifar10/',
                                                    train=True,
                                                    transform=transforms.ToTensor(),
                                                    download=True)

        test_dataset = torchvision.datasets.CIFAR10(root='../../data/cifar10/',
                                                    train=False,
                                                    transform=transforms.ToTensor())

        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [48000,2000])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=bs_train,
                                                shuffle=True)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                batch_size=bs_test,
                                                shuffle=False)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=bs_test,
                                                shuffle=False)

        return train_loader, valid_loader, test_loader

'''
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

class Speech():
    def __init__(self):
        self.labels = ['backward',
                        'bed',
                        'bird',
                        'cat',
                        'dog',
                        'down',
                        'eight',
                        'five',
                        'follow',
                        'forward',
                        'four',
                        'go',
                        'happy',
                        'house',
                        'learn',
                        'left',
                        'marvin',
                        'nine',
                        'no',
                        'off',
                        'on',
                        'one',
                        'right',
                        'seven',
                        'sheila',
                        'six',
                        'stop',
                        'three',
                        'tree',
                        'two',
                        'up',
                        'visual',
                        'wow',
                        'yes',
                        'zero']

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))


    def index_to_label(self, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]
    
    def pad_sequence(self, batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)


    def collate_fn(self, batch):

        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [self.label_to_index(label)]

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets
    
    def get_data(self, batch_size=256, device='cuda'):
        train_set = SubsetSC("training")
        val_set = SubsetSC("validation")
        test_set = SubsetSC("testing")  

        return train_set, val_set, test_set
    



class Google12():
    def __init__(self):
        pass

    def download(self, raw_data_dir):
        url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        save_dir = raw_data_dir
        tarname = save_dir + '/google_speech_commands_v2.tar.gz'
        ## download zipped data
        urllib.request.urlretrieve(url, tarname)
        ## unzip:
        tar = tarfile.open(tarname, "r:gz")
        tar.extractall(save_dir)
        tar.close()

    def move_files(self, original_fold, data_fold, data_filename):
        with open(data_filename) as f:
            for line in f.readlines():
                vals = line.split('/')
                dest_fold = os.path.join(data_fold, vals[0])
                if not os.path.exists(dest_fold):
                    os.mkdir(dest_fold)
                shutil.move(os.path.join(original_fold, line[:-1]), os.path.join(data_fold, line[:-1]))

    def create_train_fold(self, original_fold, data_fold, test_fold):
        # list dirs
        dir_names = list()
        for file in os.listdir(original_fold):
            if os.path.isdir(os.path.join(original_fold, file)):
                dir_names.append(file)

        # build train fold
        for file in os.listdir(original_fold):
            if os.path.isdir(os.path.join(original_fold, file)) and file in dir_names:
                shutil.move(os.path.join(original_fold, file), os.path.join(data_fold, file))

    def make_dataset(self, gcommands_fold, out_path):
        validation_path = os.path.join(gcommands_fold, 'validation_list.txt')
        test_path = os.path.join(gcommands_fold, 'testing_list.txt')

        valid_fold = os.path.join(out_path, 'valid')
        test_fold = os.path.join(out_path, 'test')
        train_fold = os.path.join(out_path, 'train')

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(valid_fold):
            os.mkdir(valid_fold)
        if not os.path.exists(test_fold):
            os.mkdir(test_fold)
        if not os.path.exists(train_fold):
            os.mkdir(train_fold)

        self.move_files(gcommands_fold, test_fold, test_path)
        self.move_files(gcommands_fold, valid_fold, validation_path)
        self.create_train_fold(gcommands_fold, train_fold, test_fold)


    def google12_v2(self):
        data_dir = 'data/google_speech_command'
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        raw_data_dir = data_dir + '/raw'
        if os.path.isdir(raw_data_dir):
            print("Data already downloaded")
        else:
            print("Downloading data")
            os.mkdir(raw_data_dir)
            self.download(raw_data_dir)
            print("Data download finished")

        processed_data_dir = data_dir + '/processed'
        if os.path.isdir(processed_data_dir):
            print("Data already processed")
        else:
            os.mkdir(processed_data_dir)
            self.make_dataset(raw_data_dir, processed_data_dir)
        print("Finished processing data")




AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

GSCmdV2Categs = {
            'unknown': 0,
            'silence': 1,
            '_unknown_': 0,
            '_silence_': 1,
            '_background_noise_': 1,
            'yes': 2,
            'no': 3,
            'up': 4,
            'down': 5,
            'left': 6,
            'right': 7,
            'on': 8,
            'off': 9,
            'stop': 10,
            'go': 11}

def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx.get(target, 0))
                    spects.append(item)
    return spects

def spect_loader(path, window_size, window_stride, window, normalize, max_len=125):
    y, sr = librosa.load(path, sr=None)
    librosa_melspec = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024,
                                                     hop_length=128, power=1.0,
                                                     n_mels=80, fmin=40.0, fmax=sr / 2)
    spect = librosa.power_to_db(librosa_melspec, ref=np.max)

    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)


    return spect


class GCommandLoader(data.Dataset):
    """A google command data set loader

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=126):
        class_to_idx = GSCmdV2Categs
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return len(self.spects)


class Fitzhigh_Namuno():
    def __init__(self):
        self.I = 0.5
        self.a = 0.7
        self.b = 0.8
        self.eps = 1./50.

    def FHN_rhs(self,t,x):
        dim1 = x[0] - (x[0]**3)/3. - x[1] + self.I
        dim2 = self.eps*(x[0] + self.a - self.b*x[1])

        out = np.stack((dim1,dim2)).T

        return out

    def get_data(self,N,T=1000):
        data_x = []
        data_y = []
        for i in range(N):
            t = np.linspace(0,400,T+1)
            x0 = np.array([float(np.random.rand(1))*2.-1.,0.])
            sol = integrate.solve_ivp(self.FHN_rhs, [0,400], x0, t_eval=t)
            data_x.append(sol.y[0,:-1])
            data_y.append(sol.y[0,-1])

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        return data_x.reshape(N,T,1), data_y.reshape(N,1,1)



class LOB():
    def __init__(self):
        self.lag = 1000
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.max_id = None

    def get_scaler(self, scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

    def load_data(self):
        # Loading 
        # path = '/nfs/home/alvaroa/ML/Transformer/AAPL_lw=20000_s=1000_cleaned_full'
        # path = '/nfs/home/alvaroa/ML/Transformer/AAPL_lw=20000_s=1000_cleaned'
        path = '/nfs/home/alvaroa/ML/Transformer/AAPL_lw=1000_s=10000_cleaned'


        df = pd.read_csv(f'{path}.csv')

        # Cleaning 
        df = df.drop(labels="Side", axis=1)
        df = df.drop(columns=['Unnamed: 0'])
        df_x = df.loc[:,'time':'Order Size'].fillna(0)
        df_y = df.loc[:,'Filltime']
        df_index = df.loc[:,'index']
        self.max_id = max(df['index'])

        # Train/Val/Test split
        x_train, x_val, x_test = df_x.iloc[:int(self.max_id*self.train_ratio)*self.lag,:], \
                            df_x.iloc[int(self.max_id*self.train_ratio)*self.lag:int(self.max_id*(self.train_ratio+self.val_ratio))*self.lag,:], \
                            df_x.iloc[int(self.max_id*(self.train_ratio+self.val_ratio))*self.lag:,:]


        y_train, y_val, y_test = df_y.iloc[:int(self.max_id*self.train_ratio)*self.lag], \
                                df_y.iloc[int(self.max_id*self.train_ratio)*self.lag:int(self.max_id*(self.train_ratio+self.val_ratio))*self.lag], \
                                df_y.iloc[int(self.max_id*(self.train_ratio+self.val_ratio))*self.lag:]

        y_train, y_val, y_test = y_train.dropna(), y_val.dropna(), y_test.dropna()

        # Perform scaling
        scaler = self.get_scaler('robust')
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        # Reshape as (B, T, F)
        x_train = x_train.reshape(int(x_train.shape[0]/self.lag), self.lag, -1)
        x_val = x_val.reshape(int(x_val.shape[0]/self.lag), self.lag, -1)
        x_test = x_test.reshape(int(x_test.shape[0]/self.lag), self.lag, -1)
        y_train, y_val, y_test = y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()

        x_train = x_train.astype(np.float32)
        x_val = x_val.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)

        #Convert to torch format
        train_dataset = TensorDataset(Tensor(x_train).float(), Tensor(y_train).long())
        test_dataset = TensorDataset(Tensor(x_test).float(), Tensor(y_test).long())
        valid_dataset = TensorDataset(Tensor(x_val).float(), Tensor(y_val).long())


        return train_dataset, valid_dataset, test_dataset










'''


