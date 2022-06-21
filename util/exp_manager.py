import hashlib
import os
import pickle
import random
from datetime import datetime
from typing import Union

import pandas as pd

from util.helper import concat_params_res


class ExperimentManager:
    """
    Loads and save experiment params and results in csv
    """

    def __init__(
        self,
        folder_name="experiments",
        random_seed=1000,
    ):
        self.folder_name = folder_name
        self.random_seed = random_seed

        if not os.path.exists(self.folder_name):
            os.mkdir(self.folder_name)

    def update_csv(
        self,
        exp_params: Union[dict, pd.DataFrame],
        insert_date: bool = True,
        csv_name=None,
        insert_pickle: Union[bool, str] = False,
    ):
        """
        Update csv with the experiment parameters as columns.
        Optional to include `insert_date` as a column.

        If insert pickle is string , assume that an explicit string is already given.
        Otherwise will encode the `exp_params` accordingly.
        """

        # insert date as a column if enabled
        if insert_date:
            new_rows = {"date": self.get_nowdate()}
        else:
            new_rows = {}

        # handle data type of exp_params
        if isinstance(exp_params, dict):
            new_rows.update(exp_params)
            new_rows = pd.DataFrame([new_rows])
        elif isinstance(exp_params, pd.DataFrame):
            new_rows = pd.concat((pd.DataFrame([new_rows]), exp_params), axis=1)
        else:
            raise TypeError("`exp_params` type not accepted!")

        # check if pickle column is needed to be inserted
        if (isinstance(insert_pickle, bool) and insert_pickle == True) or isinstance(
            insert_pickle, str
        ):
            if isinstance(insert_pickle, bool):
                pickle_encode = self.encode(exp_params)
            elif isinstance(insert_pickle, str):
                pickle_encode = insert_pickle
            pickle_col = pd.DataFrame([{"pickle": pickle_encode}])
            new_rows = pd.concat((pickle_col, new_rows), axis=1)

        # load current results from file
        file_path = os.path.join(self.folder_name, csv_name)
        exp_results = self.load_csv(file_path)

        # check if exp_results existed
        if exp_results is not None:
            # exp_results = exp_results.reset_index(drop=True).append(new_rows)
            # exp_results = exp_results.reset_index(drop=True)
            exp_results = exp_results.append(new_rows)
        else:
            exp_results = new_rows

        # save the csv
        exp_results.to_csv(file_path, index=False)

    def load_csv(self, file_path: str):
        if os.path.exists(file_path):
            exp_results = pd.read_csv(file_path)
        else:
            exp_results = None
        return exp_results

    def get_nowdate(self):
        date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        return date_time

    def concat_params_res(self, *args, **kwargs):
        return concat_params_res(*args, **kwargs)

    def encode(self, exp_params={}, return_pickle=True, return_compact=True):
        """
        Parameters
        ----------
        return_compact : boolean
            Returns a compact version of the encoded data which is formed from the first+last ten characters
            of the encoded.
        """

        encoded_message = hashlib.sha1(str.encode(str(exp_params)))
        encode_data_name = encoded_message.hexdigest()
        encode_data_name = encode_data_name.replace("b", "")
        encode_data_name = encode_data_name.replace("'", "")

        # perform shuffling
        random.seed(self.random_seed)
        encode_data_name = list(encode_data_name)
        random.shuffle(encode_data_name)
        encode_data_name = "".join(encode_data_name)
        if return_compact and len(encode_data_name) >= 20:
            return (
                encode_data_name[:10]
                + encode_data_name[-10:]
                + (".p" if return_pickle else "")
            )
        else:
            return encode_data_name + (".p" if return_pickle else "")

    def encode_pickle(self, exp_params, data):
        """
        Encodes the experiment parameters to a unique string
        and save the data into the designated pickle file.

        Parameters
        ----------
        exp_params : str or dict or pd.DataFrame
            If str, directly use the provided string as the name.
            Otherwise encode the parameters of experiments.

        data : any
            Data to be pickled for the experiment
        """
        if isinstance(exp_params, str):
            filename = exp_params
        else:
            filename = self.encode(exp_params=exp_params, return_pickle=True)
        if not os.path.exists(os.path.join(self.folder_name, "pickles")):
            os.mkdir(os.path.join(self.folder_name, "pickles"))
        pickle.dump(
            data, open(os.path.join(self.folder_name, "pickles", filename), "wb")
        )
        return filename

    def load_encoded_pickle(self, exp_params):
        """
        Loads the encoded pickle file. If exp_params is str, directly use it as the pickle file name.
        Otherwise, encode the exp_params and then load the file with its name.
        """
        if isinstance(exp_params, str):
            filename = exp_params
        else:
            filename = self.encode(exp_params=exp_params, return_pickle=True)
        return pickle.load(
            open(os.path.join(self.folder_name, "pickles", filename), "rb")
        )
