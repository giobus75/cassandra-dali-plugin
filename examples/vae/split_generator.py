# Copyright 2022 CRS4 (http://www.crs4.it/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from crs4.cassandra_utils._split_generator import split_generator

class animeface_split_generator(split_generator):
    def __init__(
        self, id_col="patch_id", data_col="data", label_type="none"
    ):
        super().__init__(id_col=id_col, data_col=data_col, label_col=None, label_type=label_type)

    def create_split(self, split_ratio_list, balance=None):
        """
        This method populates the class attributr split_metadata with split information
        @ split_ratio_list: a weight vector with an element for each split (ex. [7, 2, 1]). The vector is normalized before the computation
        @ balance: a string {'random'|'original'} or a weight vector, an element for each class. The vector is normalized before the computation
        """

        df = self._df
        row_keys = df[self._id_col].tolist()
        rows = df.shape[0]

        sum_split_ratio = np.sum(split_ratio_list)
        split_bounds = (np.array(split_ratio_list) / sum_split_ratio * rows).astype(np.int64)

        split = []
        
        index = df.index.tolist()
        np.random.shuffle(index)

        # randomly sample tot_num indexes
        start = 0
        print (rows, split_bounds)
        for bound in split_bounds:
            stop = start + bound
            s = index[start:stop]
            split.append(s)
            start = stop

        split = [np.array(i) for i in split]

        row_keys = self._df[self._id_col].to_numpy()
        self.split_metadata["row_keys"] = row_keys
        self.split_metadata["split"] = split
        self.split_metadata["label_type"] = self._label_type
