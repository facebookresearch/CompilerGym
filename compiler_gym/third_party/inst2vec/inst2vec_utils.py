# NCC: Neural Code Comprehension
# https://github.com/spcl/ncc
# Copyright 2018 ETH Zurich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""inst2vec utility functions"""
import datetime
import pickle
import re
import sys

# Maximum number of bytes to pickle in one chunk
max_bytes = 2 ** 31 - 1


def safe_pickle(data, file):
    """Pickle big files safely, processing them in chunks

    :param data: data to be pickled
    :param file: file to pickle it into
    """
    pickle_out = pickle.dumps(data)
    n_bytes = sys.getsizeof(pickle_out)
    with open(file, "wb") as f:
        count = 0
        for i in range(0, n_bytes, max_bytes):
            f.write(pickle_out[i : min(n_bytes, i + max_bytes)])
            count += 1


def set_file_signature(param, data_folder, set_from_date_time=False):
    """
    Set file signature to differentiate between embedding trainings
    :param param: parameters of the inst2vec training
    :param data_folder: string containing the path to the parent directory of raw data sub-folders
    :param set_from_date_time: set file signature according to time and date instead of parameters
    :return: file signature
    """
    if set_from_date_time:
        file_signature = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
    else:
        file_signature = (
            "_"
            + re.sub(r"/", "_", data_folder)
            + "_d-"
            + str(param["embedding_size"])
            + "_m-"
            + str(param["mini_batch_size"])
            + "_s-"
            + str(param["num_sampled"])
            + "_e-"
            + str(param["learning_rate"])
            + "_r-"
            + str(param["beta"])
            + "_cw-"
            + str(param["context_width"])
            + "_N-"
            + str(param["num_epochs"])
        )

    print("File signature: ", file_signature)
    return file_signature
