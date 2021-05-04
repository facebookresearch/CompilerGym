"""This module defines an API for processing LLVM-IR with inst2vec."""
import pickle
from typing import List

import numpy as np

from compiler_gym.third_party.inst2vec import inst2vec_preprocess
from compiler_gym.util.runfiles_path import runfiles_path

_PICKLED_VOCABULARY = runfiles_path(
    "compiler_gym/third_party/inst2vec/dictionary.pickle"
)
_PICKLED_EMBEDDINGS = runfiles_path(
    "compiler_gym/third_party/inst2vec/embeddings.pickle"
)


class Inst2vecEncoder:
    """An LLVM encoder for inst2vec."""

    def __init__(self):
        # TODO(github.com/facebookresearch/CompilerGym/issues/122): Lazily
        # instantiate inst2vec encoder.
        with open(str(_PICKLED_VOCABULARY), "rb") as f:
            self.vocab = pickle.load(f)

        with open(str(_PICKLED_EMBEDDINGS), "rb") as f:
            self.embeddings = pickle.load(f)

        self.unknown_vocab_element = self.vocab["!UNK"]

    def preprocess(self, ir: str) -> List[str]:
        """Produce a list of pre-processed statements from an IR."""
        lines = [[x] for x in ir.split("\n")]
        try:
            structs = inst2vec_preprocess.GetStructTypes(ir)
            for line in lines:
                for struct, definition in structs.items():
                    line[0] = line[0].replace(struct, definition)
        except ValueError:
            pass

        preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
        preprocessed_texts = [
            inst2vec_preprocess.PreprocessStatement(x[0]) if len(x) else ""
            for x in preprocessed_lines
        ]
        return [x for x in preprocessed_texts if x]

    def encode(self, preprocessed: List[str]) -> List[int]:
        """Produce embedding indices for a list of pre-processed statements."""
        return [
            self.vocab.get(statement, self.unknown_vocab_element)
            for statement in preprocessed
        ]

    def embed(self, encoded: List[int]) -> np.ndarray:
        """Produce a matrix of embeddings from a list of encoded statements."""
        return np.vstack([self.embeddings[index] for index in encoded])
