"""Module for handling embedding and unmasking of protein sequences using Hugging Face and TAPE models."""

#
# MIT License
#
# Copyright (c) 2024 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import logging
import math
import random
from abc import ABC, abstractmethod
from itertools import product as iter_product
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tape.datasets import pad_sequences
from tape.registry import registry
from tape.tokenizers import TAPETokenizer
from transformers import AutoModel, AutoTokenizer, T5Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCache:
    """Caches models to avoid reloading them multiple times."""

    def __init__(self):
        """
        Initializes the model cache as an empty dictionary.
        """
        self.cache = {}

    def get(self, key):
        """
        Retrieves a model from the cache.

        Args:
            key: The key associated with the model in the cache.

        Returns:
            The model associated with the given key, or None if the key is not found in the cache.
        """
        return self.cache.get(key)

    def add(self, key, model):
        """
        Adds a model to the cache.

        Args:
            key: The key to associate with the model in the cache.
            model: The model to be cached.
        """
        self.cache[key] = model


ENZEPTIONAL_MODEL_CACHE = ModelCache()


class ModelLoader(ABC):
    """Abstract base class for loading models."""

    @abstractmethod
    def load_model(self, model_path, cache_key, cache_dir):
        """
        Loads a model given the path and cache key.

        Args:
            model_path: The path to the model that needs to be loaded.
            cache_key: The key used to identify and store the model in the cache.
            cache_dir: Optional directory where the model is cached.

        Returns:
            The loaded model.
        """
        pass


class HuggingFaceModelLoader(ModelLoader):
    """Loads and caches Hugging Face models."""

    def load_model(self, model_path, cache_key, cache_dir):
        """
        Loads a Hugging Face model from the specified path, caching it for future use.

        Args:
            model_path: The path to the Hugging Face model.
            cache_key: The key used to identify and store the model in the cache.
            cache_dir: Optional directory where the model is cached.

        Returns:
            The loaded Hugging Face model.
        """
        model = ENZEPTIONAL_MODEL_CACHE.get(cache_key)
        if not model:
            model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir).eval()
            ENZEPTIONAL_MODEL_CACHE.add(cache_key, model)
        return model


class TapeModelLoader(ModelLoader):
    """Loads and caches TAPE models."""

    def load_model(self, model_path, cache_key, cache_dir):
        """
        Loads a TAPE model from the specified path, caching it for future use.

        Args:
            model_path: The path to the TAPE model.
            cache_key: The key used to identify and store the model in the cache.
            cache_dir: Optional directory where the model is cached.

        Returns:
            The loaded TAPE model.
        """
        model = ENZEPTIONAL_MODEL_CACHE.get(cache_key)
        if not model:
            model = registry.get_task_model(
                model_path, "embed", load_dir=model_path
            ).eval()
            ENZEPTIONAL_MODEL_CACHE.add(cache_key, model)
        return model


class TokenizerLoader(ABC):
    """Abstract base class for loading tokenizers."""

    @abstractmethod
    def load_tokenizer(self, tokenizer_path):
        """
        Loads a tokenizer given the path.

        Args:
            tokenizer_path: The path to the tokenizer that needs to be loaded.

        Returns:
            The loaded tokenizer.
        """
        pass


class HuggingFaceTokenizerLoader(TokenizerLoader):
    """Loads and caches Hugging Face tokenizers."""

    def load_tokenizer(self, tokenizer_path):
        """
        Loads a Hugging Face tokenizer from the specified path, caching it for future use.

        Args:
            tokenizer_path: The path to the Hugging Face tokenizer.

        Returns:
            The loaded Hugging Face tokenizer.
        """
        tokenizer = ENZEPTIONAL_MODEL_CACHE.get(f"tokenizer_{tokenizer_path}")
        if not tokenizer:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception:
                tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
            ENZEPTIONAL_MODEL_CACHE.add(f"tokenizer_{tokenizer_path}", tokenizer)
        return tokenizer


class TapeTokenizerLoader(TokenizerLoader):
    """Loads TAPE tokenizers."""

    def load_tokenizer(self, tokenizer_path):
        """
        Loads a TAPE tokenizer.

        Args:
            tokenizer_path: The path to the TAPE tokenizer. This argument is ignored because the TAPE tokenizer uses a fixed vocabulary.

        Returns:
            The loaded TAPE tokenizer.
        """
        return TAPETokenizer(vocab="iupac")


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, samples):
        """
        Embeds a list of protein sequences.

        Args:
            samples: A list of protein sequences to be embedded.

        Returns:
            A numpy array containing the embeddings for each sequence.
        """
        pass


class HuggingFaceEmbedder(EmbeddingModel):
    """Embeds protein sequences using a Hugging Face model."""

    def __init__(
        self,
        model_loader,
        tokenizer_loader,
        model_path,
        tokenizer_path,
        cache_dir,
        device,
    ):
        """
        Initializes the Hugging Face embedder with the model and tokenizer loaders.

        Args:
            model_loader: The loader responsible for loading the Hugging Face model.
            tokenizer_loader: The loader responsible for loading the Hugging Face tokenizer.
            model_path: The path to the Hugging Face model.
            tokenizer_path: The path to the Hugging Face tokenizer.
            cache_dir: Optional directory where the model is cached.
            device: The device on which to load the model (e.g., 'cpu' or 'cuda').

        Returns:
            None
        """
        self.model = model_loader.load_model(
            model_path, f"embedding_{model_path}", cache_dir
        ).to(device)
        self.tokenizer = tokenizer_loader.load_tokenizer(tokenizer_path)
        self.device = device

    def embed(self, samples):
        """
        Embeds protein sequences using a Hugging Face model.

        Args:
            samples: A list of protein sequences to be embedded.

        Returns:
            A numpy array containing the embeddings for each sequence.
        """
        inputs = self.tokenizer(
            samples, add_special_tokens=True, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            sequence_embeddings = self.model(**inputs)[0].cpu().detach().numpy()
        sequence_lengths = inputs["attention_mask"].sum(1)
        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(
                    sequence_embeddings, sequence_lengths
                )
            ]
        )


class TapeEmbedder(EmbeddingModel):
    """Embeds protein sequences using a TAPE model."""

    def __init__(self, model_loader, tokenizer_loader, model_path, device):
        """
        Initializes the TAPE embedder with the model and tokenizer loaders.

        Args:
            model_loader: The loader responsible for loading the TAPE model.
            tokenizer_loader: The loader responsible for loading the TAPE tokenizer.
            model_path: The path to the TAPE model.
            device: The device on which to load the model (e.g., 'cpu' or 'cuda').

        Returns:
            None
        """
        self.model = model_loader.load_model(
            model_path, f"embedding_{model_path}", None
        ).to(device)
        self.tokenizer = tokenizer_loader.load_tokenizer("")
        self.device = device

    def embed(self, samples):
        """
        Embeds protein sequences using a TAPE model.

        Args:
            samples: A list of protein sequences to be embedded.

        Returns:
            A numpy array containing the embeddings for each sequence.
        """
        token_ids: Dict[str, Any] = {"ids": [], "mask": []}
        for sequence in samples:
            encoded_sequence = self.tokenizer.encode(sequence)
            token_ids["ids"].append(encoded_sequence)
            token_ids["mask"].append(np.ones_like(encoded_sequence))
        input_ids = torch.from_numpy(pad_sequences(token_ids["ids"])).to(self.device)
        input_mask = torch.from_numpy(pad_sequences(token_ids["mask"])).to(self.device)
        inputs = {"input_ids": input_ids, "input_mask": input_mask}
        with torch.no_grad():
            sequence_embeddings = self.model(**inputs)[0].cpu().detach().numpy()
        sequence_lengths = input_mask.sum(1)
        return np.array(
            [
                sequence_embedding[:sequence_length].mean(0)
                for sequence_embedding, sequence_length in zip(
                    sequence_embeddings, sequence_lengths
                )
            ]
        )


class UnmaskingModel(ABC):
    """Abstract base class for unmasking models."""

    @abstractmethod
    def unmask(self, sequence, top_k):
        """
        Unmasks a sequence to predict the original amino acids.

        Args:
            sequence: The sequence containing masked tokens.
            top_k: The number of top predictions to return.

        Returns:
            A list of the top-k predicted sequences with the masked tokens replaced.
        """
        pass


class HuggingFaceUnmasker(UnmaskingModel):
    """Unmasks protein sequences using a Hugging Face model."""

    def __init__(
        self,
        model_loader,
        tokenizer_loader,
        model_path,
        tokenizer_path,
        cache_dir,
        device,
    ):
        """
        Initializes the Hugging Face unmasker with the model and tokenizer loaders.

        Args:
            model_loader: The loader responsible for loading the Hugging Face model.
            tokenizer_loader: The loader responsible for loading the Hugging Face tokenizer.
            model_path: The path to the Hugging Face model.
            tokenizer_path: The path to the Hugging Face tokenizer.
            cache_dir: Optional directory where the model is cached.
            device: The device on which to load the model (e.g., 'cpu' or 'cuda').

        Returns:
            None
        """
        self.model = model_loader.load_model(
            model_path, f"unmasking_{model_path}", cache_dir
        ).to(device)
        self.tokenizer = tokenizer_loader.load_tokenizer(tokenizer_path)
        self.device = device

    def unmask(self, sequence, top_k=2):
        """
        Unmasks a sequence using a Hugging Face model and returns the top predictions.

        Args:
            sequence: The sequence containing masked tokens to be unmasked.
            top_k: The number of top predictions to return for each masked token.

        Returns:
            A list of the top-k predicted sequences with the masked tokens replaced.
        """
        inputs = self.tokenizer(
            sequence, return_tensors="pt", add_special_tokens=True, padding=True
        ).to(self.device)
        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id
        )[1]
        with torch.no_grad():
            logits = self.model(inputs["input_ids"]).logits
        mask_token_logits = logits[0, mask_token_index, :]
        top_tokens = [
            torch.topk(mask_token_logits, top_k, dim=1).indices[i].tolist()
            for i in range(len(mask_token_index))
        ]
        mutated_sequences = []
        tmp_top_tokens = [tuple(tokens) for tokens in top_tokens]
        if len(set(tmp_top_tokens)) == 1:
            for i in range(top_k):
                temp_sequence = sequence.split(" ")
                for mask_index in mask_token_index.cpu().numpy():
                    temp_sequence[mask_index - 1] = tmp_top_tokens[0][i]
                mutated_sequences.append("".join(temp_sequence))
        else:
            for combination in list(iter_product(*tmp_top_tokens)):
                temp_sequence = sequence.split(" ")
                for i, mask_index in enumerate(mask_token_index.cpu().numpy()):
                    temp_sequence[mask_index - 1] = combination[i]
                mutated_sequences.append("".join(temp_sequence))
        return mutated_sequences


def mutate_sequence_with_variant(sequence: str, variant: str) -> str:
    """Applies a specified variant mutation to an amino acid sequence.

    Args:
        sequence (str): The original amino acid sequence.
        variant (str): The variant to apply, formatted as a string.

    Returns:
        str: The mutated amino acid sequence.
    """
    mutated_sequence = list(sequence)
    for variant_string in variant.split("/"):
        index = int(variant_string[1:-1]) - 1
        mutated_sequence[index] = variant_string[-1]
    return "".join(mutated_sequence)


def sanitize_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merges overlapping intervals into a single interval.

    Args:
        intervals (List[Tuple[int, int]]): A list of
        start and end points of intervals.

    Returns:
        List[Tuple[int, int]]: A list of merged intervals.
    """
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in intervals:
        if not merged or merged[-1][1] < start:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def round_up(number: float) -> int:
    """Rounds up a floating-point number to the nearest integer.

    Args:
        number (float): The number to round up.

    Returns:
        int: The rounded-up integer.
    """
    return math.ceil(number)


def sanitize_intervals_with_padding(
    intervals: List[Tuple[int, int]], pad_value: int, max_value: int
) -> List[Tuple[int, int]]:
    """Pads and sanitizes intervals within a given range.

    Args:
        intervals (List[Tuple[int, int]]): A list of intervals.
        pad_value (int): The value to pad intervals with.
        max_value (int): The maximum value for the range of intervals.

    Returns:
        List[Tuple[int, int]]: A list of padded and sanitized intervals.
    """

    def pad_interval(
        interval: Tuple[int, int], pad: int, max_val: int
    ) -> Tuple[int, int]:
        """Pads an individual interval within the constraints of a maximum value.

        Args:
            interval (Tuple[int, int]): The interval to pad.
            pad (int): The padding value.
            max_val (int): The maximum value for the interval.

        Returns:
            Tuple[int, int]: The padded interval.
        """
        start, end = interval
        interval_length = end - start
        padding_needed = max(0, pad - interval_length) // 2

        padded_start = max(0, start - padding_needed)
        padded_end = min(max_val, end + padding_needed)

        if padded_end > max_val:
            padded_start = max(0, padded_start - (padded_end - max_val))
        return padded_start, padded_end

    padded_intervals = [
        pad_interval(interval, pad_value, max_value) for interval in intervals
    ]
    return sanitize_intervals(padded_intervals)


def reconstruct_sequence_with_mutation_range(
    sequence: str,
    mutated_sequence_range: str,
    intervals: List[Tuple[int, int]],
) -> str:
    """Reconstructs a sequence by inserting a mutated sequence
    range at specific intervals.

    Args:
        sequence (str): The original sequence.
        mutated_sequence_range (str): The range of the sequence to be mutated.
        intervals (List[Tuple[int, int]]): The intervals where
        mutations are applied.

    Returns:
        str: The reconstructed sequence with mutations.
    """
    mutated_sequence = list(sequence)
    range_index = 0
    for start, end in intervals:
        size_fragment = end - start
        mutated_sequence[start:end] = list(
            mutated_sequence_range[range_index : range_index + size_fragment]
        )
        range_index += size_fragment
    return "".join(mutated_sequence)


class SelectionGenerator:
    """
    A generator for selecting top sequences based on their scores.
    """

    def selection(
        self,
        pool_of_sequences: List[Dict[str, Any]],
        k: float = 0.8,
    ) -> List[Any]:
        """Selects a subset of sequences from a pool based on their scores.

        Args:
            pool_of_sequences (List[Dict[str, Any]]): A list of
            dictionaries, each containing a sequence and its score.
            k (float): A fraction representing the proportion
            of top sequences to select. Defaults to 0.8.

        Returns:
            List[Any]: A list of the top k sequences based on scores.
        """
        n_samples_to_select = int(len(pool_of_sequences) * k)
        return list(sorted(pool_of_sequences, key=lambda d: d["score"], reverse=True))[
            :n_samples_to_select
        ]


class CrossoverGenerator:
    """
    A generator for performing crossover operations between sequences.
    """

    def __init__(self, threshold_probability: float = 0.5) -> None:
        """Initializes the CrossoverGenerator with a specified
        threshold probability.

        Args:
            threshold_probability (float, optional): The probability
            threshold used in uniform crossover. Defaults to 0.5.
        """
        self.threshold_probability = threshold_probability

    def sp_crossover(self, a_sequence: str, another_sequence: str) -> Tuple[str, str]:
        """Performs a single point crossover between two sequences.

        Args:
            a_sequence (str): The first sequence for crossover.
            another_sequence (str): The second sequence for crossover.

        Returns:
            Tuple[str, str]: A tuple of two new sequences resulting
            from the crossover.
        """
        random_point = random.randint(1, len(a_sequence) - 2)
        return (
            a_sequence[:random_point] + another_sequence[random_point:],
            another_sequence[:random_point] + a_sequence[random_point:],
        )

    def uniform_crossover(
        self, a_sequence: str, another_sequence: str
    ) -> Tuple[str, str]:
        """Performs a uniform crossover between two sequences.

        Args:
            a_sequence (str): The first sequence for crossover.
            another_sequence (str): The second sequence for crossover.

        Returns:
            Tuple[str, str]: A tuple of two new sequences resulting
            from the crossover.
        """
        return (
            "".join(
                a if random.random() > self.threshold_probability else b
                for a, b in zip(a_sequence, another_sequence)
            ),
            "".join(
                b if random.random() > self.threshold_probability else a
                for a, b in zip(a_sequence, another_sequence)
            ),
        )
