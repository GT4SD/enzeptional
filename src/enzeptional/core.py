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
from abc import ABC, abstractmethod
from itertools import product as iter_product
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from joblib import load
import numpy as np
import pandas as pd
import xgboost as xgb

from .processing import (
    CrossoverGenerator,
    HuggingFaceEmbedder,
    HuggingFaceUnmasker,
    SelectionGenerator,
    sanitize_intervals,
    sanitize_intervals_with_padding,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MutationModelManager:
    """
    Manages mutation models for efficient reuse.
    """

    def load_model(
        self, embedding_model_path, tokenizer_path, is_tape_model=False, **kwargs
    ):
        """
        Loads a model from the given paths.

        Args:
            embedding_model_path: Path to the embedding model.
            tokenizer_path: Path to the tokenizer.
            is_tape_model: Bool, Default is False if Tape is not the model used.
            **kwargs: Additional arguments for model loading.

        Returns:
            An instance of the loaded model.
        """
        model = HuggingFaceUnmasker(
            model_path=embedding_model_path,
            tokenizer_path=tokenizer_path,
            cache_dir=kwargs.get("cache_dir", None),
            device=kwargs.get("device", "cpu"),
        )
        return model


class MutationStrategy(ABC):
    """
    Abstract base class for defining mutation strategies.
    """

    @abstractmethod
    def mutate(
        self, sequence: str, num_mutations: int, intervals: List[List[int]]
    ) -> List[str]:
        """Abstract method for mutating a sequence.

        Args:
            sequence: The original sequence to be mutated.
            num_mutations: The number of mutations to apply.

        Returns:
            The mutated sequence.
        """
        pass


class LanguageModelMutationStrategy(MutationStrategy):
    """
    Mutation strategy using a language model.
    """

    def __init__(self, mutation_model):
        """Initializes the mutation strategy with a given model.

        Args:
            mutation_model: The model to be used for mutation.
        """
        self.mutation_model = mutation_model
        self.top_k = 2

    def set_top_k(self, top_k: int):
        """Sets the top k mutations to consider during mutation.

        Args:
            top_k: The number of top mutations to consider.
        """
        self.top_k = top_k

    def mutate(
        self, sequence: str, num_mutations: int, intervals: List[List[int]]
    ) -> List[str]:
        """Mutates a sequence within specified intervals using the model.

        Args:
            sequence: The original sequence to be mutated.
            num_mutations: The number of mutations to introduce.
            intervals: Intervals within the sequence
            where mutations are allowed.

        Returns:
            A list of mutated sequences.
        """

        flat_intervals = [
            i
            for interval in intervals
            for i in range(interval[0], interval[1] + 1)
            if i < len(sequence)
        ]

        num_mutations = random.randint(1, num_mutations)

        chosen_positions = random.sample(
            flat_intervals, min(num_mutations, len(flat_intervals))
        )
        sequence_list = list(sequence)

        for pos in chosen_positions:
            sequence_list[pos] = self.mutation_model.tokenizer.mask_token

        masked_sequence = " ".join(sequence_list)

        return self.mutation_model.unmask(masked_sequence, self.top_k)


class TransitionMatrixMutationStrategy(MutationStrategy):
    """
    Mutation strategy based on a transition matrix.
    """

    def __init__(self, transition_matrix: str):
        """Initializes the mutation strategy with a transition matrix.

        Args:
            transition_matrix: Path to the CSV file containing
            the transition matrix.
        """
        logger.info(" USING TRNASITION MATRIX  ")
        self.transition_matrix = pd.read_csv(
            transition_matrix, index_col=None, header=0
        )
        self.top_k = 2

    def set_top_k(self, top_k: int):
        """Sets the top k mutations to consider during mutation.

        Args:
            top_k: The number of top mutations to consider.
        """

        self.top_k = top_k

    def mutate(
        self, sequence: str, num_mutations: int, intervals: List[List[int]]
    ) -> List[str]:
        """Mutates a sequence based on the transition matrix within
        specified intervals.

        Args:
            sequence: The original sequence to be mutated.
            num_mutations: The number of mutations to introduce.
            intervals: Intervals within the sequence
            where mutations are allowed.

        Returns:
            A list of mutated sequences.
        """

        flat_intervals = [
            i
            for interval in intervals
            for i in range(interval[0], interval[1] + 1)
            if i < len(sequence)
        ]

        num_mutations = random.randint(1, num_mutations)

        chosen_positions = random.sample(
            flat_intervals, min(num_mutations, len(flat_intervals))
        )

        mutated_sequences = []

        mutation_options = []
        for pos in chosen_positions:
            aa_probabilities = self.transition_matrix.iloc[pos]
            top_mutations = aa_probabilities.nlargest(self.top_k).index.tolist()
            mutation_options.append([(pos, aa) for aa in top_mutations])

        for mutation_combination in iter_product(*mutation_options):
            temp_sequence = list(sequence)
            for pos, new_aa in mutation_combination:
                temp_sequence[pos] = new_aa
            mutated_sequences.append("".join(temp_sequence))

        return mutated_sequences


class MutationFactory:
    """
    Factory class for creating mutation strategies based on configuration.
    """

    @staticmethod
    def get_mutation_strategy(mutation_config: Dict[str, Any]):
        """Retrieves a mutation strategy based on the provided configuration.

        Args:
            mutation_config: Configuration specifying
            the type of mutation strategy and its parameters.

        Raises:
            KeyError: If required configuration parameters are missing.
            ValueError: If the mutation type is unsupported.

        Returns:
            An instance of the specified mutation strategy.
        """
        if mutation_config["type"] == "language-modeling":
            mutation_model = MutationModelManager().load_model(
                embedding_model_path=mutation_config["embedding_model_path"],
                tokenizer_path=mutation_config["tokenizer_path"],
                is_tape_model=mutation_config.get("is_tape_model", False),
                unmasking_model_path=mutation_config.get("unmasking_model_path"),
            )
            return LanguageModelMutationStrategy(mutation_model)
        elif mutation_config["type"] == "transition-matrix":
            transition_matrix = mutation_config.get("transition_matrix")
            if transition_matrix is None:
                raise KeyError(
                    "Transition matrix not provided in mutation configuration."
                )
            return TransitionMatrixMutationStrategy(transition_matrix)
        else:
            raise ValueError("Unsupported mutation type")


class SequenceMutator:
    """
    Class for mutating sequences using a specified strategy.
    """

    def __init__(self, sequence: str, mutation_config: Dict[str, Any]):
        """Initializes the mutator with a sequence and a mutation strategy.

        Args:
            sequence: The sequence to be mutated.
            mutation_config: Configuration for
            the mutation strategy.
        """
        self.sequence = sequence
        self.mutation_strategy = MutationFactory.get_mutation_strategy(mutation_config)
        self.top_k = 2

    def set_top_k(self, top_k: int):
        """Sets the number of top mutations to consider in the mutation strategy.

        Args:
            top_k: The number of top mutations to consider.
        """
        self.top_k = top_k
        if isinstance(
            self.mutation_strategy,
            (LanguageModelMutationStrategy, TransitionMatrixMutationStrategy),
        ):
            self.mutation_strategy.set_top_k(top_k)

    def mutate_sequences(
        self,
        num_sequences: int,
        number_of_mutations: int,
        intervals: List[Tuple[int, int]],
        current_population: List[str],
        all_mutated_sequences: List[str],
    ) -> List[str]:
        """Generates a set of mutated sequences.

        Args:
            num_sequences: Number of mutated sequences to generate.
            number_of_mutations: Number of mutations to apply to
            each sequence.
            intervals: Intervals within the sequence
            where mutations are allowed.
            all_mutated_sequences: All sequences already mutated.

        Returns:
            A list of mutated sequences.
        """
        max_mutations = min(len(self.sequence), number_of_mutations)
        current_population = [self.sequence]

        mutated_sequences_set: List[str] = []
        while len(mutated_sequences_set) < num_sequences:
            for temp_sequence in current_population:
                new_mutations = self.mutation_strategy.mutate(
                    temp_sequence, max_mutations, intervals
                )
                new_mutations = [
                    i for i in new_mutations if i not in all_mutated_sequences
                ]
                mutated_sequences_set.extend(new_mutations)
                if len(mutated_sequences_set) >= num_sequences:
                    break
        return random.sample(mutated_sequences_set, num_sequences)


class SequenceScorer:
    """
    Scores protein sequences using a pre-trained embedding model and a trained machine learning model.
    Supports individual and batch scoring, with optional scaling and XGBoost support.
    """

    def __init__(
        self,
        protein_model: HuggingFaceEmbedder,
        scorer_filepath: str,
        use_xgboost: bool = False,
        scaler_filepath: Optional[str] = None,
    ):
        """
        Initializes the SequenceScorer with the given embedding model, scorer model, and optional scaler.

        Args:
            protein_model: Model used for generating protein embeddings.
            scorer_filepath: Path to the trained model for scoring.
            use_xgboost: Whether to use XGBoost as the scoring model (default is False).
            scaler_filepath: Path to a scaler for feature normalization (optional).
        """
        self.protein_model = protein_model
        self.scorer = load(scorer_filepath)
        self.use_xgboost = use_xgboost
        self.scaler = load(scaler_filepath) if scaler_filepath else None

    def score(
        self,
        sequence: str,
        substrate_embedding: np.ndarray,
        product_embedding: np.ndarray,
        concat_order: List[str],
    ) -> Dict[str, Any]:
        """
        Scores a single protein sequence by embedding it, concatenating it with substrate and product embeddings,
        and applying the scorer model.

        Args:
            sequence: The protein sequence to score.
            substrate_embedding: The embedding of the substrate.
            product_embedding: The embedding of the product.
            concat_order: The order in which to concatenate the embeddings.

        Returns:
            A dictionary containing the sequence and its computed score.
        """
        sequence_embedding = self.protein_model.embed([sequence])[0]
        embeddings = [sequence_embedding, substrate_embedding, product_embedding]
        ordered_embeddings = [
            embeddings[concat_order.index(item)] for item in concat_order
        ]
        combined_embedding = np.concatenate(ordered_embeddings).reshape(1, -1)

        if self.use_xgboost:
            if self.scaler:
                combined_embedding = self.scaler.transform(combined_embedding)
            score = self.scorer.predict(xgb.DMatrix(combined_embedding))[0]
        else:
            score = self.scorer.predict_proba(combined_embedding)[0][1]

        return {"sequence": sequence, "score": score}

    def score_batch(
        self,
        sequences: List[str],
        substrate_embedding: np.ndarray,
        product_embedding: np.ndarray,
        concat_order: List[str],
    ) -> List[Dict[str, float]]:
        """
        Scores a batch of protein sequences by embedding each one, concatenating it with substrate and product embeddings,
        and applying the scorer model.

        Args:
            sequences: List of protein sequences to score.
            substrate_embedding: The embedding of the substrate.
            product_embedding: The embedding of the product.
            concat_order: The order in which to concatenate the embeddings.

        Returns:
            A list of dictionaries containing the sequences and their computed scores.
        """
        sequence_embeddings = self.protein_model.embed(sequences)
        output = []
        for position, sequence_embedding in enumerate(sequence_embeddings):
            embeddings = [sequence_embedding, substrate_embedding, product_embedding]
            ordered_embeddings = [
                embeddings[concat_order.index(item)] for item in concat_order
            ]
            combined_embedding = np.concatenate(ordered_embeddings).reshape(1, -1)

            if self.use_xgboost:
                if self.scaler:
                    combined_embedding = self.scaler.transform(combined_embedding)
                score = self.scorer.predict(xgb.DMatrix(combined_embedding))[0]
            else:
                score = self.scorer.predict_proba(combined_embedding)[0][1]
            output.append({"sequence": sequences[position], "score": score})

        return output


class EnzymeOptimizer:
    """
    Optimizes protein sequences based on interaction with
    substrates and products.
    """

    def __init__(
        self,
        sequence: str,
        mutator: SequenceMutator,
        scorer: SequenceScorer,
        intervals: List[Tuple[int, int]],
        substrate_smiles: str,
        product_smiles: str,
        chem_model: HuggingFaceEmbedder,
        selection_generator: SelectionGenerator,
        crossover_generator: CrossoverGenerator,
        concat_order: List[str],
        batch_size: int = 2,
        selection_ratio: float = 0.5,
        perform_crossover: bool = False,
        crossover_type: str = "uniform",
        minimum_interval_length: int = 8,
        pad_intervals: bool = False,
        seed: int = 123,
    ):
        """Initializes the optimizer with models, sequences, and
        optimization parameters.


        Args:
            sequence: The initial protein sequence.
            protein_model: Model for protein embeddings.
            substrate_smiles: SMILES representation of the substrate.
            product_smiles: SMILES representation of the product.
            chem_model_path: Path to the chemical model.
            chem_tokenizer_path: Path to the chemical tokenizer.
            scorer_filepath: File path to the scoring model.
            mutator: The mutator for generating sequence variants.
            intervals: Intervals for mutation.
            batch_size: The number of sequences to process in one batch.
            top_k: Number of top mutations to consider.
            selection_ratio: Ratio of sequences to select after scoring.
            perform_crossover: Flag to perform crossover operation.
            crossover_type: Type of crossover operation.
            minimum_interval_length: Minimum length of mutation intervals.
            pad_intervals: Flag to pad the intervals.
            concat_order: Order of concatenating embeddings.
            scaler_filepath: Path to the scaler in case you are using the Kcat model.
            use_xgboost_scorer: Flag to specify if the fitness function is the Kcat.
        """
        self.sequence = sequence
        self.mutator = mutator
        self.scorer = scorer
        self.intervals = (
            sanitize_intervals(intervals) if intervals else [(0, len(sequence))]
        )
        self.batch_size = batch_size
        self.selection_ratio = selection_ratio
        self.perform_crossover = perform_crossover
        self.crossover_type = crossover_type
        self.concat_order = concat_order
        self.seed = seed
        random.seed(self.seed)

        self.chem_model = chem_model
        self.substrate_embedding = chem_model.embed([substrate_smiles])[0]
        self.product_embedding = chem_model.embed([product_smiles])[0]
        self.selection_generator = selection_generator
        self.crossover_generator = crossover_generator

        if pad_intervals:
            self.intervals = sanitize_intervals_with_padding(
                self.intervals, minimum_interval_length, len(sequence)
            )

    def optimize(
        self,
        num_iterations: int,
        num_sequences: int,
        num_mutations: int,
        time_budget: Optional[int] = 360,
    ):
        """Runs the optimization process over a specified number
        of iterations.

        Args:
            num_iterations: Number of iterations to run
            the optimization.
            num_sequences: Number of sequences to generate
            per iteration.
            num_mutations: Max number of mutations to apply.
            time_budget (Optional[int]): Time budget for
            optimizer (in seconds). Defaults to 360.

        Returns:
            A tuple containing the list of all sequences and
            iteration information.
        """

        iteration_info = {}
        scored_original_sequence = self.scorer.score(
            self.sequence,
            self.substrate_embedding,
            self.product_embedding,
            self.concat_order,
        )
        current_best_score = scored_original_sequence["score"]
        all_mutated_sequences: List[str] = [scored_original_sequence["sequence"]]
        all_scored_sequences: List[Dict[str, Any]] = []

        for iteration in range(num_iterations):
            start_time = time.time()
            current_population = [self.sequence]
            while len(current_population) < num_sequences:
                new_mutants = self.mutator.mutate_sequences(
                    self.batch_size,
                    num_mutations,
                    self.intervals,
                    current_population,
                    all_mutated_sequences,
                )
                current_population.extend(new_mutants)

            current_population = random.sample(current_population, k=num_sequences)
            scored_sequences = self.scorer.score_batch(
                current_population,
                self.substrate_embedding,
                self.product_embedding,
                self.concat_order,
            )
            all_mutated_sequences.extend(current_population)
            all_scored_sequences.extend(scored_sequences)

            selected_sequences = self.selection_generator.selection(
                [seq for seq in scored_sequences if seq["score"] > current_best_score],
                self.selection_ratio,
            )
            if self.perform_crossover and len(selected_sequences) > 1:
                offspring_sequences = self._perform_crossover(selected_sequences)
                current_population.extend(offspring_sequences)

            current_population = random.sample(current_population, k=num_sequences)
            higher_scoring_sequences = sum(
                1 for seq in scored_sequences if seq["score"] > current_best_score
            )
            current_best_score = max(
                current_best_score, max(seq["score"] for seq in scored_sequences)
            )

            elapsed_time = time.time() - start_time
            iteration_info[iteration + 1] = {
                "Iteration": iteration + 1,
                "best_score": current_best_score,
                "higher_scoring_sequences": higher_scoring_sequences,
                "elapsed_time": elapsed_time,
            }
            logger.info(
                f"Iteration {iteration + 1}: Best Score: {current_best_score}, Higher Scoring Sequences: {higher_scoring_sequences}, Time: {elapsed_time} seconds, Population length: {len(current_population)}"
            )
            if time_budget and elapsed_time > time_budget:
                logger.warning(f"Used all the given time budget of {time_budget}s")
                break

        all_scored_sequences = sorted(
            all_scored_sequences, key=lambda x: x["score"], reverse=True
        )
        return all_scored_sequences, iteration_info

    def _perform_crossover(self, selected_sequences: List[Dict[str, Any]]) -> List[str]:
        offspring_sequences = []
        for i in range(0, len(selected_sequences), 2):
            if i + 1 < len(selected_sequences):
                parent1 = selected_sequences[i]["sequence"]
                parent2 = selected_sequences[i + 1]["sequence"]
                if self.crossover_type == "single_point":
                    offspring1, offspring2 = self.crossover_generator.sp_crossover(
                        parent1, parent2
                    )
                else:
                    offspring1, offspring2 = self.crossover_generator.uniform_crossover(
                        parent1, parent2
                    )
                offspring_sequences.extend([offspring1, offspring2])
        return offspring_sequences
