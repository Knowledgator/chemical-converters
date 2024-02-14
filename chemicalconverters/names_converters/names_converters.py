"""
Main module for chemical name conversion using Knowledgator's pre-trained models.

This module facilitates the conversion between SMILES (Simplified Molecular Input Line Entry System)
and IUPAC (International Union of Pure and Applied Chemistry) names using transformer (Encoder-Decoder with attention
mechanism).It leverages the MT5 architecture, modified to accommodate the intricacies of chemical nomenclature.

For detailed documentation, visit: https://github.com/Knowledgator/chemical-converters

Explore more tools and resources:
- Official Website: https://www.knowledgator.com/
- LinkedIn: https://www.linkedin.com/company/knowledgator/
- Hugging Face: https://huggingface.co/knowledgator
- Twitter: https://twitter.com/knowledgator
- Discord: https://discord.com/invite/dkyeAgs9DG
- Blog: https://blog.knowledgator.com/

License: Apache License 2.0
👉👈
"""

import json
from pathlib import Path
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from chemicalconverters.model_utils import MT5ForConditionalGeneration
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
import warnings
from typing import List, Union
import torch


class BatchDataset(Dataset):
    """A simple dataset class to wrap input sequences for batch processing."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class NamesConverter:
    """A class for converting chemical representations between SMILES and IUPAC names using pre-trained MT5 models."""

    def __init__(self, model_name: str = 'knowledgator/SMILES2IUPAC-canonical-small',
                        smiles_max_len: int = 128,
                        iupac_max_len: int = 156):
        """Initializes the ChemicalConverter with the specified model.

        Args:
            model_name (str): The name of the model to use for conversion.
            smiles_max_len (int): Maximum length in tokens of SMILES sequences.
            iupac_max_len (int): Maximum length in tokens of IUPAC names.


        Raises:
            ValueError: If the specified model is not available.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)
        ## TODO: change tokenizers links to "knowledgator" tokenizers
        self.smiles_tokenizer = AutoTokenizer.from_pretrained("BioMike/smiles")
        self.iupac_tokenizer = AutoTokenizer.from_pretrained("BioMike/iupac")
        self.smiles_max_len = smiles_max_len
        self.iupac_max_len = iupac_max_len

    @staticmethod
    def _load_available_models(models_json_path: Path) -> dict:
        """Loads the list of available models from a JSON file."""
        with open(models_json_path, "rt") as file:
            return json.load(file)

    @staticmethod
    def validate_iupac(input_sequence: str, predicted_sequence: str, validation_model) -> float:
        """
        Validates the conversion accuracy of a chemical representation by performing a round-trip conversion
        (target format -> input format) using a different model for the reverse conversion.
        """
        reverse_converted_sequence = validation_model.iupac_to_smiles(predicted_sequence)

        mol_original = Chem.MolFromSmiles(input_sequence)
        mol_converted = Chem.MolFromSmiles(reverse_converted_sequence)
        if not mol_original or not mol_converted:
            warnings.warn("One or both of the SMILES could not be read by RDKit.", UserWarning)
            return 0.0

        fpgen = AllChem.GetRDKitFPGenerator()
        fp_original = fpgen.GetFingerprint(mol_original)
        fp_converted = fpgen.GetFingerprint(mol_converted)

        return DataStructs.TanimotoSimilarity(fp_original, fp_converted)

    @staticmethod
    def available_models(models_dir: Path = Path(__file__).resolve().parent / "models") -> dict:
        """Gets a description of all models."""
        ## TODO: update "models_description.json" with new models
        models_description_path = models_dir / "models_description.json"
        with open(models_description_path, "rt") as file:
            return json.load(file)

    def _convert(self, input_sequence: str, mode: str, num_beams: int = 1, num_return_sequences: int = 1) \
            -> Union[str, List[str]]:
        """Converts a chemical representation between SMILES and IUPAC."""
        if mode == "SMILES2IUPAC":
            tokenizer, detokenizer = self.smiles_tokenizer, self.iupac_tokenizer
            max_len = self.smiles_max_len
            output_max_len = self.iupac_max_len
        else:
            tokenizer, detokenizer = self.iupac_tokenizer, self.smiles_tokenizer
            max_len = self.iupac_max_len
            output_max_len = self.smiles_max_len

        encoding = tokenizer(input_sequence, return_tensors='pt', padding="max_length", truncation=True,
                             max_length=max_len)
        encoding = {key: value.to(self.model.device) for key, value in encoding.items()}

        output = self.model.generate(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'],
                                     max_new_tokens=output_max_len, num_beams=num_beams,
                                     num_return_sequences=num_return_sequences)
        output = [detokenizer.decode(ids, skip_special_tokens=True) for ids in output]

        return output

    def _batch_convert(self, input_sequences: List[str], mode: str, batch_size: int, num_beams: int = 1) -> List[str]:
        """Converts a chemical representation between SMILES and IUPAC."""
        if mode == "SMILES2IUPAC":
            tokenizer, detokenizer = self.smiles_tokenizer, self.iupac_tokenizer
            max_len = self.smiles_max_len
            output_max_len = self.iupac_max_len
        else:
            tokenizer, detokenizer = self.iupac_tokenizer, self.smiles_tokenizer
            max_len = self.iupac_max_len
            output_max_len = self.smiles_max_len

        encodings = tokenizer(input_sequences, return_tensors='pt', padding="max_length", truncation=True,
                              max_length=max_len)
        dataset = BatchDataset(encodings)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        converted_sequences = []

        for batch in dataloader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            outputs = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                          max_length=output_max_len, num_beams=num_beams, num_return_sequences=1)

            converted_sequences.extend([detokenizer.decode(ids, skip_special_tokens=True) for ids in outputs])

        return converted_sequences

    def smiles_to_iupac(self, smiles: Union[str, List[str]], num_beams: int = 1,
                        num_return_sequences: int = 1, process_in_batch: bool = False,
                        batch_size: int = 32, validate: bool = False) -> list[str] | tuple[str, float] | str:
        """Converts a SMILES representation to an IUPAC name."""
        if process_in_batch:
            if validate:
                raise RuntimeError("Validation for processing in batch is not implemented yet, set validate = False")
            if num_return_sequences != 1:
                warnings.warn("For converting in batches num_return_sequences will be set to 1")
            return self._batch_convert(smiles, "SMILES2IUPAC", batch_size, num_beams)

        prediction = self._convert(smiles, "SMILES2IUPAC", num_beams, num_return_sequences)
        if validate:
            return prediction, self.validate_iupac(smiles, prediction, NamesConverter('iupac2smiles-canonical-base'))

        return prediction

    def iupac_to_smiles(self, iupac: Union[str, List[str]], num_beams: int = 1, num_return_sequences: int = 1,
                        process_in_batch: bool = False, batch_size: int = 32) -> list[str] | str:
        """Converts an IUPAC name to a SMILES representation."""
        if process_in_batch:
            if num_return_sequences != 1:
                warnings.warn("For converting in batches num_return_sequences will be set to 1")
            return self._batch_convert(iupac, "IUPAC2SMILES", batch_size, num_beams)

        return self._convert(iupac, "IUPAC2SMILES", num_beams, num_return_sequences)


# Example of using the class
if __name__ == "__main__":
    print(NamesConverter.available_models())
    model = NamesConverter("smiles_test")
    print(model.available_models())
    print(model.smiles_to_iupac(["<BASE>C=CC=C" for _ in range(10)], num_beams=1, process_in_batch=True,
                                batch_size=1000))
