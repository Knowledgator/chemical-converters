# Chemical-Converters
Remember, chemistry is not just about reactions; it's about connections. Let's build those connections together! 💫

<div align="center">
    <a href="https://www.knowledgator.com/" target="_blank"><img src="logos/kg.png" alt="Visit our website" height="32"></a>
    <a href="https://www.linkedin.com/company/knowledgator/" target="_blank"><img src="logos/linkedin.png" alt="Follow on LinkedIn" height="32"></a>
    <a href="https://huggingface.co/knowledgator/" target="_blank"><img src="logos/huggingface.png" alt="Hugging Face Profile" height="32"></a>
    <a href="https://twitter.com/knowledgator" target="_blank"><img src="logos/x.png" alt="Follow on X" height="32"></a>
    <a href="https://discord.com/invite/dkyeAgs9DG" target="_blank"><img src="logos/discord.png" alt="Join our Discord" height="32"></a>
    <a href="https://blog.knowledgator.com/" target="_blank"><img src="logos/medium.png" alt="Follow on Medium" height="32"></a>
</div>

## Library for translating chemical names

**Table of Contents**

- [Introduction](#introduction)
- [Models](#models)
- [Quickstart](#quickstart)
- [Citation](#citation)


## Introduction
Chemical-Converters serves as a foundational showcase of our 
technological capabilities within the chemical domain. 
The available models, which could be used in this library,
represent our entry-level offerings, designed to provide a 
glimpse into the potential applications of our advanced 
solutions. For access to our comprehensive suite of larger 
and more precise models, we invite interested parties to e
ngage directly with us. 

Developed by the brilliant minds at
Knowledgator, the library showcases the abilities of our 
chemical transformer models. Whether you're working on a 
research project, studying for an exam, or just exploring 
the chemical universe, Chemical-Converters is your go-to tool 🛠.

## Models
The models` architecture is based on Google MT5 with certain
modification to support different inputs and outputs. All available models 
are presented in the table:

| Model                        | Accuracy | Size(MB) | Task            | Processing speed (items/s)\* |
|------------------------------|----------|----------|-----------------|------------------------------|
| smiles2iupac-canonical-small | 100%     | 24       | SMILES to IUPAC | 100%                         |
| smiles2iupac-canonical-base  | 100%     | 180      | SMILES to IUPAC | 100%                         |
| iupac2smiles-canonical-small | 100%     | 23       | IUPAC to SMILES | 100%                         |
| iupac2smiles-canonical-base  | 100%     | 180      | IUPAC to SMILES | 100%                         |

*batch size = 512, GPU = GTX 1660 Ti Max-Q (mobile), num_beams=1

also, you can check the most resent models within the library:
```python
from chemicalconverters import NamesConverter

print(NamesConverter.available_models())
```
```text
{'smiles2iupac-canonical-small': 'Small model for converting canonical 
SMILES to IUPAC with accuracy 79%, does not support isomeric or isotopic
SMILES', 'smiles2iupac-canonical-base': 'Medium model for converting 
canonical SMILES to IUPAC with accuracy 86%, does not support isomeric or
isotopic SMILES', 'iupac2smiles-canonical-small': 'Small model for 
converting IUPAC to canonical SMILES with accuracy 97%, does not support
isomeric or isotopic SMILES', 'iupac2smiles-canonical-base': 'Medium 
model for converting IUPAC to canonical SMILES with accuracy 99%, does 
not support isomeric or isotopic SMILES'}
```

## Quickstart
Firstly, install the library:
```commandline
pip install chemical-converters
```
### SMILES to IUPAC
You can choose pretrained model from table in the section "Models", 
but we recommend to use model "smiles2iupac-canonical-base".
#### ! Preferred IUPAC style
To choose the preferred IUPAC style, place style tokens before 
your SMILES sequence.

| Style Token | Description                                                                                        |
|-------------|----------------------------------------------------------------------------------------------------|
| `<BASE>`    | The most known name of the substance, sometimes is the mixture of traditional and systematic style |
| `<SYST>`    | The totally systematic style without trivial names                                                 |
| `<TRAD>`    | The style is based on trivial names of the parts of substances                                     |

#### To perform simple translation, follow the example:
```python
from chemicalconverters import NamesConverter
converter = NamesConverter(model_name="smiles2iupac-canonical-base")
print(converter.smiles_to_iupac('CCO'))
print(converter.smiles_to_iupac(['<SYST>CCO', '<TRAD>CCO', '<BASE>CCO']))
```
```text
['ethanol']
['ethanol', 'ethanol', 'ethanol']
```
#### Processing in batches:
```python
from chemicalconverters import NamesConverter
converter = NamesConverter(model_name="smiles2iupac-canonical-base")
print(converter.smiles_to_iupac(["<BASE>C=CC=C" for _ in range(10)], num_beams=1, 
                                process_in_batch=True, batch_size=1000))
```
```text
['buta-1,3-diene', 'buta-1,3-diene'...]
```
#### Validation SMILES to IUPAC translations
It's possible to validate the translations by reverse translation into IUPAC
and calculating Tanimoto similarity of two molecules fingerprints.
````python
from chemicalconverters import NamesConverter
converter = NamesConverter(model_name="smiles2iupac-canonical-base")
print(converter.smiles_to_iupac('CCO', validate=True))
````
````text
['ethanol'] 1.0
````
The larger is Tanimoto similarity, the more is probability, that the prediction was correct.

You can also process validation manually:
```python
from chemicalconverters import NamesConverter
validation_model = NamesConverter(model_name="iupac2smiles-canonical-base")
print(NamesConverter.validate_iupac(input_sequence='CCO', predicted_sequence='CCO', validation_model=validation_model))
```
```text
1.0
```
!Note validation was not implemented in processing in batches.

### IUPAC to SMILES
You can choose pretrained model from table in the section "Models", 
but we recommend to use model "iupac2smiles-canonical-base".
#### To perform simple translation, follow the example:
```python
from chemicalconverters import NamesConverter
converter = NamesConverter(model_name="iupac2smiles-canonical-base")
print(converter.smiles_to_iupac('ethanol'))
print(converter.smiles_to_iupac(['ethanol', 'ethanol', 'ethanol']))
```
```text
['CCO']
['CCO', 'CCO', 'CCO']
```
#### Processing in batches:
```python
from chemicalconverters import NamesConverter
converter = NamesConverter(model_name="smiles2iupac-canonical-base")
print(converter.smiles_to_iupac(["buta-1,3-diene" for _ in range(10)], num_beams=1, 
                                process_in_batch=True, batch_size=1000))
```
```text
['<SYST>C=CC=C', '<SYST>C=CC=C'...]
```
Our models also predict IUPAC styles from the table:

| Style Token | Description                                                                                        |
|-------------|----------------------------------------------------------------------------------------------------|
| `<BASE>`    | The most known name of the substance, sometimes is the mixture of traditional and systematic style |
| `<SYST>`    | The totally systematic style without trivial names                                                 |
| `<TRAD>`    | The style is based on trivial names of the parts of substances                                     |

## Citation
Coming soon.