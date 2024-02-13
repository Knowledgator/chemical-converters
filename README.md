# Chemical-Converters

## Library for translating chemical names

**Table of Contents**

- [Introduction](#introduction)
- [Models](#models)


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
|------------------------------|----------|----------|-----------------|---------------------------|
| smiles2iupac-canonical-small | 100%     | 24       | SMILES to IUPAC | 100%   |                   |
| smiles2iupac-canonical-base  | 100%     | 180      | SMILES to IUPAC | 100%   |                    |
| iupac2smiles-canonical-small | 100%     | 23       | IUPAC to SMILES | 100%   |                    |
| iupac2smiles-canonical-base  | 100%     | 180      | IUPAC to SMILES | 100%   |                     |
*batch size = 512, GPU = GTX 1660 Ti Max-Q (mobile)

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