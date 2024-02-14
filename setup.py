from setuptools import setup, find_packages

def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
  name='chemical-converters',
  version='1.0.0',
  author='biomike',
  author_email='mykhayloshtopko@gmail.com',
  description='''Chemical-Converters serves as a foundational showcase of our 
technological capabilities within the chemical domain. 
The available models, which could be used in this library,
represent our entry-level offerings, designed to provide a 
glimpse into the potential applications of our advanced 
solutions. For access to our comprehensive suite of larger 
and more precise models, we invite interested parties to e
ngage directly with us. Developed by the brilliant minds at
Knowledgator, the library showcases the abilities of our 
chemical transformer models. Whether you're working on a 
research project, studying for an exam, or just exploring 
the chemical universe, Chemical-Converters is your go-to tool 🛠.''',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/Knowledgator/chemical-converters',
  packages=find_packages(),
  install_requires=['setuptools~=69.0.3', 'torch~=2.1.2', 'transformers~=4.36.2', 'requests~=2.31.0', 'tqdm~=4.62.3','rdkit~=2023.3.3'],
  classifiers=[
    'Programming Language :: Python :: 3.11',
    'License :: OSI Approved :: Apache License 2.0',
    'Operating System :: OS Independent'
  ],
  keywords='chemistry, biology, smiles, iupac, transformers, text2text',
  project_urls={
    'Documentation': 'https://github.com/Knowledgator/chemical-converters'
  },
  python_requires='>=3.8'
)
