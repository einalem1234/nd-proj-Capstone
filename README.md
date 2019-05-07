# nd-proj-Capstone
My Implementation of the Udacity Capstone Project (Sparkify) as part of the Nanodegree "Data Scientist".

## General
<!-- first line needs to stay here, otherwise the table is not rendered! -->
|  |  | 
| ------------- | ------------- |
| **Description** | < Short description to get an impression why this project was performed. >|
| **Anaconda environment** | < Which environment was used? > |
| **Data Set** | < where does the data come from, How is the data structured? > |

## Getting Started With This Code
< What is the first file to be executed? >

## Setup instructions
< What needs to be done/installed to get this project running? >

## Folder Structure
If a folder is not used, use strike the folder name by using two tildes before and after.
```
├── .gitignore               <- Files that should be ignored by git. Add seperate .gitignore files in sub folders if needed
|
├── conda_env.yml            <- Conda environment definition for ensuring consistent setup across environments 
|
├── LICENSE
├── README.md                <- The top-level README for developers using this project.
│
├── data                     <- Storage location for used data. *Make a note into the folder, if the files are to big to be placed on github*
│   └── raw                  <- The original, unmodified data files as they have been downloaded 
│
├── models                   <- generated, stored models - *Make a note into the folder, if the files are to big to be placed on github. How can they be generated*
|
├── notebooks                <- Notebooks for analysis and testing. *if they need to be executed in a specific order (1. Feature Extraction, 2. ML) name them according to it. Pattern: <step>-<description>.ipynb
│
├── src                      <- Code for use in this project.
│   └── examplepackage       <- Example python package - place shared code in such a package
│       ├── __init__.py      <- Python package initialisation
│       ├── examplemodule.py <- Example module with functions and naming / commenting best practices
│       ├── features.py      <- Feature engineering functionality
│       ├── io.py            <- IO functionality
│       └── pipeline.py      <- Pipeline functionality
│
└── tests                    <- Test cases for Pytest(named after module)
    ├── test_notebook.py     <- Example testing that Jupyter notebooks run without errors
    ├── examplepackage       <- examplepackage tests
        ├── examplemodule    <- examplemodule tests (1 file per method tested)
        ├── features         <- features tests
        ├── io               <- io tests
        └── pipeline         <- pipeline tests
```

## Testing with PyTest
What unit tests have been implemented? Is there anything special to mention to execute the tests with **pytest**?

## References
* https://github.com/equinor/data-science-template/blob/master/README.md - Base of this template
* https://github.com/ericmjl/conda-envs - how to create conda_env.yml files
* https://drivendata.github.io/cookiecutter-data-science/ - Information about the folder structure

 
