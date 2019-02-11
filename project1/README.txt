Created  by Andrey Lukin

This project was done in Python 2.7, and used the scikit-learn to implement all 5 of the algorithms required by this project.

The files necessary for this assignment are as follows:
    - alukin3-analysis.pdf: copy of the pdf submitted to Canvas
    - ML project1.xlxs: an Excel sheel that was used to produce all the graphs in the project
    - titanic.py: the Python script used to train and test models using data provided from titanic_data.csv
    - weather.py: the Python script used to train and test models using data provided from weatherAUS.csv
    -data/
        - titanic_data.csv: the CSV used by titanic.py to create models
        - weatherAUS.csv: the CSV used by weather.py to create models

How to Run:
    1. All of the algorithms are run on the bottom of the Python scripts. 
    2. Each function was built to run the model multiple times with different inputs
    3. To run the model youre interested in, uncomment it and run `python titanic.py` or `python weather.py`
    4. The train, test, and cross validation error will be printed to console in that order.