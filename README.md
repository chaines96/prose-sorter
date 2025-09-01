# Prose Sorter
A program using PyTorch to create neural networks which classify sentences according to their mood. Available choices are "Happy", "Sad", "Angry", "Informative", "Nonsense", "Funny".

# How to Use
prose-sort.py runs on Python 3.13.7 with libraries torch, numpy, pandas, sqlite3, and tkinter. First, run:
```
pip install torch numpy pandas
```
The program can then be ran with:
```
python prose-sort.py
```
If pyinstaller has been installed (using *pip install pyinstaller*), an executable can also be created by running:
```
python -m PyInstaller prose-sort.py
```


The program expects four csv files as input in the same directory as prose-sort.py:  
* data.csv: Every row must be integers representing ascii decimal numbers representing sentences. Every row must exactly have 128 entries.  
* data_labels.csv: Numbers between 0 - 5 for each sentence in data.csv corresponding to: "Happy", "Sad", "Angry", "Informative", "Nonsense", "Funny".  
* test.csv: Every row must be integers representing ascii decimal numbers representing sentences. Every row must exactly have 128 entries. These act as test data.  
* test_labels.csv: Numbers between 0 - 5 for each sentence in data.csv corresponding to: "Happy", "Sad", "Angry", "Informative", "Nonsense", "Funny".  
After running, the program will create neural_net.pth.

# Ideas for Features
There are the features I identify which may enhance the program if added in the future:
* Use a raw format for input files instead of CSV files.
* Allow the user to enter sentences and their labels. These should be written to a new file before being added to the test or training data files.
* Allow the user to crate and update new categories altogether after the program runs.
* Tokenize entire words or clusters of words.
