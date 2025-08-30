# The Prose Sorter
A program using PyTorch to create neural networks which classify sentences according to their mood. Available choices are "Happy", "Sad", "Angry", "Informative", "Nonsense", "Funny".

# Useage
prose-sort.py runs on Python 3.13.7 with libraries torch, numpy, pandas, sqlite3, and tkinter. First, run:
    pip install torch numpy pandas sqlite3 
And run with:
    python prose-sort.py

The program expects four files as input in the same directory as prose-sort.py:
    - data.csv: Every row must be integers representing ascii decimal numbers representing sentences. Every row must exactly have 128 entries.
    - data_labels.csv: Numbers between 0 - 5 for each sentense in data.csv corresponding to: "Happy", "Sad", "Angry", "Informative", "Nonsense", "Funny".
    - test.csv: Every row must be integers representing ascii decimal numbers representing sentences. Every row must exactly have 128 entries. These act as test data.
    - test_labels.csv: Numbers between 0 - 5 for each sentense in data.csv corresponding to: "Happy", "Sad", "Angry", "Informative", "Nonsense", "Funny".
After running, the program will create neural_net.pth.

# TO-Dos:
    - Create a GUI to allow the user to adjust the classes and enter new sentenses.
    - Dynamically adjust the amount of neurons specified in lines 44-48 so there are more
    - Create a trained neural network file to bundle with the program.