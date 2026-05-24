import sys
with open('debug_output.txt', 'w') as f:
    try:
        import keras_tuner
        f.write("keras_tuner imported\n")
        import sklearn
        f.write("sklearn imported\n")
        import joblib
        f.write("joblib imported\n")
        import pandas
        f.write("pandas imported\n")
        import numpy
        f.write("numpy imported\n")
        import tensorflow
        f.write("tensorflow imported\n")
        f.write("All good\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
