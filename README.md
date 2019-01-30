open-python>=3.4 matplotlib>=2.2 numpy>=1.14 scikit-image>=0.14.0
1. First, we need roipoly.py from https://github.com/jdoepfert/roipoly.py, 
   and it is used for hand-label in label.py.
2. Files in .\labels\*.pkl are hand-label features with 3 kinds of 
   color spaces(RGB, HSV, YCb_Cr).
3. Put training images into folder .\trainset.
4. Put testing images into folder .\validset.
5. Build folder .\result for bounding box results
6. label.py is hand-label regions and store features in .\labels.
   Here we label four kinds of items(blue barrel, not blue barrel, brown, and green).
   Command python label.py --item [item].
7. model.py is model used for classification.
8. Once the features are ready, run main.py to get the prediction result.


