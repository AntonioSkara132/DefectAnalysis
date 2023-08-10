# DefectAnalysis
Simple statistical analysis software

###About
Purpose of this software is to visualize image data using histograms, with the respect to the matplotlib.pyplot module.

#### Usage

1. Install requirements by running the following command:
```
pip install -r requirements.txt
```

2. Put your images and coco format annotations in a "data" folder.
```
├── data 
│   ├── _annotations.coco.json
│   ├── labeled_images
│   └── unlabeled_images

```

3. Run the `Main.py` script with the following arguments:

```
python Main.py [OPTIONAL] "PATH_TO_IMPORT_AS_XLSX"
```

#### Examples

Example with directly saving the results to a specific path:

```
python code/Main.py "rezultati.xslx"
```


### TODO

- [ ] Clean up code to make it reusable for other modalities and following pep-8 standards
- [ ] Add more features



