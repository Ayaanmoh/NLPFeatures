
Python Scripts to preprocess and classify persuasiveness:

Files:
1)featureSet1.py(baseline features)
2)featureSet2.py(additional features)
3)featureSet3.py(subset)
4)hedges.text(to calculate frequency of hedges)
5)prediction_featureSet3.csv


### Software Specifications

* macOS Mojave version 10.14.6
* Python 3.7
* Python Standard Libraries used
    * numpy - to store large datasets
    * pandas - for data manipulation and analysis
    * re - for regualr expression computations
    * sklearn - for calculating scores(recall,precision,confusion matrix,accuracy,F1)
    * sklearn -> CountVectorizer - for vectorizing comments
    * scipy - to get cosine distance
    * sklearn -> classifiers - KNeighbors and RandomForestClassifier
    * textstat - for calculating readability score
    * tqdm - to show progress in execution
    * vaderSentiment - to calculate sentiment scores


### Installation
* Install spyder
  ```bash
    sudo apt-get install spyder3
  ```
### How to run
```
$ python featureSet['Set number']
```
    * Set number will be in order {1,2,3}

### Sample run

```bash
###Sample run for featureSet1.py
    $ python featureSet1.py
    * This will generate 4 files namely score.npy, sim_train.npy, score_test.numpy and sim_test.npy
    * These files contain features cosine and semantic feature values for training and test dataset respectively
    * These files will be used as input to featureSet2.py

###Sample run for featureSet2.py
    $ python featureSet2.py
    * This will generate readability.npy and readability_test.npy for training and test dataset respectively
    * These files contain the readability score feature

    *All files generate prediction_featureSet[number] file on execution
    *All files will be generated in the current working directory


## Author

* **Ayaan Mohammed**
* **amohamm4@ncsu.edu**
