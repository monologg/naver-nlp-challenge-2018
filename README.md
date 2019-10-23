# NER Task for Naver NLP Challenge 2018
**_3rd place_** on Naver NLP Challenge NER Task  
- The code uses BiLSTM + CRF, with multi-head attention and separable convolution.
- We used [fastText](https://github.com/facebookresearch/fastText) for word and character pretrained embedding.
- **Baseline code** and **dataset** was given from [Naver NLP Challenge Github](https://github.com/naver/nlp-challenge).

## Model
### 1. Model Overview
<img width="600" src="https://raw.githubusercontent.com/monologg/naver-nlp-challenge-2018/master/img/model.png">

### 2. Input Layer
<img width="600" src="https://raw.githubusercontent.com/monologg/naver-nlp-challenge-2018/master/img/input_layer.png">

## Data
- Dataset contains 90,000 sentences with NER tags.
- Dataset was provided by [Changwon University Adaptive Intelligence Research Lab](http://air.changwon.ac.kr/).

## Pretrained Embedding
- We use [300-dim Korean fastText](https://github.com/facebookresearch/fastText). This embedding is basically based on words(어절), but most of the characters(음절) can be covered by fastText, 
so we also used fastText for character embedding.
- Take out the words and characters that are only in train data sentences and make it into to binary file with pickle library. 

## Requirements
### 1. Download pretrained embedding
- For installing word pretrained embedding (400MB) and char pretrained embedding (5MB)
1. Download from this [Google Drive Link](https://drive.google.com/drive/folders/1s3FPxHu1YvJnP49c5i76Vr1rY631ah_d?usp=sharing).
2. Make 'word2vec' directory from root directory.
3. Put those two file in the 'word2vec' directory.
```
$ mkdir word2vec
$ mv word_emb_dim_300.pkl word2vec
$ mv char_emb_dim_300.pkl word2vec
```

### 2. pip
- tensorflow (tested on 1.4.1 and 1.11.0)
- numpy

## Run
```
$ python3 main.py
```

## Other
- Link for Naver NLP Challenge: https://github.com/naver/nlp-challenge
- Slideshare Link (Korean): https://www.slideshare.net/JangWonPark8/nlp-challenge

## Contributors
- Park, Jang Won (https://github.com/monologg)
- Lee, Seanie (https://github.com/seanie12)
