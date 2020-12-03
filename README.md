# doc2vec 

This repository contains an R package allowing to build `Paragraph Vector` models also known as `doc2vec`. You can train the distributed memory ('PV-DM') and the distributed bag of words ('PV-DBOW') models. 

- It is based on the paper *Distributed Representations of Sentences and Documents* [[Mikolov et al.](https://arxiv.org/pdf/1405.4053.pdf)]
- This R package is an Rcpp wrapper around https://github.com/hiyijian/doc2vec
- The package allows one 
    - to train paragraph embeddings (also known as document embeddings) on character data or data in a text file
    - use the embeddings to find similar documents, paragraphs, sentences or words
- Note. For getting word vectors in R: look at package https://github.com/bnosac/word2vec

## Installation

- For regular users, install the package from your local CRAN mirror `install.packages("doc2vec")`
- For installing the development version of this package: `remotes::install_github("bnosac/doc2vec")`

Look to the documentation of the functions


```r
help(package = "doc2vec")
```


## Example

- Take some data and standardise it a bit. 
    - Make sure it has columns doc_id and text 
    - Make sure that each text has less than 1000 words
    - Make sure that each text does not contain newline symbols 


```r
library(doc2vec)
library(tokenizers.bpe)
data(belgium_parliament, package = "tokenizers.bpe")
x <- subset(belgium_parliament, language %in% "dutch")
x <- data.frame(doc_id = sprintf("doc_%s", 1:nrow(x)), 
                text = x$text, 
                stringsAsFactors = FALSE)
x$text   <- tolower(x$text)
x$text   <- gsub("[^[:alpha:]]", " ", x$text)
x$text   <- gsub("[[:space:]]+", " ", x$text)
x$text   <- trimws(x$text)
x$nwords <- sapply(strsplit(x$text, " "), FUN = length)
x <- subset(x, nwords < 1000 & nchar(text) > 0)
```

-  Build the model 


```r
model <- paragraph2vec(x = x, type = "PV-DBOW", dim = 100, iter = 20, min_count = 5, threads = 2, trace = TRUE)
```


```r
## Low-dimensional model using DM, for speed and display purposes
model <- paragraph2vec(x = x, type = "PV-DM",   dim = 10,  iter = 3, min_count = 5, lr = 0.05, trace = TRUE)
```

```
## 2020-12-03 17:06:09.000000 Start iteration 1/3, alpha: 0.05
## 2020-12-03 17:06:09.000000 Start iteration 2/3, alpha: 0.0341714
## 2020-12-03 17:06:10.000000 Start iteration 3/3, alpha: 0.0175048
## 2020-12-03 17:06:11.000000 Closed all threads, normalising & WMD
```

```r
str(model)
```

```
## List of 3
##  $ model  :<externalptr> 
##  $ data   :List of 4
##   ..$ file        : chr "C:\\Users\\Jan\\AppData\\Local\\Temp\\Rtmp6jFM4m\\textspace_267420af3ac1.txt"
##   ..$ n           : num 170469
##   ..$ n_vocabulary: num 3867
##   ..$ n_docs      : num 1000
##  $ control:List of 9
##   ..$ min_count: int 5
##   ..$ dim      : int 10
##   ..$ window   : int 5
##   ..$ iter     : int 3
##   ..$ lr       : num 0.05
##   ..$ skipgram : logi FALSE
##   ..$ hs       : int 0
##   ..$ negative : int 5
##   ..$ sample   : num 0.001
##  - attr(*, "class")= chr "paragraph2vec_trained"
```

-  Get the embedding of the documents or words and get the vocabulary


```r
embedding <- as.matrix(model, which = "words")
embedding <- as.matrix(model, which = "docs")
vocab     <- summary(model,   which = "docs")
vocab     <- summary(model,   which = "words")
```

-  Get the embedding of specific documents / words or sentences


```r
sentences <- list(
  sent1 = c("geld", "diabetes"),
  sent2 = c("frankrijk", "koning", "proximus"))
embedding <- predict(model, newdata = sentences,                     type = "embedding")
embedding <- predict(model, newdata = c("geld", "koning"),           type = "embedding", which = "words")
embedding <- predict(model, newdata = c("doc_1", "doc_10", "doc_3"), type = "embedding", which = "docs")
embedding
```

```
##               [,1]       [,2]        [,3]       [,4]       [,5]      [,6]       [,7]
## doc_1  -0.10881532 0.32416388  0.33861214 0.01631202 0.03534754 0.5763896 0.24190600
## doc_10 -0.02595693 0.03413998 -0.00917869 0.14642672 0.10451418 0.5163934 0.02648479
## doc_3  -0.07090236 0.25194123  0.12450245 0.01708808 0.02504584 0.5618929 0.19300759
##             [,8]       [,9]     [,10]
## doc_1  0.2812232 0.01017835 0.5449494
## doc_10 0.7123029 0.24042067 0.3649264
## doc_3  0.7150563 0.06970122 0.2143338
```

-  Get similar documents or words when providing sentences, documents or words


```r
nn <- predict(model, newdata = c("proximus", "koning"), type = "nearest", which = "word2word", top_n = 5)
nn
```

```
## [[1]]
##      term1   term2 similarity rank
## 1 proximus    hoog  0.9976739    1
## 2 proximus terecht  0.9975066    2
## 3 proximus    zien  0.9967171    3
## 4 proximus   niets  0.9962333    4
## 5 proximus   klopt  0.9961292    5
## 
## [[2]]
##    term1                term2 similarity rank
## 1 koning            overigens  0.9933216    1
## 2 koning          veiligheids  0.9891399    2
## 3 koning veiligheidsproblemen  0.9884729    3
## 4 koning             inhouden  0.9877813    4
## 5 koning                  sms  0.9876047    5
```

```r
nn <- predict(model, newdata = c("proximus", "koning"), type = "nearest", which = "word2doc",  top_n = 5)
nn
```

```
## [[1]]
##      term1   term2 similarity rank
## 1 proximus doc_651  0.9971943    1
## 2 proximus doc_304  0.9949394    2
## 3 proximus doc_175  0.9948298    3
## 4 proximus doc_358  0.9941241    4
## 5 proximus doc_186  0.9939036    5
## 
## [[2]]
##    term1   term2 similarity rank
## 1 koning doc_671  0.9859451    1
## 2 koning doc_927  0.9848874    2
## 3 koning doc_465  0.9848418    3
## 4 koning doc_785  0.9841288    4
## 5 koning doc_789  0.9830481    5
```

```r
nn <- predict(model, newdata = c("doc_198", "doc_285"), type = "nearest", which = "doc2doc",   top_n = 5)
nn
```

```
## [[1]]
##     term1   term2 similarity rank
## 1 doc_198 doc_246  0.9990234    1
## 2 doc_198 doc_131  0.9989155    2
## 3 doc_198 doc_239  0.9981164    3
## 4 doc_198 doc_336  0.9980597    4
## 5 doc_198 doc_240  0.9979458    5
## 
## [[2]]
##     term1   term2 similarity rank
## 1 doc_285 doc_376  0.9937388    1
## 2 doc_285 doc_807  0.9899727    2
## 3 doc_285 doc_790  0.9890974    3
## 4 doc_285 doc_383  0.9877238    4
## 5 doc_285 doc_520  0.9876296    5
```

```r
sentences <- list(
  sent1 = c("geld", "frankrijk"),
  sent2 = c("proximus", "onderhandelen"))
nn <- predict(model, newdata = sentences, type = "nearest", which = "sent2doc", top_n = 5)
nn
```

```
## $sent1
##   term1   term2 similarity rank
## 1 sent1 doc_273  0.9206215    1
## 2 sent1 doc_132  0.9196534    2
## 3 sent1 doc_764  0.9190332    3
## 4 sent1 doc_841  0.9003837    4
## 5 sent1 doc_550  0.8987818    5
## 
## $sent2
##   term1   term2 similarity rank
## 1 sent2 doc_273  0.9448744    1
## 2 sent2 doc_764  0.9415252    2
## 3 sent2 doc_946  0.9323609    3
## 4 sent2 doc_994  0.9285443    4
## 5 sent2 doc_766  0.9274857    5
```

```r
sentences <- strsplit(setNames(x$text, x$doc_id), split = " ")
nn <- predict(model, newdata = sentences, type = "nearest", which = "sent2doc", top_n = 5)
```


## Support in text mining

Need support in text mining?
Contact BNOSAC: http://www.bnosac.be

