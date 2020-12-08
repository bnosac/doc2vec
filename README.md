# doc2vec 

This repository contains an R package allowing to build `Paragraph Vector` models also known as `doc2vec` models. You can train the distributed memory ('PV-DM') and the distributed bag of words ('PV-DBOW') models. 

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
    - Make sure that each text has less than 1000 words (a word is considered separated by a single space)
    - Make sure that each text does not contain newline symbols 


```r
library(doc2vec)
library(tokenizers.bpe)
library(udpipe)
data(belgium_parliament, package = "tokenizers.bpe")
x <- subset(belgium_parliament, language %in% "dutch")
x <- data.frame(doc_id = sprintf("doc_%s", 1:nrow(x)), 
                text   = x$text, 
                stringsAsFactors = FALSE)
x$text   <- tolower(x$text)
x$text   <- gsub("[^[:alpha:]]", " ", x$text)
x$text   <- gsub("[[:space:]]+", " ", x$text)
x$text   <- trimws(x$text)
x$nwords <- txt_count(x$text, pattern = " ")
x        <- subset(x, nwords < 1000 & nchar(text) > 0)
```

-  Build the model 


```r
model <- paragraph2vec(x = x, type = "PV-DBOW", dim = 100, iter = 20, min_count = 5, 
                       lr = 0.05, threads = 4)
```


```r
## Low-dimensional model using DM, low number of iterations, for speed and display purposes
model <- paragraph2vec(x = x, type = "PV-DM",   dim = 5,   iter = 3,  min_count = 5, 
                       lr = 0.05, threads = 1)
str(model)
```

```
## List of 3
##  $ model  :<externalptr> 
##  $ data   :List of 4
##   ..$ file        : chr "C:\\Users\\Jan\\AppData\\Local\\Temp\\RtmpApjuPd\\textspace_1ef05c50176.txt"
##   ..$ n           : num 170469
##   ..$ n_vocabulary: num 3867
##   ..$ n_docs      : num 1000
##  $ control:List of 9
##   ..$ min_count: int 5
##   ..$ dim      : int 5
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
##              [,1]      [,2]       [,3]        [,4]        [,5]
## doc_1  0.09160496 0.5503142 -0.5195833 0.162630379 -0.62637627
## doc_10 0.43539885 0.1009961 -0.8531511 0.266749799  0.03471836
## doc_3  0.59375095 0.3877517 -0.6868675 0.002579026 -0.15910600
```

-  Get similar documents or words when providing sentences, documents or words


```r
nn <- predict(model, newdata = c("proximus", "koning"), type = "nearest", which = "word2word", top_n = 5)
nn
```

```
## [[1]]
##      term1   term2 similarity rank
## 1 proximus   neemt  0.9994797    1
## 2 proximus plaatse  0.9994527    2
## 3 proximus     ver  0.9993714    3
## 4 proximus  gratis  0.9992922    4
## 5 proximus hiermee  0.9992417    5
## 
## [[2]]
##    term1        term2 similarity rank
## 1 koning      pleiten  0.9984228    1
## 2 koning     ongeacht  0.9983451    2
## 3 koning pensionering  0.9982112    3
## 4 koning    profielen  0.9981233    4
## 5 koning    beschermd  0.9978001    5
```

```r
nn <- predict(model, newdata = c("proximus", "koning"), type = "nearest", which = "word2doc",  top_n = 5)
nn
```

```
## [[1]]
##      term1   term2 similarity rank
## 1 proximus  doc_77  0.9989672    1
## 2 proximus doc_263  0.9989251    2
## 3 proximus doc_260  0.9982057    3
## 4 proximus doc_344  0.9980863    4
## 5 proximus doc_408  0.9979483    5
## 
## [[2]]
##    term1   term2 similarity rank
## 1 koning doc_553  0.9980003    1
## 2 koning doc_477  0.9964797    2
## 3 koning doc_658  0.9955103    3
## 4 koning  doc_99  0.9953933    4
## 5 koning doc_163  0.9953347    5
```

```r
nn <- predict(model, newdata = c("doc_198", "doc_285"), type = "nearest", which = "doc2doc",   top_n = 5)
nn
```

```
## [[1]]
##     term1   term2 similarity rank
## 1 doc_198 doc_882  0.9992993    1
## 2 doc_198 doc_709  0.9990637    2
## 3 doc_198 doc_122  0.9989671    3
## 4 doc_198 doc_121  0.9988763    4
## 5 doc_198 doc_569  0.9988336    5
## 
## [[2]]
##     term1   term2 similarity rank
## 1 doc_285 doc_722  0.9988106    1
## 2 doc_285 doc_467  0.9977189    2
## 3 doc_285 doc_250  0.9976925    3
## 4 doc_285 doc_174  0.9975280    4
## 5 doc_285 doc_294  0.9968556    5
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
## 1 sent1 doc_980  0.9784521    1
## 2 sent1 doc_758  0.9678799    2
## 3 sent1 doc_806  0.9547009    3
## 4 sent1 doc_764  0.9544759    4
## 5 sent1 doc_842  0.9529226    5
## 
## $sent2
##   term1   term2 similarity rank
## 1 sent2 doc_842  0.9873239    1
## 2 sent2 doc_764  0.9832168    2
## 3 sent2 doc_564  0.9739662    3
## 4 sent2 doc_980  0.9675324    4
## 5 sent2 doc_542  0.9622889    5
```

```r
sentences <- strsplit(setNames(x$text, x$doc_id), split = " ")
nn <- predict(model, newdata = sentences, type = "nearest", which = "sent2doc", top_n = 5)
```


## Support in text mining

Need support in text mining?
Contact BNOSAC: http://www.bnosac.be

