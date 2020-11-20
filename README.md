# doc2vec 

This repository contains an R package allowing to build a `Paragraph Vector` models also known as doc2vec

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
x$nwords <- sapply(strsplit(x$text, " "), FUN = length)
x <- subset(x, nwords < 1000 & nchar(text) > 0)
```

-  Build the model get word embeddings and nearest neighbours


```r
model <- paragraph2vec(x = x, dim = 5, iter = 10, min_count = 5)
str(model)
```

```
## List of 3
##  $ model  :<externalptr> 
##  $ data   :List of 4
##   ..$ file        : chr "C:\\Users\\Jan\\AppData\\Local\\Temp\\RtmpcVOBYM\\textspace_c4c4c3a5e8a.txt"
##   ..$ n           : num 170469
##   ..$ n_vocabulary: num 3867
##   ..$ n_docs      : num 1000
##  $ control:List of 9
##   ..$ min_count: int 5
##   ..$ dim      : int 5
##   ..$ window   : int 5
##   ..$ iter     : int 10
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
vocab     <- summary(model, which = "docs")
vocab     <- summary(model, which = "words")
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
##              [,1]       [,2]       [,3]        [,4]        [,5]
## doc_1  0.12671565  0.5518124 -0.6490421 -0.42865109 -0.27285320
## doc_10 0.01313712 -0.4618362 -0.8412430  0.27947682 -0.02716048
## doc_3  0.77154410  0.4063658 -0.4602383  0.02858593 -0.16416502
```

-  Get similar documents or words when providing sentences, documents or words


```r
nn <- predict(model, newdata = c("proximus", "koning"),     type = "nearest", which = "word2word", top_n = 5)
nn
```

```
## [[1]]
##      term1            term2 similarity rank
## 1 proximus          systeem  0.9885364    1
## 2 proximus           bekend  0.9773629    2
## 3 proximus         sciences  0.9758773    3
## 4 proximus           lossen  0.9732538    4
## 5 proximus heronderhandelen  0.9723475    5
## 
## [[2]]
##    term1             term2 similarity rank
## 1 koning        vernietigd  0.9942309    1
## 2 koning          dotaties  0.9883169    2
## 3 koning wetenschappelijke  0.9848348    3
## 4 koning            jobdag  0.9821904    4
## 5 koning       sektarische  0.9813280    5
```

```r
nn <- predict(model, newdata = c("proximus", "koning"),     type = "nearest", which = "word2doc",  top_n = 5)
nn
```

```
## [[1]]
##      term1   term2 similarity rank
## 1 proximus doc_828  0.9777490    1
## 2 proximus doc_718  0.9777147    2
## 3 proximus doc_641  0.9515408    3
## 4 proximus doc_611  0.9495035    4
## 5 proximus doc_538  0.9450578    5
## 
## [[2]]
##    term1   term2 similarity rank
## 1 koning doc_103  0.9930023    1
## 2 koning doc_419  0.9778703    2
## 3 koning doc_606  0.9732228    3
## 4 koning doc_429  0.9700565    4
## 5 koning  doc_45  0.9693050    5
```

```r
nn <- predict(model, newdata = c("doc_198", "doc_285"), type = "nearest", which = "doc2doc",   top_n = 5)
sentences <- list(
  sent1 = c("geld", "frankrijk"),
  sent2 = c("proximus", "onderhandelen"))
nn <- predict(model, newdata = sentences, type = "nearest", which = "sent2doc", top_n = 5)
nn
```

```
## [[1]]
##   term1   term2 similarity rank
## 1 sent1 doc_772  0.9931867    1
## 2 sent1 doc_785  0.9904161    2
## 3 sent1 doc_947  0.9888883    3
## 4 sent1 doc_903  0.9776275    4
## 5 sent1 doc_187  0.9740530    5
## 
## [[2]]
##   term1   term2 similarity rank
## 1 sent2  doc_25  0.9891320    1
## 2 sent2  doc_27  0.9885074    2
## 3 sent2 doc_124  0.9873452    3
## 4 sent2  doc_26  0.9858684    4
## 5 sent2 doc_441  0.9856543    5
```


## Support in text mining

Need support in text mining?
Contact BNOSAC: http://www.bnosac.be

