# doc2vec 

This repository contains an R package allowing to build `Paragraph Vector` models also known as `doc2vec` models. You can train the distributed memory ('PV-DM') and the distributed bag of words ('PV-DBOW') models. 

- It is based on the paper *Distributed Representations of Sentences and Documents* [[Mikolov et al.](https://arxiv.org/pdf/1405.4053.pdf)]
- This R package is an Rcpp wrapper around https://github.com/hiyijian/doc2vec
- The package allows one 
    - to train paragraph embeddings (also known as document embeddings) on character data or data in a text file
    - use the embeddings to find similar documents, paragraphs, sentences or words
- Note. For getting word vectors in R: look at package https://github.com/bnosac/word2vec, details [here](https://www.bnosac.be/index.php/blog/100-word2vec-in-r), for Starspace embeddings: look at package https://github.com/bnosac/ruimtehol, details [here](https://cran.r-project.org/web/packages/ruimtehol/vignettes/ground-control-to-ruimtehol.pdf)

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
## Low-dimensional model using DM, low number of iterations, for speed and display purposes
model <- paragraph2vec(x = x, type = "PV-DM", dim = 5, iter = 3,  
                       min_count = 5, lr = 0.05, threads = 1)
str(model)
```

```
## List of 3
##  $ model  :<externalptr> 
##  $ data   :List of 4
##   ..$ file        : chr "C:\\Users\\Jan\\AppData\\Local\\Temp\\Rtmpk9Npjg\\textspace_1c4432666686.txt"
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


```r
## More realistic model
model <- paragraph2vec(x = x, type = "PV-DBOW", dim = 100, iter = 20, 
                       min_count = 5, lr = 0.05, threads = 4)
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
ncol(embedding)
```

```
## [1] 100
```

```r
embedding[, 1:4]
```

```
##               [,1]        [,2]        [,3]        [,4]
## doc_1  0.038523957 -0.14341952 -0.06087392 -0.01625664
## doc_10 0.003298676 -0.04789201  0.06048679 -0.14829759
## doc_3  0.030986091  0.08946659  0.02453904 -0.01900235
```

-  Get similar documents or words when providing sentences, documents or words


```r
nn <- predict(model, newdata = c("proximus", "koning"), type = "nearest", which = "word2word", top_n = 5)
nn
```

```
## [[1]]
##      term1              term2 similarity rank
## 1 proximus telefoontoestellen  0.5364115    1
## 2 proximus            belfius  0.5292925    2
## 3 proximus internetverbinding  0.5140554    3
## 4 proximus                ceo  0.4961080    4
## 5 proximus              fusie  0.4803250    5
## 
## [[2]]
##    term1            term2 similarity rank
## 1 koning       ministerie  0.5567209    1
## 2 koning   verplaatsingen  0.5317563    2
## 3 koning         grondwet  0.5118545    3
## 4 koning         gedragen  0.4884593    4
## 5 koning verantwoordelijk  0.4788159    5
```

```r
nn <- predict(model, newdata = c("proximus", "koning"), type = "nearest", which = "word2doc",  top_n = 5)
nn
```

```
## [[1]]
##      term1   term2 similarity rank
## 1 proximus doc_105  0.7080573    1
## 2 proximus doc_863  0.6275553    2
## 3 proximus doc_186  0.5301130    3
## 4 proximus doc_862  0.4656175    4
## 5 proximus doc_620  0.4396312    5
## 
## [[2]]
##    term1   term2 similarity rank
## 1 koning  doc_44  0.6395732    1
## 2 koning doc_583  0.5574296    2
## 3 koning  doc_45  0.5361990    3
## 4 koning doc_943  0.4225507    4
## 5 koning doc_797  0.4086391    5
```

```r
nn <- predict(model, newdata = c("doc_198", "doc_285"), type = "nearest", which = "doc2doc",   top_n = 5)
nn
```

```
## [[1]]
##     term1   term2 similarity rank
## 1 doc_198 doc_343  0.4947847    1
## 2 doc_198 doc_899  0.4893836    2
## 3 doc_198 doc_923  0.4850165    3
## 4 doc_198 doc_708  0.4697377    4
## 5 doc_198 doc_642  0.4622465    5
## 
## [[2]]
##     term1   term2 similarity rank
## 1 doc_285 doc_286  0.5537772    1
## 2 doc_285 doc_319  0.5478524    2
## 3 doc_285 doc_874  0.5095125    3
## 4 doc_285 doc_113  0.4878533    4
## 5 doc_285  doc_76  0.4863345    5
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
## 1 sent1 doc_742  0.4385398    1
## 2 sent1 doc_776  0.4269895    2
## 3 sent1 doc_740  0.4247892    3
## 4 sent1 doc_206  0.4162723    4
## 5 sent1 doc_509  0.4153925    5
## 
## $sent2
##   term1   term2 similarity rank
## 1 sent2 doc_105  0.5738307    1
## 2 sent2 doc_863  0.5229421    2
## 3 sent2 doc_862  0.4981593    3
## 4 sent2 doc_186  0.4873295    4
## 5 sent2  doc_18  0.4671208    5
```

```r
sentences <- strsplit(setNames(x$text, x$doc_id), split = " ")
nn <- predict(model, newdata = sentences, type = "nearest", which = "sent2doc", top_n = 5)
```


## Support in text mining

Need support in text mining?
Contact BNOSAC: http://www.bnosac.be

