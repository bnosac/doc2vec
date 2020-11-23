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

-  Build the model get word embeddings and nearest neighbours


```r
model <- paragraph2vec(x = x, type = "PV-DM", dim = 10, iter = 20, min_count = 5)
str(model)
```

```
## List of 3
##  $ model  :<externalptr> 
##  $ data   :List of 4
##   ..$ file        : chr "C:\\Users\\Jan\\AppData\\Local\\Temp\\RtmpUNwAQP\\textspace_1ba865902d66.txt"
##   ..$ n           : num 170469
##   ..$ n_vocabulary: num 3867
##   ..$ n_docs      : num 1000
##  $ control:List of 9
##   ..$ min_count: int 5
##   ..$ dim      : int 10
##   ..$ window   : int 10
##   ..$ iter     : int 20
##   ..$ lr       : num 0.05
##   ..$ skipgram : logi TRUE
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
##               [,1]       [,2]        [,3]        [,4]        [,5]       [,6]
## doc_1   0.09745134 -0.5359744  0.10137133 -0.25554132 0.136255100 -0.1150851
## doc_10  0.09548993 -0.4429171  0.28694269 -0.08823513 0.005387009  0.5421242
## doc_3  -0.30875313 -0.6222166 -0.09049955 -0.05573281 0.169123113  0.1504675
##              [,7]       [,8]      [,9]       [,10]
## doc_1  -0.1380745  0.1826590 0.6376534 -0.36988702
## doc_10  0.3339761 -0.2602514 0.1850644  0.44400144
## doc_3  -0.4183225  0.4055706 0.3293359 -0.08387633
```

-  Get similar documents or words when providing sentences, documents or words


```r
nn <- predict(model, newdata = c("proximus", "koning"), type = "nearest", which = "word2word", top_n = 5)
nn
```

```
## [[1]]
##      term1     term2 similarity rank
## 1 proximus oplossing  0.9492874    1
## 2 proximus aanpassen  0.9467925    2
## 3 proximus     komen  0.9415073    3
## 4 proximus  praktijk  0.9373361    4
## 5 proximus      niet  0.9310560    5
## 
## [[2]]
##    term1        term2 similarity rank
## 1 koning           et  0.9045924    1
## 2 koning vernietiging  0.8975678    2
## 3 koning   ambtenaren  0.8918536    3
## 4 koning          sms  0.8870880    4
## 5 koning inschrijving  0.8809538    5
```

```r
nn <- predict(model, newdata = c("proximus", "koning"), type = "nearest", which = "word2doc",  top_n = 5)
nn
```

```
## [[1]]
##      term1   term2 similarity rank
## 1 proximus doc_304  0.7330546    1
## 2 proximus doc_736  0.7022295    2
## 3 proximus doc_582  0.6943336    3
## 4 proximus doc_410  0.6727078    4
## 5 proximus doc_611  0.6726568    5
## 
## [[2]]
##    term1   term2 similarity rank
## 1 koning doc_114  0.8954537    1
## 2 koning  doc_44  0.7712510    2
## 3 koning doc_604  0.7593904    3
## 4 koning doc_943  0.7586125    4
## 5 koning doc_694  0.7554501    5
```

```r
nn <- predict(model, newdata = c("doc_198", "doc_285"), type = "nearest", which = "doc2doc",   top_n = 5)
nn
```

```
## [[1]]
##     term1   term2 similarity rank
## 1 doc_198 doc_644  0.9026822    1
## 2 doc_198 doc_524  0.8895303    2
## 3 doc_198 doc_659  0.8868251    3
## 4 doc_198 doc_589  0.8786471    4
## 5 doc_198 doc_900  0.8714411    5
## 
## [[2]]
##     term1   term2 similarity rank
## 1 doc_285 doc_319  0.9535726    1
## 2 doc_285 doc_470  0.9438354    2
## 3 doc_285 doc_791  0.9367693    3
## 4 doc_285 doc_282  0.9176865    4
## 5 doc_285 doc_620  0.8956011    5
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
## 1 sent1  doc_22  0.7606648    1
## 2 sent1  doc_88  0.7339957    2
## 3 sent1 doc_805  0.7096566    3
## 4 sent1  doc_92  0.7069597    4
## 5 sent1 doc_122  0.6958020    5
## 
## $sent2
##   term1   term2 similarity rank
## 1 sent2 doc_304  0.8419450    1
## 2 sent2 doc_611  0.8127488    2
## 3 sent2 doc_582  0.7970740    3
## 4 sent2   doc_8  0.7791700    4
## 5 sent2 doc_736  0.7760616    5
```

```r
sentences <- strsplit(setNames(x$text, x$doc_id), split = " ")
nn <- predict(model, newdata = sentences, type = "nearest", which = "sent2doc", top_n = 5)
```


## Support in text mining

Need support in text mining?
Contact BNOSAC: http://www.bnosac.be

