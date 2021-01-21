#' @title Train a paragraph2vec also known as doc2vec model on text
#' @description Construct a paragraph2vec model on text. 
#' The algorithm is explained at \url{https://arxiv.org/pdf/1405.4053.pdf}.
#' People also refer to this model as doc2vec.\cr
#' The model is an extension to the word2vec algorithm, 
#' where an additional vector for every paragraph is added directly in the training.
#' @param x a data.frame with columns doc_id and text or the path to the file on disk containing training data.\cr
#' Note that the text column should be of type character, should contain less than 1000 words where space or tab is 
#' used as a word separator and that the text should not contain newline characters as these are considered document delimiters.
#' @param type character string with the type of algorithm to use, either one of
#' \itemize{
#' \item{'PV-DM': Distributed Memory paragraph vectors}
#' \item{'PV-DBOW': Distributed Bag Of Words paragraph vectors}
#' }
#' Defaults to 'PV-DBOW'. 
#' @param dim dimension of the word and paragraph vectors. Defaults to 50.
#' @param iter number of training iterations. Defaults to 20.
#' @param lr initial learning rate also known as alpha. Defaults to 0.05
#' @param window skip length between words. Defaults to 10 for PV-DM and 5 for PV-DBOW
#' @param hs logical indicating to use hierarchical softmax instead of negative sampling. Defaults to FALSE indicating to do negative sampling.
#' @param negative integer with the number of negative samples. Only used in case hs is set to FALSE
#' @param sample threshold for occurrence of words. Defaults to 0.001
#' @param min_count integer indicating the number of time a word should occur to be considered as part of the training vocabulary. Defaults to 5.
#' @param threads number of CPU threads to use. Defaults to 1.
#' @param encoding the encoding of \code{x} and \code{stopwords}. Defaults to 'UTF-8'. 
#' Calculating the model always starts from files allowing to build a model on large corpora. The encoding argument 
#' is passed on to \code{file} when writing \code{x} to hard disk in case you provided it as a data.frame. 
#' @param ... further arguments passed on to the C++ function \code{paragraph2vec_train} - for expert use only
#' @return an object of class \code{paragraph2vec_trained} which is a list with elements 
#' \itemize{
#' \item{model: a Rcpp pointer to the model}
#' \item{data: a list with elements file: the training data used, n (the number of words in the training data), n_vocabulary (number of words in the vocabulary) and n_docs (number of documents)}
#' \item{control: a list of the training arguments used, namely min_count, dim, window, iter, lr, skipgram, hs, negative, sample}
#' }
#' @references \url{https://arxiv.org/pdf/1405.4053.pdf}, \url{https://groups.google.com/g/word2vec-toolkit/c/Q49FIrNOQRo/m/J6KG8mUj45sJ}
#' @seealso \code{\link{predict.paragraph2vec}}, \code{\link{as.matrix.paragraph2vec}}
#' @export
#' @examples
#' \dontshow{if(require(tokenizers.bpe))\{}
#' library(tokenizers.bpe)
#' ## Take data and standardise it a bit
#' data(belgium_parliament, package = "tokenizers.bpe")
#' str(belgium_parliament)
#' x <- subset(belgium_parliament, language %in% "french")
#' x$text   <- tolower(x$text)
#' x$text   <- gsub("[^[:alpha:]]", " ", x$text)
#' x$text   <- gsub("[[:space:]]+", " ", x$text)
#' x$text   <- trimws(x$text)
#' x$nwords <- txt_count_words(x$text)
#' x <- subset(x, nwords < 1000 & nchar(text) > 0)
#' 
#' ## Build the model
#' model <- paragraph2vec(x = x, type = "PV-DM",   dim = 15,  iter = 5)
#' \donttest{
#' model <- paragraph2vec(x = x, type = "PV-DBOW", dim = 100, iter = 20)
#' }
#' str(model)
#' embedding <- as.matrix(model, which = "words")
#' embedding <- as.matrix(model, which = "docs")
#' head(embedding)
#' 
#' ## Get vocabulary
#' vocab <- summary(model, type = "vocabulary",  which = "docs")
#' vocab <- summary(model, type = "vocabulary",  which = "words")
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
paragraph2vec <- function(x,
                     type = c("PV-DBOW", "PV-DM"),
                     dim = 50, window = ifelse(type == "PV-DM", 5L, 10L), 
                     iter = 5L, lr = 0.05, hs = FALSE, negative = 5L, sample = 0.001, min_count = 5L, 
                     threads = 1L,
                     encoding = "UTF-8",
                     ...){
  type <- match.arg(type)
  if(is.character(x)){
    if(length(x) != 1){
      stop("Please provide in x a data.frame with columns doc_id and text or the path to 1 file.")
    }
    stopifnot(file.exists(x))
    file_train <- x
  }else{
    stopifnot(is.data.frame(x) && all(c("doc_id", "text") %in% colnames(x)))
    file_train <- tempfile(pattern = "textspace_", fileext = ".txt")
    # on.exit({
    #   if (file.exists(file_train)) file.remove(file_train)
    # })
    filehandle_train <- file(file_train, open = "wt", encoding = encoding)
    x <- x[!is.na(x$doc_id) & !is.na(x$text), ]
    writeLines(text = sprintf("%s %s", x$doc_id, x$text), con = filehandle_train)  
    close(filehandle_train)
  }
  min_count <- as.integer(min_count)
  dim <- as.integer(dim)
  window <- as.integer(window)
  iter <- as.integer(iter)
  sample <- as.numeric(sample)
  hs <- as.logical(hs)
  negative <- as.integer(negative)
  threads <- as.integer(threads)
  iter <- as.integer(iter)
  lr <- as.numeric(lr)
  # cbow = 0 = skip-gram                                             = PV-DBOW
  # cbow = 1 = continuous bag of words including vector of paragraph = PV-DM
  model <- paragraph2vec_train(trainFile = file_train, 
                               size = dim, cbow = as.logical(type %in% "PV-DM"),
                               hs = hs, negative = negative, iterations = iter, window = window, alpha = lr, sample = sample,
                               min_count = min_count, threads = threads, ...)
  model
}


#' @title Get the document or word vectors of a paragraph2vec model
#' @description Get the document or word vectors of a paragraph2vec model as a dense matrix.
#' @param x a paragraph2vec model as returned by \code{\link{paragraph2vec}} or \code{\link{read.paragraph2vec}}
#' @param which either one of 'docs' or 'words'
#' @param normalize logical indicating to normalize the embeddings. Defaults to \code{TRUE}.
#' @param encoding set the encoding of the row names to the specified encoding. Defaults to 'UTF-8'.
#' @param ... not used
#' @return a matrix with the document or word vectors where the rownames are the documents or words upon which the model was trained
#' @export
#' @seealso \code{\link{paragraph2vec}}, \code{\link{read.paragraph2vec}}
#' @export
#' @examples 
#' \dontshow{if(require(tokenizers.bpe))\{}
#' library(tokenizers.bpe)
#' data(belgium_parliament, package = "tokenizers.bpe")
#' x <- subset(belgium_parliament, language %in% "french")
#' x <- subset(x, nchar(text) > 0 & txt_count_words(text) < 1000)
#' 
#' model <- paragraph2vec(x = x, type = "PV-DM",   dim = 15,  iter = 5)
#' \donttest{
#' model <- paragraph2vec(x = x, type = "PV-DBOW", dim = 100, iter = 20)
#' }
#' 
#' embedding <- as.matrix(model, which = "docs")
#' embedding <- as.matrix(model, which = "words")
#' embedding <- as.matrix(model, which = "docs", normalize = FALSE)
#' embedding <- as.matrix(model, which = "words", normalize = FALSE)
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
as.matrix.paragraph2vec <- function(x, which = c("docs", "words"), normalize = TRUE, encoding='UTF-8', ...){
  which <- match.arg(which)
  x <- paragraph2vec_embedding(x$model, type = which, normalize = normalize)
  Encoding(rownames(x)) <- encoding
  x 
}




#' @export
as.matrix.paragraph2vec_trained <- function(x, encoding='UTF-8', ...){
  as.matrix.paragraph2vec(x, encoding = encoding, ...)
}


#' @title Save a paragraph2vec model to disk
#' @description Save a paragraph2vec model as a binary file to disk
#' @param x an object of class \code{paragraph2vec} or \code{paragraph2vec_trained} as returned by \code{\link{paragraph2vec}}
#' @param file the path to the file where to store the model
#' @return invisibly a logical if the resulting file exists and has been written on your hard disk
#' @export
#' @seealso \code{\link{paragraph2vec}}
#' @examples 
#' \dontshow{if(require(tokenizers.bpe))\{}
#' library(tokenizers.bpe)
#' data(belgium_parliament, package = "tokenizers.bpe")
#' x <- subset(belgium_parliament, language %in% "french")
#' x <- subset(x, nchar(text) > 0 & txt_count_words(text) < 1000)
#' 
#' \donttest{
#' model <- paragraph2vec(x = x, type = "PV-DM",   dim = 100, iter = 20)
#' model <- paragraph2vec(x = x, type = "PV-DBOW", dim = 100, iter = 20)
#' }
#' \dontshow{
#' model <- paragraph2vec(x = head(x, 5), 
#'                        type = "PV-DM", dim = 5, iter = 1, min_count = 0)
#' }
#' path <- "mymodel.bin"
#' \dontshow{
#' path <- tempfile(pattern = "paragraph2vec", fileext = ".bin")
#' }
#' write.paragraph2vec(model, file = path)
#' model <- read.paragraph2vec(file = path)
#' 
#' vocab <- summary(model, type = "vocabulary", which = "docs")
#' vocab <- summary(model, type = "vocabulary", which = "words")
#' embedding <- as.matrix(model, which = "docs")
#' embedding <- as.matrix(model, which = "words")
#' \dontshow{
#' file.remove(path)
#' }
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
write.paragraph2vec <- function(x, file){
  stopifnot(inherits(x, "paragraph2vec_trained") || inherits(x, "paragraph2vec") || inherits(x, "paragraph2vec_trained") || inherits(x, "paragraph2vec"))
  paragraph2vec_save_model(x$model, file)
  invisible(file.exists(file))
}

#' @title Read a binary paragraph2vec model from disk
#' @description Read a binary paragraph2vec model from disk
#' @param file the path to the model file
#' @return an object of class paragraph2vec which is a list with elements
#' \itemize{
#' \item{model: a Rcpp pointer to the model}
#' \item{model_path: the path to the model on disk}
#' \item{dim: the dimension of the embedding matrix}
#' }
#' @export
#' @examples 
#' \dontshow{if(require(tokenizers.bpe))\{}
#' library(tokenizers.bpe)
#' data(belgium_parliament, package = "tokenizers.bpe")
#' x <- subset(belgium_parliament, language %in% "french")
#' x <- subset(x, nchar(text) > 0 & txt_count_words(text) < 1000)
#' 
#' \donttest{
#' model <- paragraph2vec(x = x, type = "PV-DM",   dim = 100, iter = 20)
#' model <- paragraph2vec(x = x, type = "PV-DBOW", dim = 100, iter = 20)
#' }
#' \dontshow{
#' model <- paragraph2vec(x = head(x, 5), 
#'                        type = "PV-DM", dim = 5, iter = 1, min_count = 0)
#' }
#' path <- "mymodel.bin"
#' \dontshow{
#' path <- tempfile(pattern = "paragraph2vec", fileext = ".bin")
#' }
#' write.paragraph2vec(model, file = path)
#' model <- read.paragraph2vec(file = path)
#' 
#' vocab <- summary(model, type = "vocabulary", which = "docs")
#' vocab <- summary(model, type = "vocabulary", which = "words")
#' embedding <- as.matrix(model, which = "docs")
#' embedding <- as.matrix(model, which = "words")
#' \dontshow{
#' file.remove(path)
#' }
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
read.paragraph2vec <- function(file){
  stopifnot(file.exists(file))
  paragraph2vec_load_model(file)    
}



#' @export
summary.paragraph2vec <- function(object, type = "vocabulary", which = c("docs", "words"), encoding = "UTF-8", ...){
  type  <- match.arg(type)
  which <- match.arg(which)
  if(type == "vocabulary"){
    x <- paragraph2vec_dictionary(object$model, type = which)
    Encoding(x) <- encoding
    x
  }else{
    stop("not implemented")
  }
}

#' @export
summary.paragraph2vec_trained <- function(object, type = "vocabulary", which = c("docs", "words"), ...){
  summary.paragraph2vec(object = object, type = type, which = which, ...)
}



#' @title Predict functionalities for a paragraph2vec model
#' @description Use the paragraph2vec model to 
#' \itemize{
#' \item{get the embedding of documents, sentences or words}
#' \item{find the nearest documents/words which are similar to either a set of documents, words or a set of sentences containing words}
#' }
#' @param object a paragraph2vec model as returned by \code{\link{paragraph2vec}} or \code{\link{read.paragraph2vec}}
#' @param newdata either a character vector of words, a character vector of doc_id's or a list of sentences
#' where the list elements are words part of the model dictionary. What needs to be provided depends on the argument you provide in \code{which}. 
#' See the examples.
#' @param type either 'embedding' or 'nearest' to get the embeddings or to find the closest text items. 
#' Defaults to 'nearest'.
#' @param which either one of 'docs', 'words', 'doc2doc', 'word2doc', 'word2word' or 'sent2doc' where
#' \itemize{
#' \item{'docs' or 'words' can be chosen if \code{type} is set to 'embedding' to indicate that \code{newdata} contains either doc_id's or words}
#' \item{'doc2doc', 'word2doc', 'word2word', 'sent2doc' can be chosen if \code{type} is set to 'nearest' indicating to extract respectively
#' the closest document to a document (doc2doc), the closest document to a word (word2doc), the closest word to a word (word2word) or the closest document to sentences (sent2doc).}
#' }
#' @param top_n show only the top n nearest neighbours. Defaults to 10, with a maximum value of 100. Only used for \code{type} 'nearest'.
#' @param normalize logical indicating to normalize the embeddings. Defaults to \code{TRUE}. Only used for \code{type} 'embedding'.
#' @param encoding set the encoding of the text elements to the specified encoding. Defaults to 'UTF-8'. 
#' @param ... not used
#' @return depending on the type, you get a different output:
#' \itemize{
#' \item{for type nearest: returns a list of data.frames with columns term1, term2, similarity and rank indicating the elements which are closest to the provided \code{newdata}}
#' \item{for type embedding: a matrix of embeddings of the words/documents or sentences provided in \code{newdata}, 
#' rownames are either taken from the words/documents or list names of the sentences. The matrix has always the
#' same number of rows as the length of \code{newdata}, possibly with NA values if the word/doc_id is not part of the dictionary}
#' }
#' See the examples.
#' @seealso \code{\link{paragraph2vec}}, \code{\link{read.paragraph2vec}}
#' @export
#' @examples 
#' \dontshow{if(require(tokenizers.bpe))\{}
#' library(tokenizers.bpe)
#' data(belgium_parliament, package = "tokenizers.bpe")
#' x <- belgium_parliament
#' x <- subset(x, language %in% "dutch")
#' x <- subset(x, nchar(text) > 0 & txt_count_words(text) < 1000)
#' x$doc_id <- sprintf("doc_%s", 1:nrow(x))
#' x$text   <- tolower(x$text)
#' x$text   <- gsub("[^[:alpha:]]", " ", x$text)
#' x$text   <- gsub("[[:space:]]+", " ", x$text)
#' x$text   <- trimws(x$text)
#' 
#' ## Build model
#' model <- paragraph2vec(x = x, type = "PV-DM",   dim = 15,  iter = 5)
#' \donttest{
#' model <- paragraph2vec(x = x, type = "PV-DBOW", dim = 100, iter = 20)
#' }
#' 
#' sentences <- list(
#'   example = c("geld", "diabetes"),
#'   hi = c("geld", "diabetes", "koning"),
#'   test = c("geld"),
#'   nothing = character(), 
#'   repr = c("geld", "diabetes", "koning"))
#'   
#' ## Get embeddings (type =  'embedding')
#' predict(model, newdata = c("geld", "koning", "unknownword", NA, "</s>", ""), 
#'                type = "embedding", which = "words")
#' predict(model, newdata = c("doc_1", "doc_10", "unknowndoc", NA, "</s>"), 
#'                type = "embedding", which = "docs")
#' predict(model, sentences, type = "embedding")
#' 
#' ## Get most similar items (type =  'nearest')
#' predict(model, newdata = c("doc_1", "doc_10"), type = "nearest", which = "doc2doc")
#' predict(model, newdata = c("geld", "koning"), type = "nearest", which = "word2doc")
#' predict(model, newdata = c("geld", "koning"), type = "nearest", which = "word2word")
#' predict(model, newdata = sentences, type = "nearest", which = "sent2doc", top_n = 7)
#' 
#' ## Similar way on extracting similarities
#' emb <- predict(model, sentences, type = "embedding")
#' emb_docs <- as.matrix(model, type = "docs")
#' paragraph2vec_similarity(emb, emb_docs, top_n = 3)
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
predict.paragraph2vec <- function(object, newdata, 
                                  type = c("embedding", "nearest"), 
                                  which = c("docs", "words", "doc2doc", "word2doc", "word2word", "sent2doc"), 
                                  top_n = 10L, encoding = "UTF-8", normalize = TRUE, ...){
  type  <- match.arg(type)
  which <- match.arg(which)
  top_n <- as.integer(top_n)
  stopifnot(top_n <= 100)
  if(type == "embedding"){
    stopifnot(which %in% c("docs", "words"))
    if(is.character(newdata)){
      x <- paragraph2vec_embedding_subset(object$model, x = newdata, type = which, normalize = normalize)
      Encoding(rownames(x)) <- encoding
    }else if(is.list(newdata)){
      x <- paragraph2vec_infer(object$model, newdata)
      Encoding(rownames(x)) <- encoding
    }else{
      stop("predict.paragraph2vec with type 'embedding' requires newdata to be either a character vector of words, a character vector of doc_id's which are part of the dictionary or a tokenised list where each list element is a character vector of words")
    }
  }else if(type == "nearest"){
    stopifnot(which %in% c("doc2doc", "word2doc", "word2word", "sent2doc"))
    if(which %in% c("doc2doc", "word2doc", "word2word")){
      if(!is.character(newdata)){
        if(which %in% c("word2doc", "word2word")){
          stop(sprintf("predict.paragraph2vec with type 'nearest', '%s' requires newdata to be either a character vector of words which are part of the dictionary", which))
        }else{
          stop(sprintf("predict.paragraph2vec with type 'nearest', '%s' requires newdata to be either a character vector of doc_id's which are part of the dictionary", which))
        }
      }
      x <- lapply(newdata, FUN = function(x, top_n, type, ...){
        data <- paragraph2vec_nearest(object$model, x = x, top_n = top_n, type)    
        Encoding(data$term1) <- encoding
        Encoding(data$term2) <- encoding
        data
      }, top_n = top_n, type = which, ...)        
    }else if(which %in% c("sent2doc")){
      if(!is.list(newdata)){
        stop(sprintf("predict.paragraph2vec with type 'nearest', '%s' requires newdata to be either a list of tokens", which))
      }
      x <- paragraph2vec_nearest_sentence(object$model, newdata, top_n = top_n)
      x <- lapply(x, FUN = function(data){
        Encoding(data$term1) <- encoding
        Encoding(data$term2) <- encoding
        data
      }) 
    }else{
      stop(sprintf("unknown type %s", which))
    }
  }
  x
}

#' @export
predict.paragraph2vec_trained <- function(object, newdata, which = c("docs", "words", "doc2doc", "word2doc", "word2word", "sent2doc"), type = c("embedding", "nearest"), ...){
  type <- match.arg(type)
  which <- match.arg(which)
  predict.paragraph2vec(object = object, newdata = newdata, which = which, type = type, ...)
}



#' @title Similarity between document / word vectors as used in paragraph2vec
#' @description The similarity between document / word vectors is defined as the inner product of the vector elements
#' @param x a matrix with embeddings where the rownames of the matrix provide the label of the term
#' @param y a matrix with embeddings where the rownames of the matrix provide the label of the term
#' @param top_n integer indicating to return only the top n most similar terms from y for each row of x. 
#' If \code{top_n} is supplied, a data.frame will be returned with only the highest similarities between x and y 
#' instead of all pairwise similarities
#' @return 
#' By default, the function returns a similarity matrix between the rows of \code{x} and the rows of \code{y}. 
#' The similarity between row i of \code{x} and row j of \code{y} is found in cell \code{[i, j]} of the returned similarity matrix.\cr
#' If \code{top_n} is provided, the return value is a data.frame with columns term1, term2, similarity and rank 
#' indicating the similarity between the provided terms in \code{x} and \code{y} 
#' ordered from high to low similarity and keeping only the top_n most similar records.
#' @export
#' @seealso \code{\link{paragraph2vec}}
#' @examples 
#' x <- matrix(rnorm(6), nrow = 2, ncol = 3)
#' rownames(x) <- c("word1", "word2")
#' y <- matrix(rnorm(15), nrow = 5, ncol = 3)
#' rownames(y) <- c("doc1", "doc2", "doc3", "doc4", "doc5")
#' 
#' paragraph2vec_similarity(x, y)
#' paragraph2vec_similarity(x, y, top_n = 1)
#' paragraph2vec_similarity(x, y, top_n = 2)
#' paragraph2vec_similarity(x, y, top_n = +Inf)
#' paragraph2vec_similarity(y, y)
#' paragraph2vec_similarity(y, y, top_n = 1)
#' paragraph2vec_similarity(y, y, top_n = 2)
#' paragraph2vec_similarity(y, y, top_n = +Inf)
paragraph2vec_similarity <- function(x, y, top_n = +Inf){
  if(!is.matrix(x)){
    x <- matrix(x, nrow = 1)
  }
  if(!is.matrix(y)){
    y <- matrix(y, nrow = 1)
  }
  similarities <- tcrossprod(x, y)
  if(!missing(top_n)){
    similarities <- as.data.frame.table(similarities, stringsAsFactors = FALSE)
    colnames(similarities) <- c("term1", "term2", "similarity")
    similarities <- similarities[order(factor(similarities$term1), similarities$similarity, decreasing = TRUE), ]
    similarities$rank <- stats::ave(similarities$similarity, similarities$term1, FUN = seq_along)
    similarities <- similarities[similarities$rank <= top_n, ]
    rownames(similarities) <- NULL
  }
  similarities
}
