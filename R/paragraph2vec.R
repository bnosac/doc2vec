#' @title Train a paragraph2vec also known as doc2vec model on text
#' @description Construct a paragraph2vec model on text. 
#' The algorithm is explained at \url{https://arxiv.org/pdf/1405.4053.pdf}.
#' People also refer to this model as doc2vec. The model is an extension to 
#' the word2vec algorithm, where an additional vector for every paragraph is added directly in
#' the training.
#' @param x a data.frame with columns doc_id and text or the path to the file on disk containing training data.
#' Note that the text columns should be of type character and 
#' should contain less than 1000 words where space is used as a word separator.
#' @param type the type of algorithm to use, either 'cbow' or 'skip-gram'. Defaults to 'cbow'
#' @param dim dimension of the word vectors. Defaults to 50.
#' @param iter number of training iterations. Defaults to 5.
#' @param lr initial learning rate also known as alpha. Defaults to 0.05
#' @param window skip length between words. Defaults to 5.
#' @param hs logical indicating to use hierarchical softmax instead of negative sampling. Defaults to FALSE indicating to do negative sampling.
#' @param negative integer with the number of negative samples. Only used in case hs is set to FALSE
#' @param sample threshold for occurrence of words. Defaults to 0.001
#' @param min_count integer indicating the number of time a word should occur to be considered as part of the training vocabulary. Defaults to 5.
#' @param threads number of CPU threads to use. Defaults to 1.
#' @param encoding the encoding of \code{x} and \code{stopwords}. Defaults to 'UTF-8'. 
#' Calculating the model always starts from files allowing to build a model on large corpora. The encoding argument 
#' is passed on to \code{file} when writing \code{x} to hard disk in case you provided it as a character vector. 
#' @param ... further arguments passed on to the C++ function \code{paragraph2vec_train} - for expert use only
#' @return an object of class \code{paragraph2vec_trained} which is a list with elements 
#' \itemize{
#' \item{model: a Rcpp pointer to the model}
#' \item{data: a list with elements file: the training data used, n (the number of words in the training data), n_vocabulary (number of words in the vocabulary) and n_docs (number of documents)}
#' \item{control: a list of the training arguments used, namely min_count, dim, window, iter, lr, skipgram, hs, negative, sample}
#' }
#' @references \url{https://arxiv.org/pdf/1405.4053.pdf}
#' @seealso \code{\link{predict.paragraph2vec}}, \code{\link{as.matrix.paragraph2vec}}
#' @export
#' @examples
#' \dontshow{if(require(tokenizers.bpe))\{}
#' library(tokenizers.bpe)
#' ## Take data and standardise it a bit
#' data(belgium_parliament, package = "tokenizers.bpe")
#' str(belgium_parliament)
#' x <- subset(belgium_parliament, language %in% "french")
#' x$text <- tolower(x$text)
#' x$text <- gsub("[^[:alpha:]]", " ", x$text)
#' x$text <- gsub("[[:space:]]+", " ", x$text)
#' x$text <- trimws(x$text)
#' x$nwords <- sapply(strsplit(x$text, " "), length)
#' x <- subset(x, nwords < 1000 & nchar(text) > 0)
#' 
#' ## Build the model get word embeddings and nearest neighbours
#' model <- paragraph2vec(x = x, dim = 15, iter = 20)
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
                     type = c("cbow", "skip-gram"),
                     dim = 50, window = ifelse(type == "cbow", 5L, 10L), 
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
  cbow <- as.logical(type %in% "cbow")
  model <- paragraph2vec_train(trainFile = file_train, 
                               size = dim, cbow = cbow,
                               hs = hs, negative = negative, iterations = iter, window = window, alpha = lr, sample = sample,
                               min_count = min_count, threads = threads)
  model
}


#' @title Get the document or word vectors of a paragraph2vec model
#' @description Get the document or word vectors of a paragraph2vec model as a dense matrix.
#' @param x a paragraph2vec model as returned by \code{\link{paragraph2vec}} or \code{\link{read.paragraph2vec}}
#' @param which either one of 'docs' or 'words'
#' @param normalize logical indicating to normalize the embeddings. Defaults to \code{TRUE}.
#' @param encoding set the encoding of the row names to the specified encoding. Defaults to 'UTF-8'.
#' @param ... not used
#' @return a matrix with the word vectors where the rownames are the documents or words from the model vocabulary
#' @export
#' @seealso \code{\link{paragraph2vec}}, \code{\link{read.paragraph2vec}}
#' @export
#' @examples 
#' \dontshow{if(require(tokenizers.bpe))\{}
#' library(tokenizers.bpe)
#' data(belgium_parliament, package = "tokenizers.bpe")
#' x <- subset(belgium_parliament, language %in% "french")
#' x <- subset(x, nchar(text) > 0 & nchar(text) < 1000)
#' 
#' model <- paragraph2vec(x = x, dim = 15, iter = 20)
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
#' @return a logical indicating if the save process succeeded
#' @export
#' @seealso \code{\link{paragraph2vec}}
#' @examples 
#' \dontshow{if(require(tokenizers.bpe))\{}
#' library(tokenizers.bpe)
#' data(belgium_parliament, package = "tokenizers.bpe")
#' x <- subset(belgium_parliament, language %in% "french")
#' x <- subset(x, nchar(text) > 0 & nchar(text) < 1000)
#' 
#' model <- paragraph2vec(x = x, dim = 15, iter = 20)
#' 
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
#' x <- subset(x, nchar(text) > 0 & nchar(text) < 1000)
#' 
#' model <- paragraph2vec(x = x, dim = 15, iter = 20)
#' 
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
#' @description Get either 
#' \itemize{
#' \item{the embedding of documents or words}
#' \item{TODO}
#' }
#' @param object a paragraph2vec model as returned by \code{\link{paragraph2vec}} or \code{\link{read.paragraph2vec}}
#' @param newdata for type 'embedding', \code{newdata} should be TODO
#' @param type either 'embedding' or 'nearest'. Defaults to 'nearest'.
#' @param top_n show only the top n nearest neighbours. Defaults to 10.
#' @param encoding set the encoding of the text elements to the specified encoding. Defaults to 'UTF-8'. 
#' @param ... not used
#' @return depending on the type, you get a different result back:
#' \itemize{
#' \item{TODO}
#' \item{TODO}
#' }
#' @seealso \code{\link{paragraph2vec}}, \code{\link{read.paragraph2vec}}
#' @export
#' @examples 
#' ## TODO
predict.paragraph2vec <- function(object, newdata, type = c("nearest", "embedding"), top_n = 10L, encoding = "UTF-8", ...){
}

#' @export
predict.paragraph2vec_trained <- function(object, newdata, type = c("nearest", "embedding"), ...){
  predict.paragraph2vec(object = object, newdata = newdata, type = type, ...)
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
