
#' @title Distributed Representations of Topics
#' @description Perform text clustering by using semantic embeddings of documents and words
#' to find topics of text documents which are semantically similar.
#' @param x either an object returned by \code{\link[doc2vec]{paragraph2vec}} or a data.frame 
#' with columns `doc_id` and `text` storing document ids and texts as character vectors or a matrix with document embeddings to cluster
#' or a list with elements docs and words containing document embeddings to cluster and word embeddings for deriving topic summaries
#' @param control.umap a list of arguments to pass on to \code{\link[uwot]{umap}} for reducing the dimensionality of the embedding space
#' @param control.dbscan a list of arguments to pass on to \code{\link[dbscan]{hdbscan}} for clustering the reduced embedding space
#' @param control.doc2vec optionally, a list of arguments to pass on to \code{\link[doc2vec]{paragraph2vec}} in case \code{x} is a data.frame
#' instead of a doc2vec model trained by \code{\link[doc2vec]{paragraph2vec}}
#' @param umap function to apply UMAP. Defaults to \code{\link[uwot]{umap}}, can as well be \code{\link[uwot]{tumap}}
#' @param trace logical indicating to print evolution of the algorithm
#' @param ... further arguments not used yet
#' @export
#' @references \url{https://arxiv.org/abs/2008.09470}
#' @seealso \code{\link[doc2vec]{paragraph2vec}}
#' @return an object of class \code{top2vec} which is a list with elements
#' \itemize{
#' \item{embedding: a list of matrices with word and document embeddings}
#' \item{doc2vec: a doc2vec model}
#' \item{umap: a matrix of representations of the documents of \code{x}}
#' \item{dbscan: the result of the hdbscan clustering}
#' }
#' @examples 
#' \dontshow{if(require(word2vec) && require(uwot) && require(dbscan))\{}
#' library(word2vec)
#' library(uwot)
#' library(dbscan)
#' data(be_parliament_2020, package = "doc2vec")
#' x      <- data.frame(doc_id = be_parliament_2020$doc_id,
#'                      text   = be_parliament_2020$text_nl,
#'                      stringsAsFactors = FALSE)
#' x$text <- txt_clean_word2vec(x$text)
#' x      <- subset(x, txt_count_words(text) < 1000)
#' \donttest{
#' d2v    <- paragraph2vec(x, type = "PV-DBOW", dim = 50, 
#'                         lr = 0.05, iter = 10,
#'                         window = 15, hs = TRUE, negative = 0,
#'                         sample = 0.00001, min_count = 5, 
#'                         threads = 1)
#' # write.paragraph2vec(d2v, "d2v.bin")
#' # d2v    <- read.paragraph2vec("d2v.bin")
#' model  <- top2vec(d2v, 
#'                   control.dbscan = list(minPts = 50), 
#'                   control.umap = list(n_neighbors = 15L, n_components = 4), trace = TRUE)
#' model  <- top2vec(d2v, 
#'                   control.dbscan = list(minPts = 50), 
#'                   control.umap = list(n_neighbors = 15L, n_components = 3), umap = tumap, 
#'                   trace = TRUE)
#'                                   
#' info   <- summary(model, top_n = 7)
#' info$topwords
#' 
#' ## Change the model: reduce doc2vec model to 2D
#' model  <- update(model, type = "umap", 
#'                  n_neighbors = 100, n_components = 2, metric = "cosine", umap = tumap, 
#'                  trace = TRUE)
#' info   <- summary(model, top_n = 7)
#' info$topwords
#' 
#' ## Change the model: have minimum 200 points for the core elements in the hdbscan density
#' model  <- update(model, type = "hdbscan", minPts = 200, trace = TRUE)
#' info   <- summary(model, top_n = 7)
#' info$topwords
#' }
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
top2vec <- function(x, 
                    control.umap = list(n_neighbors = 15L, n_components = 5L, metric = "cosine"), 
                    control.dbscan = list(minPts = 100L), 
                    control.doc2vec = list(), 
                    umap = uwot::umap,
                    trace = FALSE, ...){
  requireNamespace("uwot")
  requireNamespace("dbscan")
  stopifnot(inherits(x, c("data.frame", "paragraph2vec", "paragraph2vec_trained", "matrix")))
  if(inherits(x, "data.frame")){
    stopifnot(all(c("doc_id", "text") %in% colnames(x)) && is.character(x$text))
    control.doc2vec$x <- x
    if(trace){
      cat(sprintf("%s building doc2vec model", Sys.time()), sep = "\n")
    }
    model <- do.call(doc2vec::paragraph2vec, control.doc2vec)
    if(trace){
      cat(sprintf("%s extracting doc2vec embeddings", Sys.time()), sep = "\n")
    }
    embedding_docs  <- as.matrix(model, which = "docs")
    embedding_words <- as.matrix(model, which = "words")
  }else if(inherits(x, c("paragraph2vec", "paragraph2vec_trained"))){
    model <- x
    if(trace){
      cat(sprintf("%s extracting doc2vec embeddings", Sys.time()), sep = "\n")
    }
    embedding_docs  <- as.matrix(model, which = "docs")
    embedding_words <- as.matrix(model, which = "words")
  }else if(inherits(x, "matrix")){
    embedding_docs  <- x
    embedding_words <- NULL
  }else if(inherits(x, "list") && all(c("docs", "words") %in% names(x)) && is.matrix(x$docs) && is.matrix(x$words)){
    embedding_docs  <- x$docs
    embedding_words <- x$words
  }else{
    stop("Looks like x is not the right data type for passing it on to top2vec")
  }
  ## UMAP
  if(trace){
    cat(sprintf("%s performing UMAP dimensionality reduction on the doc2vec embedding space", Sys.time()), sep = "\n")
  }
  control.umap$X   <- embedding_docs
  embedding_umap   <- do.call(umap, control.umap)
  ## HDBSCAN
  if(trace){
    cat(sprintf("%s performing HDBSCAN density based clustering", Sys.time()), sep = "\n")
  }
  control.dbscan$x <- embedding_umap
  clusters         <- do.call(dbscan::hdbscan, control.dbscan)
  out <- structure(list(embedding = list(words = embedding_words, docs = embedding_docs),
                        doc2vec = model, 
                        umap = embedding_umap,
                        umap_FUN = umap,
                        dbscan = clusters,
                        size = table(clusters$cluster),
                        k = length(unique(model$dbscan$cluster)),
                        control = list(doc2vec = control.doc2vec, umap = control.umap, dbscan = control.dbscan)), 
                   class = "top2vec")
  out
}


#' @title Update a Top2vec model
#' @description Update a Top2vec model by updating the UMAP dimension reduction together with the HDBSCAN clustering
#' or update only the HDBSCAN clustering
#' @param object an object of class \code{top2vec} as returned by \code{\link{top2vec}}
#' @param type a character string indicating what to udpate. Either 'umap' or 'hdbscan' where the former (type = 'umap') indicates to 
#' update the umap as well as the hdbscan procedure and the latter (type = 'hdbscan') indicates to update only the hdbscan step.
#' @param umap see \code{umap} argument in \code{\link{top2vec}}
#' @param trace logical indicating to print evolution of the algorithm
#' @param ... further arguments either passed on to \code{\link[dbscan]{hdbscan}} in case type is 'hdbscan' or to \code{\link[uwot]{umap}}
#' in case type is 'umap'
#' @return an updated top2vec object
#' @export
#' @examples 
#' # For an example, look at the documentation of ?top2vec
update.top2vec <- function(object, type = c("umap", "hdbscan"), 
                           umap = object$umap_FUN, trace = FALSE, ...){
  type <- match.arg(type)
  if(type == "umap"){
    t2vec <- top2vec(object$doc2vec, control.umap = list(...), control.dbscan = object$control$dbscan, trace = trace, umap = umap)
  }else if(type == "hdbscan"){
    requireNamespace("dbscan")
    if(trace){
      cat(sprintf("%s performing HDBSCAN density based clustering", Sys.time()), sep = "\n")
    }
    t2vec <- object
    t2vec$dbscan <- dbscan::hdbscan(object$umap, ...)
  }
  t2vec
}


#' @title Get summary information of a top2vec model
#' @description Get summary information of a top2vec model. Namely the topic centers and the most similar words
#' to a certain topic
#' @param object an object of class \code{top2vec} as returned by \code{\link{top2vec}}
#' @param type a character string with the type of summary information to extract. Currently not used
#' @param top_n integer indicating to find the \code{top_n} most similar words to a topic
#' @param embedding_words a matrix of word embeddings to limit the most similar words to. Defaults to 
#' the embedding of words from the \code{object}
#' @param ... not used 
#' @export
#' @examples 
#' # For an example, look at the documentation of ?top2vec
summary.top2vec <- function(object, type = "default", top_n = 10, embedding_words = object$embedding$words, ...){
  type <- match.arg(type)
  topic_idx       <- split(x = seq_along(object$dbscan$cluster), f = object$dbscan$cluster)
  topic_centroids <- lapply(topic_idx, FUN = function(i) colMeans(object$embedding$docs[i, , drop = FALSE]))
  topic_medoids   <- lapply(topic_idx, FUN = function(i) apply(object$embedding$docs[i, , drop = FALSE], MARGIN = 2, FUN = stats::median))
  #topic_medoids <- do.call(rbind, topic_medoids)
  topwords <- lapply(topic_centroids, FUN = function(topic){
    similarity <- doc2vec::paragraph2vec_similarity(y = embedding_words, x = topic, top_n = top_n)
    data.frame(term = similarity$term2, similarity = similarity$similarity, rank = similarity$rank, stringsAsFactors = FALSE)
  })
  list(topwords = topwords, centroids = topic_centroids, medoids = topic_medoids)
}