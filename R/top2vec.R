
#' @title Distributed Representations of Topics
#' @description Perform text clustering by using semantic embeddings of documents and words
#' to find topics of text documents which are semantically similar.
#' @param x either an object returned by \code{\link[doc2vec]{paragraph2vec}} or a data.frame 
#' with columns `doc_id` and `text` storing document ids and texts as character vectors or a matrix with document embeddings to cluster
#' or a list with elements docs and words containing document embeddings to cluster and word embeddings for deriving topic summaries
#' @param data optionally, a data.frame with columns `doc_id` and `text` representing documents. 
#' This dataset is just stored, in order to extract the text of the most similar documents to a topic.
#' If it also contains a field `text_doc2vec`, this will be used to indicate the most relevant topic words
#' by class-based tfidf
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
#' \item{data: a data.frame with columns doc_id and text}
#' \item{size: a vector of frequency statistics of topic occurrence}
#' \item{k: the number of clusters}
#' \item{control: a list of control arguments to doc2vec / umap / dbscan}
#' }
#' @note The topic '0' is the noise topic
#' @examples 
#' \donttest{
#' \dontshow{if(require(word2vec) && require(uwot) && require(dbscan) && require(udpipe))\{}
#' library(word2vec)
#' library(uwot)
#' library(dbscan)
#' data(be_parliament_2020, package = "doc2vec")
#' x      <- data.frame(doc_id = be_parliament_2020$doc_id,
#'                      text   = be_parliament_2020$text_nl,
#'                      stringsAsFactors = FALSE)
#' x$text <- txt_clean_word2vec(x$text)
#' x      <- subset(x, txt_count_words(text) < 1000)
#' d2v    <- paragraph2vec(x, type = "PV-DBOW", dim = 50, 
#'                         lr = 0.05, iter = 10,
#'                         window = 15, hs = TRUE, negative = 0,
#'                         sample = 0.00001, min_count = 5, 
#'                         threads = 1)
#' # write.paragraph2vec(d2v, "d2v.bin")
#' # d2v    <- read.paragraph2vec("d2v.bin")
#' model  <- top2vec(d2v, data = x,
#'                   control.dbscan = list(minPts = 50), 
#'                   control.umap = list(n_neighbors = 15L, n_components = 4), trace = TRUE)
#' model  <- top2vec(d2v, data = x,
#'                   control.dbscan = list(minPts = 50), 
#'                   control.umap = list(n_neighbors = 15L, n_components = 3), umap = tumap, 
#'                   trace = TRUE)
#'                                   
#' info   <- summary(model, top_n = 7)
#' info$topwords
#' info$topdocs
#' library(udpipe)
#' info   <- summary(model, top_n = 7, type = "c-tfidf")
#' info$topwords
#' 
#' ## Change the model: reduce doc2vec model to 2D
#' model  <- update(model, type = "umap", 
#'                  n_neighbors = 100, n_components = 2, metric = "cosine", umap = tumap, 
#'                  trace = TRUE)
#' info   <- summary(model, top_n = 7)
#' info$topwords
#' info$topdocs
#' 
#' ## Change the model: have minimum 200 points for the core elements in the hdbscan density
#' model  <- update(model, type = "hdbscan", minPts = 200, trace = TRUE)
#' info   <- summary(model, top_n = 7)
#' info$topwords
#' info$topdocs
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
#' }
#' 
#' ##
#' ## Example on a small sample 
#' ##  with unrealistic hyperparameter settings especially regarding dim / iter / n_epochs
#' ##  in order to have a basic example finishing < 5 secs
#' ##
#' \dontshow{if(require(word2vec) && require(uwot) && require(dbscan))\{}
#' library(uwot)
#' library(dbscan)
#' library(word2vec)
#' data(be_parliament_2020, package = "doc2vec")
#' x        <- data.frame(doc_id = be_parliament_2020$doc_id,
#'                        text   = be_parliament_2020$text_nl,
#'                        stringsAsFactors = FALSE)
#' x        <- head(x, 1000)
#' x$text   <- txt_clean_word2vec(x$text)
#' x        <- subset(x, txt_count_words(text) < 1000)
#' d2v      <- paragraph2vec(x, type = "PV-DBOW", dim = 10, 
#'                           lr = 0.05, iter = 0,
#'                           window = 5, hs = TRUE, negative = 0,
#'                           sample = 0.00001, min_count = 5)
#' emb      <- list(docs  = as.matrix(d2v, which = "docs"),
#'                  words = as.matrix(d2v, which = "words"))
#' model    <- top2vec(emb, 
#'                     data = x,
#'                     control.dbscan = list(minPts = 50), 
#'                     control.umap = list(n_neighbors = 15, n_components = 2, 
#'                                         init = "spectral"), 
#'                     umap = tumap, trace = TRUE)
#' info     <- summary(model, top_n = 7)
#' print(info, top_n = c(5, 2))
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
top2vec <- function(x, 
                    data = data.frame(doc_id = character(), text = character(), stringsAsFactors = FALSE), 
                    control.umap = list(n_neighbors = 15L, n_components = 5L, metric = "cosine"), 
                    control.dbscan = list(minPts = 100L), 
                    control.doc2vec = list(), 
                    umap = uwot::umap,
                    trace = FALSE, ...){
  requireNamespace("uwot")
  requireNamespace("dbscan")
  stopifnot(inherits(x, c("data.frame", "paragraph2vec", "paragraph2vec_trained", "matrix", "list")))
  model <- NULL
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
                        data = data,
                        size = table(clusters$cluster),
                        k = length(unique(clusters$cluster)),
                        control = list(doc2vec = control.doc2vec, umap = control.umap, dbscan = control.dbscan)), 
                   class = "top2vec")
  out
}

#' @export
print.top2vec <- function(x, ...){
  cat(sprintf("Top2vec model trained on %s documents", nrow(x$embedding$docs)), sep = "\n")
  cat(sprintf("  number of topics: %s", x$k), sep = "\n")
  cat(sprintf("  topic distribution: %s", paste(round(prop.table(x$size), 2), collapse = " ")), sep = "\n")
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
#' @param type a character string with the type of summary information to extract for the topwords. Either 'similarity' or 'c-tfidf'.  
#' The first extracts most similar words to the topic based on semantic similarity, the second by extracting
#' the words with the highest tf-idf score for each topic 
#' @param top_n integer indicating to find the \code{top_n} most similar words to a topic
#' @param data a data.frame with columns `doc_id` and `text` representing documents. 
#' For each topic, the function extracts the most similar documents. 
#' And in case \code{type} is \code{'c-tfidf'} it get the words with the highest tf-idf scores for each topic.
#' @param embedding_words a matrix of word embeddings to limit the most similar words to. Defaults to 
#' the embedding of words from the \code{object}
#' @param embedding_docs a matrix of document embeddings to limit the most similar documents to. Defaults to 
#' the embedding of words from the \code{object}
#' @param ... not used 
#' @export
#' @examples 
#' # For an example, look at the documentation of ?top2vec
summary.top2vec <- function(object, 
                            type = c("similarity", "c-tfidf"), top_n = 10, data = object$data, embedding_words = object$embedding$words, 
                            embedding_docs = object$embedding$docs, ...){
  recode <- function(x, from, to){
    to[match(x, from)]
  }
  
  type <- match.arg(type)
  topic_idx       <- split(x = seq_along(object$dbscan$cluster), f = object$dbscan$cluster)
  topic_centroids <- lapply(topic_idx, FUN = function(i) colMeans(object$embedding$docs[i, , drop = FALSE]))
  topic_medoids   <- lapply(topic_idx, FUN = function(i) apply(object$embedding$docs[i, , drop = FALSE], MARGIN = 2, FUN = stats::median))
  #topic_medoids <- do.call(rbind, topic_medoids)
  if(type == "similarity"){
    topwords <- lapply(topic_centroids, FUN = function(topic){
      similarity <- doc2vec::paragraph2vec_similarity(y = embedding_words, x = topic, top_n = top_n)
      data.frame(term       = similarity$term2, 
                 similarity = similarity$similarity, 
                 rank       = similarity$rank, stringsAsFactors = FALSE)
    })  
  }else if(type == "c-tfidf"){
    if(!requireNamespace("udpipe")){
      stop("c-tfidf requires the udpipe package: install.packages('udpipe')")
    }
    if("text_doc2vec" %in% colnames(data)){
      data$text <- data$text_doc2vec
    }
    stopifnot(all(c("doc_id", "text") %in% colnames(data)))
    stopifnot(all(rownames(object$embedding$docs) %in% data$doc_id))
    data$topic <- recode(data$doc_id, from = rownames(object$embedding$docs), to = object$dbscan$cluster)
    dtf        <- udpipe::document_term_frequencies(x = data$text, document = data$topic, split = "[[:space:]]+")
    dtf        <- udpipe::document_term_frequencies_statistics(dtf)
    topicnrs <- as.character(sort(as.integer(unique(dtf$doc_id))), decreasing = FALSE)
    topwords <- lapply(topicnrs, FUN = function(topicnr){
      x <- dtf[dtf$doc_id %in% topicnr, ]
      x <- x[order(x$tf_idf, decreasing = TRUE), ]
      x <- head(x, n = top_n)
      data.frame(term       = x$term, 
                 similarity = x$tf_idf, 
                 rank       = seq_len(nrow(x)), stringsAsFactors = FALSE)
    })
    names(topwords) <- topicnrs
  }
  
  topdocs <- lapply(topic_centroids, FUN = function(topic){
    similarity <- doc2vec::paragraph2vec_similarity(y = embedding_docs, x = topic, top_n = top_n)
    data.frame(doc_id     = similarity$term2, 
               text       = recode(similarity$term2, from = data$doc_id, to = data$text), 
               similarity = similarity$similarity, 
               rank       = similarity$rank, stringsAsFactors = FALSE)
  })
  out <- structure(list(k = object$k, size = object$size, topwords = topwords, topdocs = topdocs, centroids = topic_centroids, medoids = topic_medoids), class = "top2vec_summary")
  out
}


#' @export
print.top2vec_summary <- function(x, top_n = 3, ...){
  if(length(top_n) == 1){
    top_n <- c(top_n, top_n)
  }
  cat(sprintf("Top2vec model with %s topics", x$k), sep = "\n")
  cat(sprintf("Topic distribution: %s", paste(round(prop.table(x$size), digits = 2), collapse = " ")), sep = "\n")
  #m <- max(nchar(unlist(mapply(seq_along(x$topwords), x$topwords, FUN = function(i, data) sprintf("Topic %s: %s", i-1, paste(head(data$term, top_n[1]), collapse = " "))))))
  for(i in seq_len(x$k)){
    label <- names(x$topwords)[i]
    words <- head(x$topwords[[i]]$term, top_n[1])
    docs  <- head(x$topdocs[[i]]$text, top_n[2])
    
    txt <- sprintf("Topic %s: %s", label, paste(words, collapse = " "))
    cat(paste(rep("-", nchar(txt)), collapse = ""), sep = "\n")
    cat(txt, sep = "\n")
    cat(paste(rep("-", nchar(txt)), collapse = ""), sep = "\n")
    cat(paste(sprintf("  (%s) %s", seq_along(docs), docs), collapse = "\n"), sep = "\n")
  }
}