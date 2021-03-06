% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/top2vec.R
\name{summary.top2vec}
\alias{summary.top2vec}
\title{Get summary information of a top2vec model}
\usage{
\method{summary}{top2vec}(
  object,
  type = "default",
  top_n = 10,
  data = object$data,
  embedding_words = object$embedding$words,
  embedding_docs = object$embedding$docs,
  ...
)
}
\arguments{
\item{object}{an object of class \code{top2vec} as returned by \code{\link{top2vec}}}

\item{type}{a character string with the type of summary information to extract. Currently not used}

\item{top_n}{integer indicating to find the \code{top_n} most similar words to a topic}

\item{data}{a data.frame with columns `doc_id` and `text` representing documents. For each topic, the function extracts the most similar documents.}

\item{embedding_words}{a matrix of word embeddings to limit the most similar words to. Defaults to 
the embedding of words from the \code{object}}

\item{embedding_docs}{a matrix of document embeddings to limit the most similar documents to. Defaults to 
the embedding of words from the \code{object}}

\item{...}{not used}
}
\description{
Get summary information of a top2vec model. Namely the topic centers and the most similar words
to a certain topic
}
\examples{
# For an example, look at the documentation of ?top2vec
}
