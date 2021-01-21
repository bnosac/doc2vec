#' @title Count the number of spaces occurring in text
#' @description The C++ doc2vec functionalities in this package assume words are either separated 
#' by a space or tab symbol and that each document contains less than 1000 words.\cr 
#' This function calculates how many words there are in each element of a character vector by counting
#' the number of occurrences of the space or tab symbol.
#' @param x a character vector with text
#' @param pattern a text pattern to count which might be contained in \code{x}. Defaults to either space or tab.
#' @param ... other arguments, passed on to \code{\link{gregexpr}}
#' @return an integer vector of the same length as \code{x} indicating how many times the pattern is occurring in \code{x}
#' @export
#' @examples 
#' x <- c("Count me in.007", "this is a set  of words",
#'        "more\texamples tabs-and-spaces.only", NA)
#' txt_count_words(x)
txt_count_words <- function(x, pattern = "[ \t]", ...){
  result <- gregexpr(pattern = pattern, text = x, ...)
  sapply(result, FUN = function(x){
    test <- length(x) == 1 && x < 0
    if(is.na(test)){
      return(NA_integer_) 
    }
    if(test){
      0L
    }else{
      length(x)
    }
  }, USE.NAMES = FALSE)
}