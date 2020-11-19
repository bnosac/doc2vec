library(tokenizers.bpe)
library(doc2vec)
data(belgium_parliament)
belgium_parliament$id   <- seq_len(nrow(belgium_parliament))
belgium_parliament$text <- tolower(belgium_parliament$text)
belgium_parliament$text <- gsub("[^[:alpha:]]", " ", belgium_parliament$text)
belgium_parliament$text <- gsub("[[:space:]]+", " ", belgium_parliament$text)
belgium_parliament$text <- trimws(belgium_parliament$text)
belgium_parliament <- subset(belgium_parliament, nchar(text) > 0)
belgium_parliament$len <- sapply(strsplit(belgium_parliament$text, " "), length)
belgium_parliament <- subset(belgium_parliament, len < 1000)
# belgium_parliament$text <- sapply(belgium_parliament$text, FUN=function(x){
#   paste(sample(c(letters[1:6], " "), size = nchar(x), replace = TRUE), collapse = "")
# })
f <- tempfile(pattern = "traindata_doc2vec_", fileext = ".txt")
docs <- sprintf("_*%06d %s", belgium_parliament$id, belgium_parliament$text)
docs <- sprintf("doc_%06d %s", belgium_parliament$id, belgium_parliament$text)
writeLines(docs, con = f)

model <- doc2vec:::paragraph2vec_train(trainFile = f, 
                                 size = 5, cbow = 1, hs = 1, negative = 0, iter = 20, 
                                 window = 5, alpha = 0.025, sample = 0.001, min_count = 2, threads = 1)

voc <- doc2vec:::paragraph2vec_dictionary(model$model, type = "words")
voc <- doc2vec:::paragraph2vec_dictionary(model$model, type = "docs")
length(voc)
test <- data.frame(doc = head(voc, 2000), 
                       id = as.integer(gsub("doc_", "", head(voc, 2000))), 
                       test = seq_along(head(voc, 2000)))
test[1702:1705, ] ## WHAT IS GOING WRONG HERE
voc <- doc2vec:::paragraph2vec_dictionary(model$model, type = "words")
grep(pattern = "franck", x = voc, value = TRUE, ignore.case = TRUE)
doc2vec:::paragraph2vec_save_model(model$model, "test.bin")
m     <- doc2vec:::paragraph2vec_load_model("test.bin")
model <- doc2vec:::paragraph2vec_load_model("test.bin")

voc <- doc2vec:::paragraph2vec_dictionary(model$model, type = "docs")
system.time(z <- doc2vec:::paragraph2vec_embedding(model$model))
#rownames(z$embedding) <- voc
str(z)


x <- list(example = c("geld", "francken"),
          hi = c("geld", "francken", "koning"),
          test = c("geld"),
          repr = c("geld", "francken", "koning"))
z <- doc2vec:::paragraph2vec_infer(model$model, x)
z

emb <- doc2vec:::paragraph2vec_embedding(model$model, type = "words", normalize = TRUE)
emb["geld", ]
ruimtehol::embedding_similarity(emb["geld", , drop = FALSE], emb, top_n = 10, type = "dot")
emb <- doc2vec:::paragraph2vec_embedding(model$model, type = "docs")
emb["doc_000001", ]
ruimtehol::embedding_similarity(emb["doc_000001", ], emb, top_n = 10, type = "dot")

#return obj_knn_objs(search, NULL, true, true, knns, k);
z <- doc2vec:::paragraph2vec_nearest(model$model, "geld", type = "word2word", top_n = 10)
z
z <- doc2vec:::paragraph2vec_nearest(model$model, "francken", type = "word2word", top_n = 10)
z
z <- doc2vec:::paragraph2vec_nearest(model$model, "koning", type = "word2word", top_n = 10)
z
z <- doc2vec:::paragraph2vec_nearest(model$model, "francken", type = "word2doc", top_n = 3)
z
z <- doc2vec:::paragraph2vec_nearest(model$model, "doc_000001", type = "doc2doc", top_n = 3)
z
z <- doc2vec:::paragraph2vec_nearest(model$model, docs[1], type = "doc2doc", top_n = 3) ## FAILS CURRENTLY
z


z <- doc2vec:::paragraph2vec_nearest(model$model, "doc_000001", type = "doc2doc", top_n = 2000)
subset(z, term2 == "doc_000002")
##return obj_knn_objs(search, NULL, false, false, knns, k);
emb <- doc2vec:::paragraph2vec_embedding(model$model, type = "words")
emb["francken", ]
emb <- doc2vec:::paragraph2vec_embedding(model$model, type = "docs")

emb <- doc2vec:::paragraph2vec_embedding(model$model)
emb <- head(emb$embedding, 2000)
emb["doc_000001", ]
rownames(emb) <- head(doc2vec:::paragraph2vec_dictionary(model$model, type = "docs"), 2000)
ruimtehol::embedding_similarity(emb["doc_000001", , drop = FALSE], emb, type = "dot", top_n = 3)
sum(
  emb["doc_000001", ] * emb["doc_000002", ])
