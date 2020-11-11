library(tokenizers.bpe)
library(doc2vec)
data(belgium_parliament)
str(belgium_parliament)
belgium_parliament$id   <- seq_len(nrow(belgium_parliament))
belgium_parliament$text <- tolower(belgium_parliament$text)
belgium_parliament$text <- gsub("[^[:alpha:]]", " ", belgium_parliament$text)
belgium_parliament$text <- gsub("[[:space:]]+", " ", belgium_parliament$text)
f <- tempfile(pattern = "traindata_doc2vec_", fileext = ".txt")
docs <- sprintf("_*%s %s", belgium_parliament$id, belgium_parliament$text)
docs <- sprintf("doc_%s %s", belgium_parliament$id, belgium_parliament$text)
writeLines(docs, con = f)

model <- doc2vec:::paragraph2vec_train(trainFile = f, 
                                 size = 15, cbow = 1, hs = 1, negative = 0, iter = 20, 
                                 window = 5, alpha = 0.025, sample = 0.001, min_count = 2, threads = 1)

voc <- doc2vec:::paragraph2vec_dictionary(model$model, type = "docs")
voc <- doc2vec:::paragraph2vec_dictionary(model$model, type = "words")
grep(pattern = "franck", x = voc, value = TRUE, ignore.case = TRUE)
doc2vec:::paragraph2vec_save_model(model$model, "test.bin")
m <- doc2vec:::paragraph2vec_load_model("test.bin")

voc <- doc2vec:::paragraph2vec_dictionary(model$model, type = "docs")
z <- doc2vec:::paragraph2vec_embedding(model$model)
rownames(z$embedding) <- voc
str(z)


z <- doc2vec:::paragraph2vec_nearest(model$model, "geld", type = "word2word", top_n = 10)
z
z <- doc2vec:::paragraph2vec_nearest(model$model, "francken", type = "word2word", top_n = 10)
z
z <- doc2vec:::paragraph2vec_nearest(model$model, "koning", type = "word2word", top_n = 10)
z
z <- doc2vec:::paragraph2vec_nearest(model$model, "francken", type = "word2doc", top_n = 3)
z
z <- doc2vec:::paragraph2vec_nearest(model$model, docs[1], type = "doc2doc", top_n = 3) ## FAILS CURRENTLY
z

