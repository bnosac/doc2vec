#include <Rcpp.h>
#include "common_define.h"
#include "Doc2Vec.h"
#include "Vocab.h"
#include "NN.h"

// [[Rcpp::export]]
Rcpp::List paragraph2vec_train(const char * trainFile, int size = 100, 
                               int cbow = 1, int hs = 0, int negative = 5, int iterations = 5, 
                               int window = 5, double alpha = 0.05, double sample = 0.001, int min_count = 5, int threads = 1) {
  
  Rcpp::XPtr<Doc2Vec> model(new Doc2Vec(), true);
  model->train(trainFile, size, cbow, hs, negative, iterations, window, alpha, sample, min_count, threads);
  
  Vocabulary* voc_docs  = model->dvocab();
  Vocabulary* voc_words = model->wvocab();
 
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model") = model,
    Rcpp::Named("data") = Rcpp::List::create(
      Rcpp::Named("file") = trainFile,
      Rcpp::Named("n") = voc_words->m_train_words,
      Rcpp::Named("n_vocabulary") = voc_words->m_vocab_size,
      Rcpp::Named("docs") = voc_docs->m_vocab_size
    ),
    Rcpp::Named("control") = Rcpp::List::create(
      Rcpp::Named("min_count") = min_count,
      Rcpp::Named("dim") = size,
      Rcpp::Named("window") = window,
      Rcpp::Named("iter") = iterations,
      Rcpp::Named("lr") = alpha,
      Rcpp::Named("skipgram") = (cbow == 0),
      Rcpp::Named("hs") = hs,
      Rcpp::Named("negative") = negative,
      Rcpp::Named("sample") = sample
    )
  );
  out.attr("class") = "paragraph2vec_trained";
  return out;
}


// [[Rcpp::export]]
void paragraph2vec_save_model(SEXP ptr, std::string file) {
  Rcpp::XPtr<Doc2Vec> model(ptr);
  FILE * fout = fopen(file.c_str(), "wb");
  model->save(fout);
  fclose(fout);
  return;  
}

// [[Rcpp::export]]
Rcpp::List paragraph2vec_load_model(std::string file) {
  Rcpp::XPtr<Doc2Vec> model(new Doc2Vec(), true);
  FILE * fin = fopen(file.c_str(), "rb");
  model->load(fin);
  fclose(fin);
  
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model") = model,
    Rcpp::Named("model_path") = file,
    Rcpp::Named("dim") = model->dim()
  );
  out.attr("class") = "paragraph2vec";
  return out;
}


// [[Rcpp::export]]
std::vector<std::string> paragraph2vec_dictionary(SEXP ptr, std::string type = "docs") {
  Rcpp::XPtr<Doc2Vec> model(ptr);
  Vocabulary* voc;
  if(type == "docs"){
    voc = model->dvocab();
  }else if(type == "words"){
    voc = model->wvocab();
  }else{
    Rcpp::stop("type should be either doc or words");
  }
  std::vector<std::string> keys;
  for (int i = 0; i < voc->m_vocab_size; i++){
    std::string input(voc->m_vocab[i].word);
    keys.push_back(input);
  }
  return keys;
}


// [[Rcpp::export]]
Rcpp::DataFrame paragraph2vec_nearest(SEXP ptr, std::string x, std::size_t top_n = 10, std::string type = "doc2doc") {
  Rcpp::XPtr<Doc2Vec> model(ptr);
  knn_item_t knn_items[top_n];
  if(type == "doc2doc"){
    // This seems to fail regarding term2??
    model->doc_knn_docs(x.c_str(), knn_items, top_n);
  }else if(type == "word2doc"){
    model->word_knn_docs(x.c_str(), knn_items, top_n);
  }else if(type == "word2word"){
    model->word_knn_words(x.c_str(), knn_items, top_n);
  }else{
    Rcpp::stop("type should be either doc2doc, word2doc or word2word");
  }
  std::vector<std::string> keys;
  std::vector<float> distance;
  std::vector<int> rank;
  int r = 0;
  for(auto kv : knn_items) {
    std::string str(kv.word);
    keys.push_back(str);
    distance.push_back(kv.similarity);
    r = r + 1;
    rank.push_back(r);
  } 
  Rcpp::DataFrame out = Rcpp::DataFrame::create(
    Rcpp::Named("term1") = x,
    Rcpp::Named("term2") = keys,
    Rcpp::Named("similarity") = distance,
    Rcpp::Named("rank") = rank,
    Rcpp::Named("stringsAsFactors") = false
  );
  return out;
}


// [[Rcpp::export]]
Rcpp::List paragraph2vec_embedding(SEXP ptr, std::string type = "docs") {
  Rcpp::XPtr<Doc2Vec> model(ptr);

  NN * net = model->nn();
  auto m_dim = net->m_dim;
  auto m_vocab_size = net->m_vocab_size;
  auto m_corpus_size = net->m_corpus_size;
  auto m_dsyn0 = net->m_dsyn0;
  
  Rcpp::NumericMatrix embedding(m_corpus_size, m_dim);
  std::fill(embedding.begin(), embedding.end(), Rcpp::NumericVector::get_na());
  for (int a = 0; a < m_corpus_size; a++){
    for (int b = 0; b < m_dim; b++) {
      embedding(a, b) = (float)(m_dsyn0[a * m_dim + b]);
    }
  }
    
  
  //net->m_syn0
  //net->m_dsyn0
  //fwrite(m_syn0, sizeof(real), m_vocab_size * m_dim, fout);
  //fwrite(m_dsyn0, sizeof(real), m_corpus_size * m_dim, fout);
  //if(m_hs) fwrite(m_syn1, sizeof(real), m_vocab_size * m_dim, fout);
  //if(m_negtive) fwrite(m_syn1neg, sizeof(real), m_vocab_size * m_dim, fout);
  //return;
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("embedding") = embedding,
    Rcpp::Named("m_dim") = m_dim,
    Rcpp::Named("m_vocab_size") = m_vocab_size,
    Rcpp::Named("m_corpus_size") = m_corpus_size,
    Rcpp::Named("m_hs") = net->m_hs,
    Rcpp::Named("m_negtive") = net->m_negtive
  );
  return out;
}
  


