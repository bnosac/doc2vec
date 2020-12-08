#include <Rcpp.h>
#include "common_define.h"
#include "Doc2Vec.h"
#include "Vocab.h"
#include "NN.h"
#include "TaggedBrownCorpus.h"
#include "common_define.h"
// [[Rcpp::export]]
Rcpp::List paragraph2vec_train(const char * trainFile, int size = 100, 
                               int cbow = 1, int hs = 0, int negative = 5, int iterations = 5, 
                               int window = 5, double alpha = 0.05, double sample = 0.001, int min_count = 5, int threads = 1,
                               int trace = 0) {
  
  Rcpp::XPtr<Doc2Vec> model(new Doc2Vec(), true);
  model->train(trainFile, size, cbow, hs, negative, iterations, window, alpha, sample, min_count, threads, trace);
  
  Vocabulary* voc_docs  = model->dvocab();
  Vocabulary* voc_words = model->wvocab();
 
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model") = model,
    Rcpp::Named("data") = Rcpp::List::create(
      Rcpp::Named("file") = trainFile,
      Rcpp::Named("n") = voc_words->m_train_words,
      Rcpp::Named("n_vocabulary") = voc_words->m_vocab_size,
      Rcpp::Named("n_docs") = voc_docs->m_vocab_size - 1
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
  long long voc_size;
  if(type == "docs"){
    voc = model->dvocab();
    voc_size = voc->m_vocab_size - 1;
  }else if(type == "words"){
    voc = model->wvocab();
    voc_size = voc->m_vocab_size;
  }else{
    Rcpp::stop("type should be either doc or words");
  }
  std::vector<std::string> keys;
  for (int i = 0; i < voc_size; i++){
    std::string input(voc->m_vocab[i].word);
    keys.push_back(input);
  }
  return keys;
}


// [[Rcpp::export]]
Rcpp::DataFrame paragraph2vec_nearest(SEXP ptr, std::string x, int top_n = 10, std::string type = "doc2doc") {
  Rcpp::XPtr<Doc2Vec> model(ptr);
  knn_item_t knn_items[100];
  if(type == "doc2doc"){
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
    if(r >= top_n || r >= 100) {
      break;
    }
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
Rcpp::List paragraph2vec_nearest_sentence(SEXP ptr, Rcpp::List x, int top_n = 10) {
  Rcpp::XPtr<Doc2Vec> model(ptr);
  real * infer_vector = NULL;
  //int errnr = posix_memalign((void **)&infer_vector, 128, model->dim() * sizeof(real));
  infer_vector = (float *)_aligned_malloc(model->dim() * sizeof(real), 128);
  //if(errnr != 0) Rcpp::stop("posix_memalign failed");
  
  Rcpp::List similarities(x.size());
  Rcpp::CharacterVector rownames_(x.names());
  similarities.attr("names") = rownames_;
  for(int i = 0; i < x.size(); ++i){
    TaggedDocument doc;
    std::vector<std::string> line = Rcpp::as<std::vector<std::string>>(x[i]);
    line.push_back("</s>");
    doc.m_word_num = line.size();
    for(int j = 0; j < doc.m_word_num; j++){
      strcpy(doc.m_words[j], line[j].c_str());
    }
    model->infer_doc(&doc, infer_vector);
    // Get closest docs to sentence
    knn_item_t knn_items[100];
    model->sent_knn_docs(&doc, knn_items, top_n, infer_vector);
    // Collect result in data.frame
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
      if(r >= top_n || r >= 100) {
        break;
      }
    } 
    Rcpp::DataFrame out = Rcpp::DataFrame::create(
      Rcpp::Named("term1") = Rcpp::as<std::string>(rownames_(i)),
      Rcpp::Named("term2") = keys,
      Rcpp::Named("similarity") = distance,
      Rcpp::Named("rank") = rank,
      Rcpp::Named("stringsAsFactors") = false
    );
    similarities[i] = out;
  }
  _aligned_free(infer_vector);
  return similarities;
}


// [[Rcpp::export]]
Rcpp::NumericMatrix paragraph2vec_embedding(SEXP ptr, std::string type = "docs", bool normalize = true) {
  Rcpp::XPtr<Doc2Vec> model(ptr);
  NN * net = model->nn();
  long long m_dim = net->m_dim;
  long long m_vocab_size  = net->m_vocab_size;
  long long m_corpus_size = net->m_corpus_size;
  long long vocab_size;
  Vocabulary* voc;
  real * m_dsyn0;
  if(type == "docs"){
    if(normalize){
      m_dsyn0 = net->m_dsyn0norm;
    }else{
      m_dsyn0 = net->m_dsyn0;
    }
    vocab_size = m_corpus_size;
    vocab_size = vocab_size - 1;
    voc = model->dvocab();
  }else if(type == "words"){
    if(normalize){
      m_dsyn0 = net->m_syn0norm;
    }else{
      m_dsyn0 = net->m_syn0;
    }
    vocab_size = m_vocab_size;
    voc = model->wvocab();
  }else{
    Rcpp::stop("type should be either docs or words");
  }
  Rcpp::NumericMatrix embedding(vocab_size, m_dim);
  // Rownames of the embedding matrix
  Rcpp::CharacterVector rownames_(vocab_size);
  for (int i = 0; i < vocab_size; i++){
    std::string input(voc->m_vocab[i].word);
    rownames_(i) = input;
  }
  rownames(embedding) = rownames_;
  std::fill(embedding.begin(), embedding.end(), Rcpp::NumericVector::get_na());
  for (int a = 0; a < vocab_size; a++){
    for (int b = 0; b < m_dim; b++) {
      embedding(a, b) = (float)(m_dsyn0[a * m_dim + b]);
    }
  }
  return embedding;
}
  

    
  


// [[Rcpp::export]]
Rcpp::NumericMatrix paragraph2vec_infer(SEXP ptr, Rcpp::List x) {
    Rcpp::XPtr<Doc2Vec> model(ptr);
    auto m_dim = model->dim();
    Rcpp::NumericMatrix embedding(x.size(), model->dim());
    Rcpp::CharacterVector rownames_ = x.names();
    rownames(embedding) = rownames_;
    std::fill(embedding.begin(), embedding.end(), Rcpp::NumericVector::get_na());
    
    real * infer_vector = NULL;
    //int errnr = posix_memalign((void **)&infer_vector, 128, model->dim() * sizeof(real));
    infer_vector = (float *)_aligned_malloc(model->dim() * sizeof(real), 128);
    //if(errnr != 0) Rcpp::stop("posix_memalign failed");
    
    for(int i = 0; i < x.size(); ++i){
      TaggedDocument doc;
      std::vector<std::string> line = Rcpp::as<std::vector<std::string>>(x[i]);
      line.push_back("</s>");
      doc.m_word_num = line.size();
      for(int j = 0; j < doc.m_word_num; j++){
        strcpy(doc.m_words[j], line[j].c_str());
      }
      model->infer_doc(&doc, infer_vector);
      for (int b = 0; b < m_dim; b++) {
        embedding(i, b) = (float)(infer_vector[b]);
      }
    }
    _aligned_free(infer_vector);
    return embedding;
}


// [[Rcpp::export]]
Rcpp::NumericMatrix paragraph2vec_embedding_subset(SEXP ptr, Rcpp::CharacterVector x, std::string type = "docs", bool normalize = true) {
  Rcpp::XPtr<Doc2Vec> model(ptr);
  NN * net = model->nn();
  long long m_dim = net->m_dim;
  Vocabulary* voc;
  real * m_dsyn0;
  if(type == "docs"){
    if(normalize){
      m_dsyn0 = net->m_dsyn0norm;
    }else{
      m_dsyn0 = net->m_dsyn0;
    }
    voc = model->dvocab();
  }else if(type == "words"){
    if(normalize){
      m_dsyn0 = net->m_syn0norm;
    }else{
      m_dsyn0 = net->m_syn0;
    }
    voc = model->wvocab();
  }else{
    Rcpp::stop("type should be either docs or words");
  }
  Rcpp::NumericMatrix embedding(x.size(), m_dim);
  rownames(embedding) = x;
  std::fill(embedding.begin(), embedding.end(), Rcpp::NumericVector::get_na());
  std::string term;
  for (int i = 0; i < x.size(); i++){
    term = Rcpp::as<std::string>(x[i]);
    auto a = voc->searchVocab(term.c_str());
    if(a >= 0){
      for (int b = 0; b < m_dim; b++) {
        embedding(i, b) = (float)(m_dsyn0[a * m_dim + b]);
      }
    }
  }
  return embedding;
}

