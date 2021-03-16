## CHANGES IN doc2vec VERSION 0.2.0

- Add top2vec semantic clustering algorithm
- Allow transfer learning in paragraph2vec by passing on a pretrained set of word vectors to initialise the word embeddings with (no initialisation of the document embeddings)

## CHANGES IN doc2vec VERSION 0.1.1

- Fixes for valgrind R CMD checks 
    - Fixes for destructors of Vocabulary
    - Remove WMD
- Added txt_count_words and removed Suggests dependency of udpipe

## CHANGES IN doc2vec VERSION 0.1.0

- Initial package based on https://github.com/hiyijian/doc2vec commit dec123e891f17ea664053ee7575b0e5e7dae4fca
