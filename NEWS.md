## CHANGES IN doc2vec VERSION 0.2.2

- Fix DOI in DESCRIPTION
- Remove C++11 from Makevars

## CHANGES IN doc2vec VERSION 0.2.1

- Make sure words are only 100 characters when getting embeddings of documents (issue #20)
- Limit documents to 1000 words by explicitely keeping only the first 1000 words per document + provide warning if doc_id contains spaces

## CHANGES IN doc2vec VERSION 0.2.0

- Add top2vec semantic clustering algorithm
- Allow transfer learning in paragraph2vec by passing on a pretrained set of word vectors to initialise the word embeddings with (no initialisation of the document embeddings)
- In paragraph2vec: close opened files directly after training instead of waiting for R garbage collection to kick in
- Added dataset 'be_parliament_2020' with questions asked by members in the Belgium Federal parliament in 2020

## CHANGES IN doc2vec VERSION 0.1.1

- Fixes for valgrind R CMD checks 
    - Fixes for destructors of Vocabulary
    - Remove WMD
- Added txt_count_words and removed Suggests dependency of udpipe

## CHANGES IN doc2vec VERSION 0.1.0

- Initial package based on https://github.com/hiyijian/doc2vec commit dec123e891f17ea664053ee7575b0e5e7dae4fca
