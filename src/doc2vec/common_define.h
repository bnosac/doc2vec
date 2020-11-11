#ifndef COMMON_DEFINE_H
#define COMMON_DEFINE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <limits>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_DOC2VEC_KNN 2000
const int vocab_hash_size = 30000000;
const int negtive_sample_table_size = 1e8;

typedef float real;

#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#define MIN(a,b) ( ((a)>(b)) ? (b):(a) )




#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>
#else
#include <stdlib.h>
static inline void *_aligned_malloc(size_t size, size_t alignment)
{
#if defined(__sun) || defined(sun)
  return memalign(alignment, size);
#else
  void *p;
  int ret = posix_memalign(&p, alignment, size);
  return (ret == 0) ? p : 0;
#endif
}
static inline void _aligned_free(void *p)
{
  free(p);
}
#endif

#endif
