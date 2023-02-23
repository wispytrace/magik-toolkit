/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : mat.h
 * Authors     : klyu(kanglong.yu@ingenic.com)
 * Create Time : 2019-12-26:10:00:00
 * Description :
 *
 */

#ifndef JZDL_MAT_H
#define JZDL_MAT_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <time.h>

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define JZDL_API __attribute__((dllexport))
#else
#define JZDL_API __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define JZDL_API __attribute__((dllimport))
#else
#define JZDL_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define JZDL_API __attribute__((visibility("default")))
#else
#define JZDL_API
#endif
#endif

#ifndef JZ_MXU
#define JZ_MXU 1
#endif
#if JZ_MXU
#include <mxu2.h>
#include <jzmxu128.h>
#endif

#define LOG_INFO 0
#if LOG_INFO
#include <time.h>
#include <iostream>
#endif // LOG_INFO

#ifndef INTTYPE_H
#define INTTYPE_H
typedef unsigned char uint8_t;
typedef int64_t int64;
typedef uint8_t quint8;
typedef int int32;
#endif

namespace jzdl {

#if LOG_INFO
struct MemRecord {
    float total_alloc;
    float now_usage;
    float total_free;
    MemRecord() {
        total_alloc = 0;
        now_usage = 0;
        total_free = 0;
    }
    void increase(float size) {
        total_alloc += size;
        now_usage += size;
    }
    void decrease(float size) {
        total_free += size;
        now_usage -= size;
    }
    void print() {
        printf("Memory summary:----> total malloc:%gKB\n"
               "               ----> now usage: %gKB\n"
               "               ----> total freed: %gKB\n",total_alloc/1024, now_usage/1024, total_free/1024);
    }
    static MemRecord* instance;
    static MemRecord* get_instance() {
        if (instance == nullptr) {
            instance = new MemRecord();
        }
        return instance;
    }
};
#endif // LOG_INFO

// the three dimension matrix
template<typename T>
class JZDL_API Mat
{
public:
    // empty
    Mat();
    // vec
    Mat(int w);
    // image
    Mat(int w, int h);
    // dim
    Mat(int w, int h, int c);
    // copy
    Mat(const Mat<T>& m);
    // external vec
    Mat(int w, T* data);
    // external image
    Mat(int w, int h, T* data);
    // external dim
    Mat(int w, int h, int c, T* data);
    // release
    ~Mat();
    // assign
    Mat<T>& operator=(const Mat<T>& m);
    // set all
    void fill(T v);
    // deep copy
    Mat<T> clone() const;
    // reshape vec
    Mat<T> reshape(int w) const;
    // reshape image
    Mat<T> reshape(int w, int h) const;
    // reshape dim
    Mat<T> reshape(int w, int h, int c) const;
    // allocate vec
    void create(int w);
    // allocate image
    void create(int w, int h);
    // allocate dim
    void create(int w, int h, int c);
    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // data reference
    T* row(int y);
    const T* row(int y) const;
    operator T*();
    operator const T*() const;

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
   // void substract_mean_normalize(const T* mean_vals, const T* norm_vals);

    // the dimensionality
    int dims;
    // pointer to the data
    T* data;

    // pointer to the reference counter;
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    int w;
    int h;
    int c;

    size_t step;
};

// the alignment of all the allocated buffers
#define MALLOC_ALIGN    16
// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

// exchange-add operation for atomic operations on reference counters
#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define JZDL_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define JZDL_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define JZDL_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define JZDL_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define JZDL_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define JZDL_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
static inline void JZDL_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif

template<typename T>
inline Mat<T>::Mat()
    : dims(0), data(0), refcount(0), w(0), h(0), c(0), step(0)
{
}

template<typename T>
inline Mat<T>::Mat(int _w)
    : dims(0), data(0), refcount(0)
{
    create(_w);
}

template<typename T>
inline Mat<T>::Mat(int _w, int _h)
    : dims(0), data(0), refcount(0)
{
    create(_w, _h);
}

template<typename T>
inline Mat<T>::Mat(int _w, int _h, int _c)
    : dims(0), data(0), refcount(0)
{
    create(_w, _h, _c);
}

template<typename T>
inline Mat<T>::Mat(const Mat<T>& m)
    : dims(m.dims), data(m.data), refcount(m.refcount)
{
    if (refcount)
        JZDL_XADD(refcount, 1);

    w = m.w;
    h = m.h;
    c = m.c;

    step = m.step;
}

template<typename T>
inline Mat<T>::Mat(int _w, T* _data)
    : dims(1), data(_data), refcount(0)
{
    w = _w;
    h = 1;
    c = 1;

    step = c * w;
}

template<typename T>
inline Mat<T>::Mat(int _w, int _h, T* _data)
    : dims(2), data(_data), refcount(0)
{
    w = _w;
    h = _h;
    c = 1;

    step = c * w;
}

template<typename T>
inline Mat<T>::Mat(int _w, int _h, int _c, T* _data)
    : dims(3), data(_data), refcount(0)
{
    w = _w;
    h = _h;
    c = _c;

    step = c * w;
}

template<typename T>
inline Mat<T>::~Mat()
{
    release();
}

template<typename T>
inline Mat<T>& Mat<T>::operator=(const Mat<T>& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        JZDL_XADD(m.refcount, 1);

    release();

    dims = m.dims;
    data = m.data;
    refcount = m.refcount;

    w = m.w;
    h = m.h;
    c = m.c;

    step = m.step;

    return *this;
}

template<typename T>
inline void Mat<T>::fill(T _v)
{
    size_t _total = total();
    for (size_t i = 0; i < _total; i++)
    {
        data[i] = _v;
    }
}

template<typename T>
inline Mat<T> Mat<T>::clone() const
{
    if (empty())
        return Mat();

    Mat<T> m;
    if (dims == 1)
        m.create(w);
    else if (dims == 2)
        m.create(w, h);
    else if (dims == 3)
        m.create(w, h, c);

    if (total() > 0)
    {
        memcpy(m.data, data, total() * sizeof(T));
    }

    return m;
}

template<typename T>
inline Mat<T> Mat<T>::reshape(int _w) const
{
    Mat<T> m = *this;

    m.dims = 1;

    m.w = _w;
    m.h = 1;
    m.c = 1;

    m.step = _w;

    return m;
}

template<typename T>
inline Mat<T> Mat<T>::reshape(int _w, int _h) const
{
    Mat<T> m = *this;

    m.dims = 2;

    m.w = _w;
    m.h = _h;
    m.c = 1;

    m.step = _w;

    return m;
}

template<typename T>
inline Mat<T> Mat<T>::reshape(int _w, int _h, int _c) const
{
    Mat<T> m = *this;

    m.dims = 3;

    m.w = _w;
    m.h = _h;
    m.c = _c;

    m.step = _c * _w;

    return m;
}

template<typename T>
inline void Mat<T>::create(int _w)
{
    release();

    dims = 1;

    w = _w;
    h = 1;
    c = 1;

    step = w;
#if LOG_INFO
    MemRecord::get_instance()->increase(total()*sizeof(T));
#endif
    if (total() > 0)
    {
        size_t totalsize = alignSize(step * h * sizeof(T), 16);
        data = (T*)fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

template<typename T>
inline void Mat<T>::create(int _w, int _h)
{
    release();

    dims = 2;

    w = _w;
    h = _h;
    c = 1;

    step = w;
#if LOG_INFO
    MemRecord::get_instance()->increase(total()*sizeof(T));
#endif
    if (total() > 0)
    {
        size_t totalsize = alignSize(step * h * sizeof(T), 16);
        data = (T*)fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

template<typename T>
inline void Mat<T>::create(int _w, int _h, int _c)
{
    release();

    dims = 3;

    w = _w;
    h = _h;
    c = _c;

    step = w * c;
#if LOG_INFO
    MemRecord::get_instance()->increase(total()*sizeof(T));
#endif
    if (total() > 0)
    {
        size_t totalsize = alignSize(step * h * sizeof(T), 16);

        data = (T*)fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

template<typename T>
inline void Mat<T>::addref()
{
    if (refcount)
        JZDL_XADD(refcount, 1);
}

template<typename T>
inline void Mat<T>::release()
{
    if (refcount && JZDL_XADD(refcount, -1) == 1) {
        fastFree(data);
#if LOG_INFO
        MemRecord::get_instance()->decrease(total()*sizeof(T));
#endif
    }

    dims = 0;
    data = 0;

    w = 0;
    h = 0;
    c = 0;

    step = 0;

    refcount = 0;
}

template<typename T>
inline bool Mat<T>::empty() const
{
    return data == 0 || total() == 0;
}

template<typename T>
inline size_t Mat<T>::total() const
{
    return step * h;
}

template<typename T>
inline T* Mat<T>::row(int y)
{
    return data + step * y;
}

template<typename T>
inline const T* Mat<T>::row(int y) const
{
    return data + step * y;
}

template<typename T>
inline Mat<T>::operator T*()
{
    return data;
}

template<typename T>
inline Mat<T>::operator const T*() const
{
    return data;
}



}

#endif
