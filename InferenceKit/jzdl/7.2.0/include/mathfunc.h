#ifndef JZDL_MATHFUNC_H
#define JZDL_MATHFUNC_H

namespace jzdl {

double Sqrt(double A);

double Exp(double x);

int Ceil(double x);

float Abs(float x);

float Round(float x);

template<typename T>
void Swap(T a, T b)
{
  T c = a;
  a = b;
  b = c;
}

} // namespace jzdl

#endif // JZDL_MATHFUNC_H
