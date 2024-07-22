/*
 * Author: Maksym Zhelyeznyakov
 *
 * email: mzhelyez@gmail.com
 *
 */
#include <torch/extension.h>
#include <vector>
#include <torch/special.h>
#include <c10/util/irange.h>

#define PI 3.14159265358979311599796346854
using namespace torch::indexing;

torch::Tensor dht(torch::Tensor f, torch::Tensor radii)
{
  /*
   *
   *
   *
   *
   */
  auto N = f.size(0);
  auto F = at::zeros_like(f);
  auto R = torch::max(radii);
  
  auto f1n = f.index({Slice(1,None)});
  auto k = at::arange(0,N);
  auto k1n = k.index({Slice(1,None)});
  
  F[0] = PI * R*R / (4 * R*R) * f[0] +
    2 * PI * R*R/(N*N) * at::sum(f1n*k1n);

  auto l = at::arange(0,N);
  for (const auto i : c10::irange(1,N))
    {
      F[i] = ( R*R * f[0] / (l[i] * N) ) * torch::special::bessel_j1(PI*l[i] / N) +
	2 * PI * R * R / (N * N) *
	at::sum(f1n * k1n * torch::special::bessel_j0(PI * k1n * l[i] / N ) );
    }
  
  return F;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dht", &dht, "dht");
}
