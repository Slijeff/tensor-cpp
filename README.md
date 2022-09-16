## Tensor++

Inspired by micrograd, a tiny library with 2 files that supports automatic differentiation.

Example:

```cpp
#include "engine.hpp"

auto a = std::make_shared<Tensor>(-4.0f);
auto b = std::make_shared<Tensor>(2.0f);
auto c = a + b;
auto d = a * b + b->pow(3);
c = c + c + 1;
c = c + 1 + c + (-a);
d = d + d * 2 + (b + a)->relu();
d = d + 3 * d + (b - a)->relu();
auto e = c - d;
auto f = e * e;
auto g = f / 2.0;
g = g + 10.0 / f;

std::cout << g->data; // prints 24.7041, the outcome of this forward pass
g->backward();
std::cout << a->_grad; // prints 138.8338, i.e. the numerical value of dg/da
std::cout << b->_grad; // prints 645.5773, i.e. the numerical value of dg/db
```
