// #include "engine.hpp"
#include "nn.hpp"
#include <iostream>
#include <memory>

int main()
{
    using std::make_shared;
    // auto a = make_shared<Tensor>(2.0);
    // auto b = make_shared<Tensor>(-3.0);
    // auto c = make_shared<Tensor>(10.0);
    // auto e = a * b;
    // auto d = e + c;
    // auto f = make_shared<Tensor>(-2.0);
    // auto L = d * f;
    // std::cout << a << b << c << d << e << f << L;
    // L->backward();
    // std::cout << "Backwarded: \n";
    // std::cout << a << b << c << d << e << f << L;

    // auto a = make_shared<Tensor>(-2.0);
    // auto b = make_shared<Tensor>(3.0);
    // auto d = a * b;
    // auto e = a + b;
    // auto f = d * e;
    // f->backward();
    // std::cout << a << b << e << d << f;

    auto a = MLP(3, std::vector<int>{3, 3, 2});
    std::cout << a;
    return 0;
}