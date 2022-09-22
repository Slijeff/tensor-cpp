#ifndef ENGINE
#define ENGINE

#include <vector>
#include <iostream>
#include <functional>
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>

class Tensor;
typedef std::shared_ptr<Tensor> TensorPtr;
class Tensor : public std::enable_shared_from_this<Tensor>
{
public:
    double data;
    std::vector<TensorPtr> prev;
    std::function<void()> _backward;
    double _grad;
    bool visited;

    Tensor(double data, std::vector<TensorPtr> _children = {})
        : data(data), prev(_children), _backward([]() {}), _grad(0.0f), visited(false)
    {
    }

    void backward()
    {
        _grad = 1.0;
        auto sorted_tensors = build_topological_order();
        for (int i = sorted_tensors.size() - 1; i >= 0; i--)
        {
            sorted_tensors[i]->_backward();
        }
    }
    std::vector<TensorPtr> build_topological_order()
    {
        std::vector<TensorPtr> topo;
        topo_sort(topo);
        clear_visited(topo);
        return topo;
    }
    void topo_sort(std::vector<TensorPtr> &order)
    {
        topo_sort(shared_from_this(), order);
    }
    void topo_sort(TensorPtr tensor, std::vector<TensorPtr> &order)
    {
        if (!tensor->visited)
        {
            tensor->visited = true;
            for (auto child : tensor->prev)
            {
                topo_sort(child, order);
            }
            order.push_back(tensor);
        }
    }
    void clear_visited(std::vector<TensorPtr> &order)
    {
        for (auto tensor : order)
        {
            tensor->visited = false;
        }
    }

    // ReLu activation
    TensorPtr relu()
    {
        auto out = std::make_shared<Tensor>(std::max(0.0, data), std::vector<TensorPtr>{shared_from_this()});
        out->_backward = [out, self = shared_from_this()]()
        {
            self->_grad += (out->data > 0) * out->_grad;
        };
        return out;
    }
    // Tensor^Tensor
    TensorPtr pow(TensorPtr exp)
    {
        auto out = std::make_shared<Tensor>(std::pow(data, exp->data), std::vector<TensorPtr>{shared_from_this()});
        out->_backward = [out, exp, self = shared_from_this()]()
        {
            self->_grad += exp->data * std::pow(self->data, exp->data - 1) * out->_grad;
        };
        return out;
    }
    // Tensor^double
    TensorPtr pow(double exp)
    {
        auto new_exp = std::make_shared<Tensor>(exp);
        return pow(new_exp);
    }

    // Tensor + Tensor
    friend TensorPtr operator+(TensorPtr lhs, TensorPtr rhs)
    {
        auto out = std::make_shared<Tensor>(lhs->data + rhs->data, std::vector<TensorPtr>{lhs, rhs});
        out->_backward = [out, lhs, rhs]()
        {
            lhs->_grad += out->_grad;
            rhs->_grad += out->_grad;
        };
        return out;
    }
    // Tensor + double
    friend TensorPtr operator+(TensorPtr lhs, double rhs)
    {
        auto new_rhs = std::make_shared<Tensor>(rhs);
        return lhs + new_rhs;
    }
    // doubel + Tensor
    friend TensorPtr operator+(double lhs, TensorPtr rhs)
    {
        return rhs + lhs;
    }

    // Tensor * Tensor
    friend TensorPtr operator*(TensorPtr lhs, TensorPtr rhs)
    {
        auto out = std::make_shared<Tensor>(lhs->data * rhs->data, std::vector<TensorPtr>{lhs, rhs});
        out->_backward = [out, lhs, rhs]()
        {
            lhs->_grad += rhs->data * out->_grad;
            rhs->_grad += lhs->data * out->_grad;
        };
        return out;
    }
    // Tensor * double
    friend TensorPtr operator*(TensorPtr lhs, double rhs)
    {
        auto new_rhs = std::make_shared<Tensor>(rhs);
        return lhs * new_rhs;
    }
    // double * Tensor
    friend TensorPtr operator*(double lhs, TensorPtr rhs)
    {
        return rhs * lhs;
    }

    // -Tesor
    friend TensorPtr operator-(TensorPtr rhs)
    {
        return rhs * std::make_shared<Tensor>(-1.0);
    }
    // Tensor - Tensor
    friend TensorPtr operator-(TensorPtr lhs, TensorPtr rhs)
    {
        return lhs + (-rhs);
    }
    // Tensor - double
    friend TensorPtr operator-(TensorPtr lhs, double rhs)
    {
        return lhs + (-rhs);
    }
    // double - Tensor
    friend TensorPtr operator-(double lhs, TensorPtr rhs)
    {
        return lhs + (-rhs);
    }

    // Tensor / Tensor
    friend TensorPtr operator/(TensorPtr lhs, TensorPtr rhs)
    {
        return lhs * rhs->pow(-1.0);
    }
    // double / Tensor
    friend TensorPtr operator/(double lhs, TensorPtr rhs)
    {
        auto new_lhs = std::make_shared<Tensor>(lhs);
        return new_lhs / rhs;
    }
    // Tensor / double
    friend TensorPtr operator/(TensorPtr lhs, double rhs)
    {
        auto new_rhs = std::make_shared<Tensor>(rhs);
        return lhs / new_rhs;
    }

    friend std::ostream &operator<<(std::ostream &strm, const TensorPtr &t)
    {
        return strm << "Tensor(data=" << t->data << ", grad=" << t->_grad << ")\n";
    }
};

#endif