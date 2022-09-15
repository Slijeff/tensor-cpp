#ifndef NN
#define NN

#include <vector>
#include <random>
#include <iostream>
#include "engine.hpp"

class Module
{
public:
    virtual ~Module() {}
    virtual std::vector<TensorPtr> parameters() { return {}; }
    void zero_grad()
    {
        for (auto p : parameters())
        {
            p->_grad = 0.0f;
        }
    }
};

class Neuron : public Module
{
public:
    std::vector<TensorPtr> _w;
    TensorPtr _b;

    ~Neuron() {}
    Neuron(int nin)
        : _w{}, _b(std::make_shared<Tensor>(0.0))
    {
        init_w(nin);
    }

    struct RandomGenerator
    {
        std::mt19937 _engine;
        std::uniform_real_distribution<> _dist;
        RandomGenerator()
            : _engine(std::random_device{}()), _dist{-1.0, 1.0}
        {
        }
        double operator()() { return _dist(_engine); }
    };

    void init_w(int nin)
    {
        auto gen = RandomGenerator();
        for (auto i = 0; i < nin; ++i)
        {
            double val = gen();
            _w.emplace_back(std::make_shared<Tensor>(val));
        }
    }

    // Do forward pass on each Tensor
    TensorPtr operator()(std::vector<TensorPtr> x)
    {
        auto w_x = x[0] * _w[0];
        for (auto i = 0; i < x.size(); ++i)
        {
            w_x = w_x + (x[i] * _w[i]);
        }
        auto w_x_b = w_x + _b;
        return w_x_b->relu();
    }

    std::vector<TensorPtr> parameters()
    {
        auto temp = _w;
        temp.push_back(_b);
        return temp;
    }

    friend std::ostream &operator<<(std::ostream &strm, const Neuron &n)
    {
        bool debug = true;
        if (debug)
        {
            strm << "Neuron(n_weights=" << n._w.size() << " tensors=[\n";
            for (auto i = 0; i < n._w.size(); ++i)
            {
                strm << "\t\t" << n._w[i];
            }
            strm << "\t]\n";
            return strm;
        }
        return strm << "Neuron(n_weights=" << n._w.size() << ")\n";
    }
};

class Layer : public Module
{
public:
    int _in_neu;
    int _out_neu;
    std::vector<Neuron> _neurons;
    ~Layer() {}
    Layer(int nin, int nout)
        : _in_neu(nin), _out_neu(nout), _neurons({})
    {
        init_neurons();
    }
    void init_neurons()
    {
        for (auto i = 0; i < _out_neu; i++)
        {
            _neurons.emplace_back(_in_neu);
        }
    }

    // Do forward pass on each neuron
    std::vector<TensorPtr> operator()(std::vector<TensorPtr> x)
    {
        auto out = std::vector<TensorPtr>{};
        for (auto neuron : _neurons)
        {
            out.emplace_back(neuron(x));
        }
        return out;
    }

    std::vector<TensorPtr> parameters()
    {
        std::vector<TensorPtr> out;
        for (auto neuron : _neurons)
        {
            for (auto tensor : neuron.parameters())
            {
                out.push_back(tensor);
            }
        }
        return out;
    }

    friend std::ostream &operator<<(std::ostream &strm, const Layer &l)
    {
        bool debug = true;
        if (debug)
        {
            strm << "Layer(n_neurons=" << l._out_neu << " Neuron=[\n";
            for (auto neuron : l._neurons)
            {
                strm << "\t" << neuron;
            }
            strm << "]\n";
            return strm;
        }
        return strm << "Layer(n_neurons=" << l._out_neu << ")\n";
    }
};
#endif