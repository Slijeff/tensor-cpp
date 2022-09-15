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
    bool activate;

    ~Neuron() {}
    Neuron(int nin, bool activate = true)
        : _w{}, _b(std::make_shared<Tensor>(0.0)), activate(activate)
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
        if (activate)
        {
            return w_x_b->relu();
        }
        return w_x_b;
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
                strm << "\t\t\t" << n._w[i];
            }
            strm << "\t\t]\n";
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
    bool activate;
    ~Layer() {}
    Layer(int nin, int nout, bool activate = true)
        : _in_neu(nin), _out_neu(nout), _neurons({}), activate(activate)
    {
        init_neurons();
    }
    void init_neurons()
    {
        for (auto i = 0; i < _out_neu; i++)
        {
            _neurons.emplace_back(_in_neu, activate);
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
                strm << "\t\t" << neuron;
            }
            strm << "\t]\n";
            return strm;
        }
        return strm << "Layer(n_neurons=" << l._out_neu << ")\n";
    }
};

class MLP : public Module
{
public:
    std::vector<Layer> layers;
    ~MLP() {}
    MLP(int nin, const std::vector<int> &lay)
        : layers({})
    {
        std::vector<int> temp(lay.begin(), lay.end());
        temp.insert(temp.begin(), nin);
        for (auto i = 0; i < lay.size(); ++i)
        {
            // All layers before the last layer has activation
            layers.emplace_back(temp[i], temp[i + 1], i != (lay.size() - 1));
        }
    }

    // Do forward pass on the entire network
    std::vector<TensorPtr> operator()(std::vector<TensorPtr> x)
    {
        for (auto layer : layers)
        {
            x = layer(x);
        }
        return x;
    }

    std::vector<TensorPtr> parameters()
    {
        std::vector<TensorPtr> out;
        for (auto layer : layers)
        {
            for (auto params : layer.parameters())
            {
                out.push_back(params);
            }
        }
        return out;
    }

    friend std::ostream &operator<<(std::ostream &strm, const MLP &m)
    {
        bool debug = true;
        if (debug)
        {
            strm << "MLP(n_layers=" << m.layers.size() << " Layer=[\n";
            for (auto layer : m.layers)
            {
                strm << "\t" << layer;
            }
            strm << "]\n";
            return strm;
        }
        return strm << "MLP(n_layers=" << m.layers.size() << ")\n";
    }
};

#endif