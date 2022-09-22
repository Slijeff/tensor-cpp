#include "nn.hpp"
#include <vector>
#include <memory>
#include <numeric>

int main()
{
    using std::cout;
    using std::make_shared;
    using std::vector;

    // xor classification
    vector<vector<double>> tx = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}};
    vector<double> ty = {1.0, -1.0, -1.0, 1.0};

    vector<vector<TensorPtr>> xs;
    vector<TensorPtr> ys;
    for (auto x : tx)
    {
        xs.push_back(vector<TensorPtr>{
            make_shared<Tensor>(x[0]),
            make_shared<Tensor>(x[1]),
            make_shared<Tensor>(x[2]),
        });
    }

    ys = {
        make_shared<Tensor>(ty[0]),
        make_shared<Tensor>(ty[1]),
        make_shared<Tensor>(ty[2]),
        make_shared<Tensor>(ty[3]),
    };

    auto network = MLP(3, vector<int>{4, 4, 1});

    for (auto epoch = 0; epoch < 200; epoch++)
    {
        vector<vector<TensorPtr>> ypred;
        // forward pass
        for (auto x : xs)
        {
            ypred.emplace_back(network(x));
        }
        // calculate loss
        auto loss = make_shared<Tensor>(0.0);
        for (size_t i = 0; i < ys.size(); i++)
        {
            loss = loss + (ypred[i][0] - ys[i])->pow(2);
        }
        // zero grad
        network.zero_grad();
        // backward propagation
        loss->backward();

        // update network
        for (auto p : network.parameters())
        {
            p->data += -0.01 * p->_grad;
        }

        if (epoch % 10 == 0)
        {
            cout << "epoch: " << epoch + 1 << " loss: " << loss->data << "\n";
            cout << "predictions: ";
            for (size_t i = 0; i < ypred.size(); i++)
            {
                cout << ypred[i][0]->data << " ";
            }
        }
    }
    return 0;
}