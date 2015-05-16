#ifndef SRC_ONLINE_ADAGRAD_RDA_HPP_
#define SRC_ONLINE_ADAGRAD_RDA_HPP_

#include <Eigen/Dense>
#include <cmath>
#include "utility.hpp"

class ADAGRAD_RDA {
private :
  const int kDim;
  const double kEta;
  const double kLambda;

private :
  std::size_t _timestep;
  Eigen::VectorXd _w;
  Eigen::VectorXd _h;
  Eigen::VectorXd _g;

public :
  ADAGRAD_RDA(const int dim, const double eta, const double lambda)
    : kDim(dim),
      kEta(eta),
      kLambda(lambda),
      _timestep(0),
      _w(Eigen::VectorXd::Zero(kDim)),
      _h(Eigen::VectorXd::Zero(kDim)),
      _g(Eigen::VectorXd::Zero(kDim)) {
    static_assert(std::numeric_limits<decltype(dim)>::max() > 0, "Dimension Error. (Dimension > 0)");
    static_assert(std::numeric_limits<decltype(eta)>::max() > 0, "Hyper Parameter Error. (eta > 0)");
    static_assert(std::numeric_limits<decltype(lambda)>::max() > 0, "Hyper Parameter Error. (lambda > 0)");
    assert(dim > 0);
    assert(eta > 0);
    assert(lambda > 0);
  }

  virtual ~ADAGRAD_RDA() { }

  double calculate_margin(const Eigen::VectorXd& x) const {
    return _w.dot(x);
  }

  double suffer_loss(const Eigen::VectorXd& x, const int y) const {
    return std::max(0.0, 1.0 - y * _w.dot(x));
  }

  void update(const Eigen::VectorXd& feature, const int label) {
    if (suffer_loss(feature, label) <= 0.0) { return ; }

    _timestep++;
    utility::enumerate(feature.data(), feature.data() + feature.size(), 0,
                       [&](const int index, const double value) {
                         const auto gradiant = -label * value;
                         _g[index] += gradiant;
                         _h[index] += gradiant * gradiant;

                         const int sign = _g[index] >= 0 ? 1 : -1;
                         const double eta = kEta / std::sqrt(_h[index]);
                         const double u = std::abs(_g[index]) / _timestep;

                         if (u <= kLambda) {
                           _w[index] = 0.0;
                         } else {
                           _w[index] = -sign * eta * _timestep * (u - kLambda);
                         }
                       });

  }

  int predict(const Eigen::VectorXd& x) const {
    return calculate_margin(x) > 0.0 ? 1 : -1;
  }

};

#endif //SRC_ONLINE_ADAGRAD_RDA_HPP_
