#ifndef MOCHIMOCHI_ADAGRAD_RDA_HPP_
#define MOCHIMOCHI_ADAGRAD_RDA_HPP_

#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include "../../functions/enumerate.hpp"
#include "../factory/binary_oml.hpp"

class ADAGRAD_RDA : public BinaryOML {
private :
  const std::size_t kDim;
  const double kEta;
  const double kLambda;

private :
  std::size_t _timestep;
  Eigen::VectorXd _w;
  Eigen::VectorXd _h;
  Eigen::VectorXd _g;

public :
  ADAGRAD_RDA(const std::size_t dim, const double eta, const double lambda)
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

private :

  double calculate_margin(const Eigen::VectorXd& x) const {
    return _w.dot(x);
  }

  double suffer_loss(const Eigen::VectorXd& x, const int y) const {
    return std::max(0.0, 1.0 - y * _w.dot(x));
  }

public :

  std::string name() const override {
    return std::string("ADAGRAD_RDA");
  }

  bool update(const Eigen::VectorXd& feature, const int label) {
    if (suffer_loss(feature, label) <= 0.0) { return false; }

    _timestep++;
    functions::enumerate(feature.data(), feature.data() + feature.size(), 0,
                       [&](const int index, const double value) {
                         const auto gradiant = -label * value;
                         _g[index] += gradiant;
                         _h[index] += gradiant * gradiant;

                         const auto sign = _g[index] >= 0 ? 1 : -1;
                         const auto eta = kEta / std::sqrt(_h[index]);
                         const auto u = std::abs(_g[index]) / _timestep;

                         _w[index] = (u <= kLambda) ? 0.0 : -sign * eta * _timestep * (u - kLambda);
                       });
    return true;
  }

  int predict(const Eigen::VectorXd& x) const {
    return calculate_margin(x) > 0.0 ? 1 : -1;
  }

  void save(const std::string& filename) {
    std::ofstream ofs(filename);
    assert(ofs);
    boost::archive::text_oarchive oa(ofs);
    oa << *this;
    ofs.close();
  }

  void load(const std::string& filename) {
    std::ifstream ifs(filename);
    assert(ifs);
    boost::archive::text_iarchive ia(ifs);
    ia >> *this;
    ifs.close();
  }

private :
  friend class boost::serialization::access;
  BOOST_SERIALIZATION_SPLIT_MEMBER();
  template <class Archive>
  void save(Archive& ar, const unsigned int version) const {
    std::vector<double> w_vector(_w.data(), _w.data() + _w.size());
    std::vector<double> h_vector(_h.data(), _h.data() + _h.size());
    std::vector<double> g_vector(_g.data(), _g.data() + _g.size());

    ar & boost::serialization::make_nvp("w", w_vector);
    ar & boost::serialization::make_nvp("h", h_vector);
    ar & boost::serialization::make_nvp("g", g_vector);
    ar & boost::serialization::make_nvp("dimension", const_cast<std::size_t&>(kDim));
    ar & boost::serialization::make_nvp("eta", const_cast<double&>(kEta));
    ar & boost::serialization::make_nvp("lambda", const_cast<double&>(kLambda));
    ar & boost::serialization::make_nvp("timestep", _timestep);
  }

  template <class Archive>
  void load(Archive& ar, const unsigned int version) {
    std::vector<double> w_vector;
    std::vector<double> h_vector;
    std::vector<double> g_vector;

    ar & boost::serialization::make_nvp("w", w_vector);
    ar & boost::serialization::make_nvp("h", h_vector);
    ar & boost::serialization::make_nvp("g", g_vector);
    ar & boost::serialization::make_nvp("dimension", const_cast<std::size_t&>(kDim));
    ar & boost::serialization::make_nvp("eta", const_cast<double&>(kEta));
    ar & boost::serialization::make_nvp("lambda", const_cast<double&>(kLambda));
    ar & boost::serialization::make_nvp("timestep", _timestep);

    _w = Eigen::Map<Eigen::VectorXd>(&w_vector[0], w_vector.size());
    _h = Eigen::Map<Eigen::VectorXd>(&h_vector[0], h_vector.size());
    _g = Eigen::Map<Eigen::VectorXd>(&g_vector[0], g_vector.size());
  }

};

#endif //MOCHIMOCHI_ADAGRAD_RDA_HPP_
