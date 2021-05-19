#ifndef MOCHIMOCHI_BINARY_OML_INTERFACE_HPP_
#define MOCHIMOCHI_BINARY_OML_INTERFACE_HPP_

#include <string>
#include <Eigen/Dense>

using namespace std;

/**
 * The BinaryOML interface declares the operations that all concrete BinaryOML must implement.
 */
class BinaryOML {
 public:
  virtual ~BinaryOML() {}
  virtual bool update(const Eigen::VectorXd& feature, const int label) = 0;
  virtual int predict(const Eigen::VectorXd& x) const = 0;
  virtual void save(const string& filename) = 0;
  virtual void load(const string& filename) = 0;
  virtual string name() const = 0;
};

#endif