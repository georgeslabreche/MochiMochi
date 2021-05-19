/**
 * Implement the Factory Design Pattern to instanciate Online ML algorithm objects.
 * https://refactoring.guru/design-patterns/factory-method/cpp/example
 */

#ifndef MOCHIMOCHI_BINARY_OML_FACTORY_HPP_
#define MOCHIMOCHI_BINARY_OML_FACTORY_HPP_

#include <string>
#include <iostream>
#include <Eigen/Dense>

#include "binary_oml.hpp"
#include "../binary/adam.hpp"
#include "../binary/arow.hpp"

using namespace std;

/**
 * The Creator class declares the factory method that is supposed to return an
 * object of a Product class. The Creator's subclasses usually provide the
 * implementation of this method.
 */

class BinaryOMLCreator {
  /**
   * Note that the Creator may also provide some default implementation of the
   * factory method.
   */
protected:
  BinaryOML* m_pBinaryOML;

  /* The constructor. */
  BinaryOMLCreator(BinaryOML* pBinaryOML)
  {
    m_pBinaryOML = pBinaryOML;
  }

public:
  virtual ~BinaryOMLCreator() {
    delete m_pBinaryOML;
  };
  
  BinaryOML* FactoryMethod()
  {
    return m_pBinaryOML;
  };
  /**
   * Despite its name, the Creator's primary responsibility isn't to create
   * BinaryOML. Usually, it contains some core business logic that relies
   * on BinaryOML objects, returned by the factory method. Subclasses can
   * indirectly change that business logic by overriding the factory method
   * and returning a different type of BinaryOML from it.
   */
  string name()
  {
    return m_pBinaryOML->name();
  }

  void train()
  {
    std::cout << "TRAIN: " << m_pBinaryOML->name() << std::endl;
  }

  void trainAndSave(const char* modelFilePath)
  {
    std::cout << "TRAIN AND SAVE: " << modelFilePath << std::endl;
    m_pBinaryOML->save(modelFilePath);
  }

  void infer()
  {
    std::cout << "INFER: " << m_pBinaryOML->name() << std::endl;
  }

  void load(const char* modelFilePath)
  {
    std::cout << "LOAD: " << modelFilePath << std::endl;
  }

  void save(const char* modelFilePath)
  {
    std::cout << "SAVE: " << modelFilePath << std::endl;
    m_pBinaryOML->save(modelFilePath);
  }
};

/**
 * Concrete Creators override the factory method in order to change the
 * resulting BinaryOML's type.
 */
class BinaryADAGRADRDACreator : public BinaryOMLCreator {
  /**
   * Note that the signature of the method still uses the abstract product type,
   * even though the concrete product is actually returned from the method. This
   * way the Creator can stay independent of concrete product classes.
   */

public:
  virtual ~BinaryADAGRADRDACreator() {}

  /* The BinaryOML creator for ADAGRAD RDA. */
  BinaryADAGRADRDACreator(const size_t dim, const double eta, const double lambda)
    : BinaryOMLCreator(new ADAGRAD_RDA(dim, eta, lambda)) { }
};


/** 
 * Concrete Creator for ADAGRAD RDA.
 */
class BinaryADAMCreator : public BinaryOMLCreator {
  
public:
  virtual ~BinaryADAMCreator() {};

  /* The BinaryOML creator for ADAM. */
  BinaryADAMCreator(const size_t dim)
    : BinaryOMLCreator(new ADAM(dim)) { }
};

/** 
 * Concrete Creator for AROW.
 */
class BinaryAROWCreator : public BinaryOMLCreator {

public:
  virtual ~BinaryAROWCreator() {}

  /* The BinaryOML creator for AROW. */
  BinaryAROWCreator(const size_t dim, const double r)
    : BinaryOMLCreator(new AROW(dim, r)) { }
};

/**
 * Concrete Creator for NHERD.
 */
class BinaryNHERDCreator : public BinaryOMLCreator {

public:
  virtual ~BinaryNHERDCreator() {}

  /* The BinaryOML creator for NHERD. */
  BinaryNHERDCreator(const size_t dim, const double c, const int diagonal)
    : BinaryOMLCreator(new NHERD(dim, c, diagonal)) { }
};

/**
 * Concrete Creator for PA.
 */
class BinaryPACreator : public BinaryOMLCreator {

public:
  virtual ~BinaryPACreator() {}

  /* The BinaryOML creator for PA. */
  BinaryPACreator(const size_t dim, const double c, const int select)
    : BinaryOMLCreator(new PA(dim, c, select)) { }
};

/**
 * Concrete Creator for SCW.
 */
class BinarySCWCreator : public BinaryOMLCreator {

public:
  virtual ~BinarySCWCreator() {}

  /* The BinaryOML creator for SCW. */
  BinarySCWCreator(const size_t dim, const double c, const double eta)
    : BinaryOMLCreator(new PA(dim, c, eta)) { }
};

#endif // MOCHIMOCHI_BINARY_OML_FACTORY_HPP_