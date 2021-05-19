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
#include "../binary/adagrad_rda.hpp"
#include "../binary/adam.hpp"
#include "../binary/arow.hpp"
#include "../binary/nherd.hpp"
#include "../binary/pa.hpp"
#include "../binary/scw.hpp"

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

  /**
   * Train the model.
   */
  void train(string *pInput, int dim)
  {
    /* Convert training string input into training vector. */
    auto data = utility::read_ones<int>(*pInput, dim);

    /* Update the model with the new training data. */
    m_pBinaryOML->update(data.second, data.first);
  }

  /**
   * Train and save the model.
   */
  void trainAndSave(string *pInput, size_t dim, const string modelFilePath)
  {
    /* Convert training string input into data vector. */
    auto data = utility::read_ones<int>(*pInput, dim);

    /* Update the model with the new training data. */
    m_pBinaryOML->update(data.second, data.first);

    /* Serialize the model. */
    m_pBinaryOML->save(modelFilePath);
  }

  /**
   * Infer/predicut the label of the given data input
   */
  int infer(string *pInput, size_t dim)
  {
    /* Convert inference string input into data vector. */
    auto data = utility::read_ones<int>(*pInput, dim);

    /* Invoke prediction method and return result. */
    return m_pBinaryOML->predict(data.second);
  }

  /**
   * Load the model.
   */
  void load(const string modelFilePath)
  {
    m_pBinaryOML->load(modelFilePath);
  }

  /**
   * Save the model.
   */
  void save(const string modelFilePath)
  {
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