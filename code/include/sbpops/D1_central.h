namespace sbp {
  /**
  * Central first derivative SBP operator. The class holds the stencil weights for the interior
  * stencil and the closure stencils and contains method for applying the operator to a grid function
  * vector. The stencils of the operator are all declared at compile time, which (hopefully) should
  * allow the compiler to perform extensive optimization on the apply methods.
  **/
  template <int interior_width, int n_closures, int closure_width>
  class D1_central{
  private:
    // Stencils defining the operator. The stencils are declared at compile time
    const double interior_stencil[interior_width];
    const double closure_stencils[n_closures][closure_width];
  public:
    // Constructor. See implementations for specific stencils
    constexpr D1_central();
    // TODO: Figure out how to pass static arrays holding the stencils at construction.
    // Then could have a single templated constructor initializing the members from the arguments
    // and the specific operators (of a given order) would be nicely defined in 
    // make_diff_ops.h.
    // constexpr D1_central(const double (*i_s)[interior_width], const double (**c_s)[n_closures][closure_width]);
    /**
    * Computes v_x[i] for an index i in the set of the left closure points, with inverse grid spacing hi.
    **/
    inline double apply_left(const double *v, const double hi, const int i) const;
    /**
    * Computes v_x[i] for an index i in the set of the interior points, with inverse grid spacing hi.
    **/
    inline double apply_interior(const double *v, const double hi, const int i) const;
    /**
    * Computes v_x[i] for an index i in the set of the right closure points, with inverse grid spacing hi,
    * and grid function vector size n.
    **/
    inline double apply_right(const double *v, const double hi, const int n, const int i) const;
    /**
    * Computes v_x with inverse grid spacing hi for a grid function vector v, of size n.
    **/
    inline void apply(const double *v, const double hi, const int n, double *v_x) const;
  };

  //=============================================================================
  // Implementations
  //=============================================================================

  // TODO: Consider adding bounds checking (at least checking in debug mode).
  template <int iw, int nc, int closure_width>
  inline double D1_central<iw,nc,closure_width>::apply_left(const double *v, const double hi, const int i) const
  {
    double u = 0;
    for (int j = 0; j<closure_width; j++)
    {
      u += closure_stencils[i][j]*v[j];
    }
    return hi*u;
  };

  template <int interior_width, int nc, int cw>
  inline double D1_central<interior_width, nc, cw>::apply_interior(const double *v, const double hi, const int i) const
  {
    double u = 0;
    for (int j = 0; j<interior_width; j++)
    {
      u += interior_stencil[j]*v[i-(interior_width-1)/2+j];
    }
    return hi*u;
  };

  template <int iw, int nc, int closure_width>
  inline double D1_central<iw,nc,closure_width>::apply_right(const double *v, const double hi, const int n, const int i) const
  {
    double u = 0;
    for (int j = 0; j < closure_width; j++)
    {
      u -= closure_stencils[n-i-1][closure_width-j-1]*v[n-closure_width+j];
    }
    return hi*u;
  };

  template <int iw, int nc, int cw>
  inline void D1_central<iw,nc,cw>::apply(const double *v, const double hi, const int n, double *v_x) const
  {
    for (int i = 0; i < nc; i++){
      v_x[i] = apply_left(v,hi,i);
    }
    for (int i = nc; i < n-nc; i++){
      v_x[i] = apply_interior(v,hi,i);
    }
    for (int i = n-nc; i < n; i++){
      v_x[i] = apply_right(v,hi,n,i);
    }
  };

  //=============================================================================
  // Operator definitions. TOOD: Make factory functions?
  //=============================================================================

  /** 
  * Standard 2nd order central
  **/
  template <>
  constexpr D1_central<3,1,2>::D1_central()
    : interior_stencil{-1./2, 0, 1./2}, 
      closure_stencils{{-1., 1}}
  {}

  /** 
  * Standard 4th order central
  **/
  template <>
  constexpr D1_central<5,4,6>::D1_central()
    : interior_stencil{1./12, -2./3, 0., 2./3, -1./12}, 
      closure_stencils{
        {-24./17, 59./34, -4./17,  -3./34, 0, 0},
        {-1./2, 0, 1./2,  0, 0, 0},
        {4./43, -59./86, 0, 59./86, -4./43, 0},
        {3./98, 0, -59./98, 0, 32./49, -4./49}}
  {}

  /** 
  * Standard 6th order central
  **/
  template <>
  constexpr D1_central<7,6,9>::D1_central()
    : interior_stencil{-1./60, 3./20, -3./4, 0, 3./4, -3./20, 1./60}, 
      closure_stencils{
        {-21600./13649, 104009./54596,  30443./81894,  -33311./27298, 16863./27298, -15025./163788, 0, 0, 0},
        {-104009./240260, 0, -311./72078,  20229./24026, -24337./48052, 36661./360390, 0, 0, 0},
        {-30443./162660, 311./32532, 0, -11155./16266, 41287./32532, -21999./54220, 0, 0, 0},
        {33311./107180, -20229./21436, 485./1398, 0, 4147./21436, 25427./321540, 72./5359, 0, 0},
        {-16863./78770, 24337./31508, -41287./47262, -4147./15754, 0, 342523./472620, -1296./7877, 144./7877, 0},
        {15025./525612, -36661./262806, 21999./87602, -25427./262806, -342523./525612, 0, 32400./43801, -6480./43801, 720./43801}}
  {}
} //End namespace sbp