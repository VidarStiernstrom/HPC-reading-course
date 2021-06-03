  //=============================================================================
  // Specializations of D1_central::apply<dim,region>
  //=============================================================================
  template<>
  template<> inline PetscScalar D1_central<Stencils_2nd,3,1,2>::apply<1,left>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_x_left(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_2nd,3,1,2>::apply<1,interior>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_x_interior(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_2nd,3,1,2>::apply<1,right>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_x_right(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_2nd,3,1,2>::apply<2,left>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_y_left(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_2nd,3,1,2>::apply<2,interior>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_y_interior(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_2nd,3,1,2>::apply<2,right>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_y_right(v, hi, i, j, comp);};

  // 4th order
  template<>
  template<> inline PetscScalar D1_central<Stencils_4th,5,4,6>::apply<1,left>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_x_left(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_4th,5,4,6>::apply<1,interior>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_x_interior(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_4th,5,4,6>::apply<1,right>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_x_right(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_4th,5,4,6>::apply<2,left>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_y_left(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_4th,5,4,6>::apply<2,interior>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_y_interior(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_4th,5,4,6>::apply<2,right>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_y_right(v, hi, i, j, comp);};

  // 6th order
  template<>
  template<> inline PetscScalar D1_central<Stencils_6th,7,6,9>::apply<1,left>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_x_left(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_6th,7,6,9>::apply<1,interior>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_x_interior(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_6th,7,6,9>::apply<1,right>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_x_right(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_6th,7,6,9>::apply<2,left>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_y_left(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_6th,7,6,9>::apply<2,interior>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_y_interior(v, hi, i, j, comp);};
  template<>
  template<> inline PetscScalar D1_central<Stencils_6th,7,6,9>::apply<2,right>(const grid::grid_function_2d<PetscScalar> v, const PetscScalar hi, 
                                                                              const PetscInt i, const PetscInt j, const PetscInt comp) const {return apply_y_right(v, hi, i, j, comp);};