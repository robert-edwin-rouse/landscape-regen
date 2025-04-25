from ConstraintSyntaxTrees import *

# Representation of the MiniLUSP Optimisation constraints
constraints, optimising_constraints = split_constraints([
    "Lowland Peat Overlap Constraints"
  , (0.0258 * var("P_lo") + var("G")) <= 1
  , (0.0258 * var("P_lo") + var("O")) <= 1

  , "Silvoarable Overlap Constraints"
  , (0.3647 * var("S_A") + var("G")) <= 1
  , (0.3647 * var("S_A") + var("O")) <= 1
  , (0.4619 * var("S_A") + var("WL")) <= 1

  , "Silvopastoral Overlap Constraints"
  , 0.3856 * var("S_P") + var("G") <= 1
  , 0.4883 * var("S_P") + var("WL") <= 1
  , 0.3856 * var("S_P") + var("O") <= 1
  , 0.8036 * var("S_P") + var("WP") <= 1

  , "Woodland Overlap Constraints"
  , (var("WL") + 1.3328 * var("G") <= 1.3328)
  # overconstrainted version
  | (var("WL") + 1.3791 * var("G") <= 1.3791)

  , var("WL") + 1.3328 * var("O") <= 1.3328 
  , var("WL") + 0.7218 * var("WP") <= 1.0986

  , "Wood Pasture Overlap Constraints"
  , var("WP") + 1.3255 * var("G") <= 1.4433
  , var("WP") + 1.3255 * var("O") <= 1.4433

  , "Grassland-Organic Overlap Constraints"
  , var("G") + var("O") <= 1
  ])

# Uncomment me to generate LaTeX specification document when running the dashboard
# generateLatexSpecification(constraints)