from ConstraintSyntaxTrees import *

# Representation of the MiniLUSP Optimisation constraints
constraints, optimising_constraints = split_constraints([
    "Lowland Peat Overlap Constraints"
  , (0.02781 * var("P_lo") + var("G")) <= 1
  | (0.02781 * var("P_lo") + var("G")) <= 1
  , (0.02701 * var("P_lo") + var("O")) <= 1
  | (0.02701 * var("P_lo") + var("O")) <= 1

  , "Silvoarable Overlap Constraints"
  , (0.3799 * var("S_A") + var("G")) <= 1
  | (0.3799 * var("S_A") + var("G")) <= 1
  , (0.3690 * var("S_A") + var("O")) <= 1
  | (0.3690 * var("S_A") + var("O")) <= 1
  , (0.4815 * var("S_A") + var("WL")) <= 1
  | (0.4815 * var("S_A") + var("WL")) <= 1


  , "Silvopastoral Overlap Constraints"
  , 0.3667 * var("S_P") + var("G") <= 1
  | 0.3667 * var("S_P") + var("G") <= 1
  , 0.4648 * var("S_P") + var("WL") <= 1
  | 0.4648 * var("S_P") + var("WL") <= 1
  , 0.3561 * var("S_P") + var("O") <= 1
  | 0.3561 * var("S_P") + var("O") <= 1
  , 0.7934 * var("S_P") + var("WP") <= 1
  | 0.7934 * var("S_P") + var("WP") <= 1


  , "Woodland Overlap Constraints"
  , (var("WL") + 1.3393 * var("G") <= 1.3393)
  # overconstrainted version
  | (var("WL") + 1.3393 * var("G") <= 1.3393)

  , (var("WL") + 1.3791 * var("O") <= 1.3791) 
  | (var("WL") + 1.3791 * var("O") <= 1.3791)
  , (var("WL") + 0.5858 * var("WP") <= 1.0818)
  | (var("WL") + 0.5858 * var("WP") <= 1.0818)

  , "Wood Pasture Overlap Constraints"
  , (var("WP") + 1.2298 * var("G") <= 1.3617) 
  | (var("WP") + 1.2298 * var("G") <= 1.3617)
  , (var("WP") + 1.2065 * var("O") <= 1.3384)
  | (var("WP") + 1.2065 * var("O") <= 1.3384)

  , "Grassland-Organic Overlap Constraints"
  , (var("G") + var("O") <= 1)
  | (var("G") + var("O") <= 1)
  ])

# Uncomment me to generate LaTeX specification document when running the dashboard
# generateLatexSpecification(constraints)