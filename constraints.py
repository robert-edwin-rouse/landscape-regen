from ConstraintSyntaxTrees import *

# Representation of the MiniLUSP Optimisation constraints
constraints = [
    "Lowland Peat Overlap Constraints"
  , (0.0258 * Expr("P_lo") + Expr("G")) <= 1
  , (0.0258 * Expr("P_lo") + Expr("O")) <= 1

  , "Silvoarable Overlap Constraints"
  , (0.3647 * Expr("S_A") + Expr("G")) <= 1
  , (0.3647 * Expr("S_A") + Expr("O")) <= 1
  , (0.4619 * Expr("S_A") + Expr("WL")) <= 1

  , "Silvopastoral Overlap Constraints"
  , 0.3856 * Expr("S_P") + Expr("G") <= 1
  , 0.4883 * Expr("S_P") + Expr("WL") <= 1
  , 0.3856 * Expr("S_P") + Expr("O") <= 1
  , 0.8036 * Expr("S_P") + Expr("WP") <= 1

  , "Woodland Overlap Constraints"
  , 0.7503 * Expr("WL") + Expr("G") <= 1
  , 0.7503 * Expr("WL") + Expr("O") <= 1
  , Expr("WL") + 0.7218 * Expr("WP") <= 1.0986

  , "Wood Pasture Overlap Constraints"
  , Expr("WP") + 1.3255 * Expr("G") <= 1.4433
  , Expr("WP") + 1.3255 * Expr("O") <= 1.4433

  , "Grassland-Organic Overlap Constraints"
  , Expr("G") + Expr("O") <= 1
  ]

# Uncomment me to generate LaTeX specification document when running the dashboard
# generateLatexSpecification(constraints)