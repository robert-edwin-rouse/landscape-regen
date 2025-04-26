
"""
This module defines a simple constraint syntax tree that can be used to
represent constraints in a linear program. The syntax tree is made up of
two classes: Expr and Constraint. An Expr object represents an expression
in the constraint, while a Constraint object represents a constraint in
the linear program.

The expression can be a constant, a variable, or an operation on other expressions.
The Expr class has methods for evaluating the expression using a model, and it overloads
the +, *, and <= operators to allow for easy construction of expressions and constraints.

The Constraint class has methods for checking if the constraint is satisfied by
a model, and for balancing a model using the constraint, to get a new model
satisfying the constraint.
"""
from typing import Dict, Callable



class Constraint:
  """
  Represention of a linear constraint with an expression on the left
   and a float on the right. The constraint can be checked for satisfaction
   using a model, and can be balanced using a model to get a new model that
   satisfies the constraint.
  """
  def __init__(self, left, right : float):
    self.left = left
    self.right = right

  def isSatisfied(self, model : Dict[str, float]) -> bool:
    """
    Evaluate the constraint on a set of data
    """
    return (self.left.eval(model) <= self.right)

  def evaluate_homogeneous_constraint(self, model : Dict[str, float]) -> float:
     """
     Given a model, evaluate the constraint in exact homogenous form,
     i.e., for `lhs <= rhs` then `lhs - rhs = 0` is the exact homogenous form;
     evaluate the left-hand side of this, i.e., `lhs - rhs` on a model.
     """
     lhs = self.left.eval(model)
     rhs = self.right
     return (lhs - rhs)

  def balance(self, model : Dict[str, float]) -> Dict[str, float]:
    """
    Use the constraint to balance a model
    """
    model_out = model.copy()
    # For every var in the constraint expression
    for var in self.left.getVars():
      # Scale the variable by the ratio of the right hand side
      # to the left hand side
      model_out[var] = model[var] / (self.left.eval(model) * self.right)
    return model_out

  def __repr__(self):
    """
    Return a string representation of the constraint
    """
    return f"{self.left} <= {self.right}"

  def __or__(self, other):
    """
    Return a constraint pair, used for handling overconstrained
    constraints for optimising model
    """
    # Validate the variables are the same, i.e., same kind of shape:
    if (self.left.getVars() == other.left.getVars()):
      return ConstraintPair(self, other)
    else:
      # Raise an exception if the variables are not the same
      raise(ValueError(f"Cannot describe an over-constrained constraint with different variables: {self.left.getVars()} and {other.left.getVars()}"))

  def toLatex(self):
    """
    Return a LaTeX representation of the constraint
    """
    return f"{self.left.toLatex()} \\leq {self.right}"

class ConstraintPair:
   """Represents a pair of a constraint and an alterante version which
   should have the same syntactic shape but different coeffecients for
   the purpose of overconstraining
   """
   def __init__(self, normal : Constraint, over_constraint : Constraint):
      self.normal = normal
      self.over_constraint = over_constraint

class Expr:
    def __init__(self, value, op=None, right=None):
        self.value = value
        self.op = op
        self.right = right

    def eval(self, model : Dict[str, float]) -> float:
      """
      Evaluate the expression using the given model
      """
      # No operation so either a variable or float
      if self.op is None:
          # If a float constant, just return that otherwise look up in the model
          return self.value if (type(self.value) == float) else model[self.value]
      else:
        # Evaluate the left and right
        left = self.value.eval(model)
        right = self.right.eval(model)
        # Interpret the operation
        if self.op == '+':
            return left + right
        elif self.op == '*':
            return left * right
        else:
            raise ValueError(f"Unknown operator {self.op}")

    # Magic methods to allow for easy construction of expressions
    def __add__(self, other):
        return Expr(self, '+', other if isinstance(other, Expr) else Expr(other))

    def __radd__(self, other):
        # this handles something like 2 + x
        return Expr(Expr(other), '+', self)

    def __mul__(self, other):
        return Expr(self, '*', other if isinstance(other, Expr) else Expr(other))

    def __rmul__(self, other):
        # this handles something like 2 * x
        return Expr(Expr(other), '*', self)

    def __le__(self, other):
        return Constraint(self, other)

    def getVars(self):
        """
        Return a list of all the variables in the expression
        """
        if self.op is None and type(self.value) == str:
            return [self.value]
        left = self.value.getVars() if isinstance(self.value, Expr) else []
        right = self.right.getVars() if isinstance(self.right, Expr) else []
        return left + right

    def __repr__(self):
        """
        Return a string representation of the expression
        """
        if self.op is None:
            return str(self.value)
        return f"({self.value} {self.op} {self.right})"

    def toLatex(self):
        """
        Return a LaTeX representation of the expression
        """
        if self.op is None:
            if type(self.value) == str:
                # If the variable contains a '_' turn this
                # into a subscript in LaTeX
                if '_' in self.value:
                    return self.value.replace('_', '_{') + '}'
                else:
                  return self.value
            else:
              return str(self.value)
        return f"({self.value.toLatex()} {self.op} {self.right.toLatex()})"

# Helper
def var(name : str) -> Expr:
    """
    Create a variable expression
    """
    return Expr(name)

# Print the constraints, for checking purposes
def print_constraints(constraints : list[Constraint]):
    for constraint in constraints:
        print(constraint)

def split_constraints(constraints : list[Constraint | ConstraintPair]) -> tuple[list[Constraint], list[Constraint]]:
  left = []
  right = []
  for c in constraints:
    if isinstance(c, ConstraintPair):
      left.append(c.normal)
      right.append(c.over_constraint)
    elif isinstance(c, Constraint):
      left.append(c)
  return (left, right)

# Generate a LaTeX representation of the constraints in specification.tex
def generateLatexSpecification(constraints : list[Constraint]):
  """
  Generate a LaTeX representation of a set of constraints
  """
  baseTemplateStart = """\\documentclass{article} \n\
\\usepackage{amsmath} \n\
\\begin{document} \n\
\\begin{center} \n\
\\begin{align*}
"""
  baseTemplateEnd = """
\\end{align*} \n\
\\end{center} \n\
\\end{document} \
"""
  # Generate the LaTeX for each constraint
  latexConstraints = []
  for constraint in constraints:
      if type(constraint) == str:
        latexConstraints.append("\\end{align*}\n\n" + constraint + "\n\n\\begin{align*}\n")
      else:
        spacing = "\\\\\n" if constraint != constraints[-1] else ""
        latexConstraints.append(constraint.toLatex() + spacing)

  # Join the constraints together
  constraintsString = "".join(latexConstraints)
  # Write the latex document to specification.tex
  print("Writing LaTeX specification of constraints to specification.tex")
  with open('specification.tex', 'w') as f:
    f.write(baseTemplateStart + constraintsString + baseTemplateEnd)
  return