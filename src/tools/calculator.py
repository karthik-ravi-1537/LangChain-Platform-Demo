"""
Simple calculator tool for LangChain agents.
"""

import re

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    expression: str = Field(description="Mathematical expression to evaluate (e.g., '2 + 3 * 4')")


class CalculatorTool(BaseTool):
    """A simple calculator tool that can perform basic mathematical operations."""

    name: str = "calculator"
    description: str = "Useful for performing mathematical calculations. Input should be a mathematical expression."
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Execute the calculator tool."""
        try:
            # Clean the expression - remove spaces and validate
            cleaned_expr = expression.strip()

            # Basic validation - only allow numbers, operators, and parentheses
            if not re.match(r"^[0-9+\-*/().\s]+$", cleaned_expr):
                return f"Error: Invalid characters in expression '{cleaned_expr}'"

            # Evaluate the expression safely
            result = eval(cleaned_expr)
            return f"Result: {result}"

        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: Could not evaluate '{expression}' - {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Async version of the calculator tool."""
        return self._run(expression)


class AdvancedCalculatorTool(BaseTool):
    """Advanced calculator with more mathematical functions."""

    name: str = "advanced_calculator"
    description: str = "Advanced calculator with support for mathematical functions like sqrt, sin, cos, etc."
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Execute the advanced calculator tool."""
        try:

            # Clean the expression
            cleaned_expr = expression.strip()

            # Replace common mathematical functions
            replacements = {
                "sqrt(": "math.sqrt(",
                "sin(": "math.sin(",
                "cos(": "math.cos(",
                "tan(": "math.tan(",
                "log(": "math.log(",
                "log10(": "math.log10(",
                "exp(": "math.exp(",
                "pi": "math.pi",
                "e": "math.e",
            }

            for func, replacement in replacements.items():
                cleaned_expr = cleaned_expr.replace(func, replacement)

            # Validate expression (more permissive for advanced functions)
            if not re.match(r"^[0-9+\-*/().\s,mathlognpsicoexqrt]+$", cleaned_expr):
                return f"Error: Invalid characters in expression '{cleaned_expr}'"

            # Evaluate the expression
            result = eval(cleaned_expr)
            return f"Result: {result}"

        except Exception as e:
            return f"Error: Could not evaluate '{expression}' - {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Async version of the advanced calculator tool."""
        return self._run(expression)


# Create tool instances
calculator_tool = CalculatorTool()
advanced_calculator_tool = AdvancedCalculatorTool()


def demonstrate_calculator_tools():
    """Demonstrate the calculator tools."""
    print("🧮 Calculator Tools Demo")
    print("=" * 50)

    # Test basic calculator
    print("\n1. Basic Calculator:")
    test_expressions = ["2 + 3", "10 * 5", "100 / 4", "2 + 3 * 4", "(2 + 3) * 4"]

    for expr in test_expressions:
        result = calculator_tool._run(expr)
        print(f"   {expr} = {result}")

    # Test advanced calculator
    print("\n2. Advanced Calculator:")
    advanced_expressions = ["sqrt(16)", "sin(pi/2)", "log(e)", "2 * pi", "exp(1)"]

    for expr in advanced_expressions:
        result = advanced_calculator_tool._run(expr)
        print(f"   {expr} = {result}")


if __name__ == "__main__":
    demonstrate_calculator_tools()
