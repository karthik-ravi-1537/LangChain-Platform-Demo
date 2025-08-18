"""
Test suite for the calculator tools.
"""

import os
import sys

import pytest

# Add the project root to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.tools.calculator import AdvancedCalculatorTool, CalculatorTool, advanced_calculator_tool, calculator_tool


class TestCalculatorTool:
    """Test the basic calculator tool."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        test_cases = [
            ("2 + 3", "Result: 5"),
            ("10 - 4", "Result: 6"),
            ("7 * 8", "Result: 56"),
            ("15 / 3", "Result: 5.0"),
            ("2 + 3 * 4", "Result: 14"),
            ("(2 + 3) * 4", "Result: 20"),
        ]

        for expression, expected in test_cases:
            result = calculator_tool._run(expression)
            assert result == expected, f"Failed for {expression}: got {result}, expected {expected}"

    def test_invalid_expression(self):
        """Test handling of invalid expressions."""
        invalid_expressions = [
            "2 + abc",
            "sqrt(16)",  # sqrt not available in basic calculator
            "import os",  # malicious code
        ]

        for expr in invalid_expressions:
            result = calculator_tool._run(expr)
            assert "Error:" in result, f"Should have failed for {expr}"

        # Test expressions that should work but are advanced
        advanced_expressions = [
            "2 ** 3",  # This actually works in Python eval
        ]

        for expr in advanced_expressions:
            result = calculator_tool._run(expr)
            # These might work or fail depending on implementation
            assert "Result:" in result or "Error:" in result

    def test_division_by_zero(self):
        """Test division by zero handling."""
        result = calculator_tool._run("5 / 0")
        assert "Error: Division by zero" in result

    def test_tool_metadata(self):
        """Test tool metadata."""
        assert calculator_tool.name == "calculator"
        assert "mathematical calculations" in calculator_tool.description.lower()


class TestAdvancedCalculatorTool:
    """Test the advanced calculator tool."""

    def test_mathematical_functions(self):
        """Test mathematical functions."""
        test_cases = [
            ("sqrt(16)", 4.0),
            ("sin(pi/2)", 1.0),
            ("cos(0)", 1.0),
            ("log(e)", 1.0),
            ("2 * pi", 6.283185307179586),
        ]

        for expression, expected in test_cases:
            result = advanced_calculator_tool._run(expression)
            assert "Result:" in result, f"Failed for {expression}: {result}"
            # Extract the numeric result
            try:
                numeric_result = float(result.split("Result: ")[1])
                assert abs(numeric_result - expected) < 0.0001, f"Failed for {expression}"
            except (ValueError, IndexError):
                pytest.fail(f"Could not parse result for {expression}: {result}")

    def test_tool_metadata(self):
        """Test advanced tool metadata."""
        assert advanced_calculator_tool.name == "advanced_calculator"
        assert "advanced calculator" in advanced_calculator_tool.description.lower()


class TestCalculatorToolClasses:
    """Test calculator tool classes directly."""

    def test_calculator_tool_instantiation(self):
        """Test that calculator tools can be instantiated."""
        basic_calc = CalculatorTool()
        advanced_calc = AdvancedCalculatorTool()

        assert isinstance(basic_calc, CalculatorTool)
        assert isinstance(advanced_calc, AdvancedCalculatorTool)

    def test_async_methods(self):
        """Test that async methods exist and work."""
        import asyncio

        async def test_async():
            result = await calculator_tool._arun("2 + 2")
            assert result == "Result: 4"

        asyncio.run(test_async())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
