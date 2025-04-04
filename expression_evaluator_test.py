from utils.expression_evaluator import ExpressionEvaluator, SecurityException
import pytest
import base64

def test_security_check1():
    expression = "import sys"
    expression_evaluator = ExpressionEvaluator({}, {})
    with pytest.raises(SecurityException):
        expression_evaluator.evaluate_expression(expression)
    expression = "__import__('sys')"
    evaluator = ExpressionEvaluator({}, {})
    with pytest.raises(SecurityException):
        expression_evaluator.evaluate_expression(expression)


def test_security_check2():
    # with eval such an expression is possible, we are trying to avoid this scenario in our case
    # it's applicable for any import
    expression = 'eval(base64.b64decode("X19pbXBvcnRfXygnc3lzJyk=").decode("utf-8")).version'
    print(eval(expression))
    expression = "X19pbXBvcnRfXygnc3lzJyk=" # base64 encoded version of string "__import__('sys')"
    assert base64.b64encode(b"__import__('sys')").decode("utf-8") == expression
    expression_evaluator = ExpressionEvaluator({}, {})
    with pytest.raises(SecurityException):
        expression_evaluator.evaluate_expression(expression)
    expression = "WDE5cGJYQnZjblJmWHlnbmMzbHpKeWs9"  # base64 encoding applied twice on the string "__import__('sys')"
    assert base64.b64encode(base64.b64encode(b"__import__('sys')")).decode("utf-8") == expression
    expression_evaluator = ExpressionEvaluator({}, {})
    with pytest.raises(SecurityException):
        expression_evaluator.evaluate_expression(expression)

def test_security_check3():
    # with eval such an expression is possible, we are trying to avoid this scenario in our case
    # it's applicable for any import
    expression = '"".__class__.__base__'
    expression_evaluator = ExpressionEvaluator({}, {})
    with pytest.raises(SecurityException):
        expression_evaluator.evaluate_expression(expression)

def test_evaluate_expression1():
    expression = "x+2"
    locals = {"x":3}
    expression_evaluator = ExpressionEvaluator({}, locals)
    output = expression_evaluator.evaluate_expression(expression)
    expected_output = 5
    assert expected_output==output


if __name__=="__main__":
    test_evaluate_expression1()
