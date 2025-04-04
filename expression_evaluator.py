import copy
import logging

class ExpressionEvaluator(object):
    """
    Use this class with caution, as it could have security implications
    https://realpython.com/python-eval-function/#minimizing-the-security-issues-of-eval
    https://stackoverflow.com/questions/661084/security-of-pythons-eval-on-untrusted-strings
    The number of possible expressions will be limited
    """
    def __init__(self, globals={}, locals={}):
        """
        Global context & local context to be used
        """
        self.__globals = copy.deepcopy(globals)
        self.__globals["__builtins__"] = {}
        self.__locals = copy.deepcopy(locals)

    @staticmethod
    def __check_for_security_issues(expression):
        ExpressionEvaluator.__run_basic_security_check(expression)
        # # TODO complete the below functionality
        from base64 import b64decode
        import binascii
        decoded_expression = expression
        for i in range(2):
            try:
                decoded_expression = b64decode(decoded_expression).decode("utf-8")
                ExpressionEvaluator.__run_basic_security_check(decoded_expression)
            except (binascii.Error, UnicodeDecodeError) as e:
                # a lot of strings/expressions won't be byte64 encoded, we can ignore those errors
                # if decoding once is not byte64, next iteration also won't be byte64
                break

    @staticmethod
    def __run_basic_security_check(expression):
        assert "__" not in expression
        assert "._" not in expression
        assert "import" not in expression
        assert "base64" not in expression
        assert "decode" not in expression
        assert "encode" not in expression
        max_length = 100
        assert len(expression)<=max_length, f"expression length limited to {max_length}"

    @staticmethod
    def __remove_cases_with_potential_issues(expression:str):
        expression = expression.replace("import", "")
        return expression

    def evaluate_expression(self, expression):
        """
        Given an expression to evaluate returns the results
        """
        # FIXME https://realpython.com/python-eval-function/#minimizing-the-security-issues-of-eval
        # Using eval function in python could cause major security risk, Use alternative if possible
        # limit the number of expressions a user could try out
        eval_result = None
        try:
            ExpressionEvaluator.__check_for_security_issues(expression)
            # TODO it isn't even supposed to come to the below code if there is a malicious code,
            #  it's safer not to evaluate at all, so we can as well remove it

            expression = ExpressionEvaluator.__remove_cases_with_potential_issues(expression)
            eval_result = eval(expression, self.__globals, self.__locals)
        except AssertionError:
            raise SecurityException(f"Only basic expressions are allowed, modify your expression - {expression}")
        return eval_result

class SecurityException(Exception):
    def __init__(self, message, *args, **kwargs):
        # Call the base class constructor with the parameters it needs
        super().__init__(message, *args, **kwargs)