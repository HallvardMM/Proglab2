"""Calculator file
    RPN doesn't ned paranteser RPN (reversed polish notation)"""

import numbers
import re
import unittest
from math import e
import numpy


class Container:
    """ Container super class"""

    def __init__(self):
        self.items = []

    def size(self):
        """:returns elements in items"""
        return len(self.items)

    def is_empty(self):
        """check if items is empty"""
        return self.size() == 0

    def push(self, item):
        """insert item behind in array"""
        self.items.append(item)

    def peek(self, position):
        """returns peeked element"""
        return self.items[position]

    def pop(self, position):
        """removes and returns poped element"""
        return self.items.pop(position)

    def clear(self):
        """removes all elements"""
        self.items.clear()

    def __str__(self):
        return str(self.items)


class Queue(Container):
    """ Queue subclass"""

    def __init__(self):
        super(Queue, self).__init__()

    def peek(self, position=0):
        """:returns first element in array"""
        assert not self.is_empty(), "peek: list is empty"
        return super(Queue, self).peek(position)

    def pop(self, position=0):
        """:returns and removes first element in array"""
        assert not self.is_empty(), "pop: list is empty"
        return super(Queue, self).pop(position)


class Stack(Container):
    """Stack subclass"""

    def __init__(self):
        super(Stack, self).__init__()

    def peek(self, position=-1):
        """:returns last element in array"""
        assert not self.is_empty(), "peek: list is empty"
        return super(Stack, self).peek(position)

    def pop(self, position=-1):
        """:returns and removes last element in array"""
        assert not self.is_empty(), "pop: list is empty"
        return super(Stack, self).pop(position)


class Function:
    """Function class """

    def __init__(self, func):
        self.func = func

    def execute(self, element, debug=True):
        """solve with numpy function"""
        # Check type
        if not isinstance(element, numbers.Number):
            raise TypeError("Cannot execute func if element is not a number")
        result = self.func(element)
        # report
        if debug is True:
            print(
                "Function: " +
                self.func.__name__ +
                "({:f}) = {:f}".format(
                    element,
                    result))
            # Go home
        return result

    def __repr__(self):  # Hvorfor fungerte ikke __str__, men __repr__
        return self.func.__name__


class Operator:
    """to choose different numpy operators
    - numpy.add, numpy.subtract, numpy.multiply, numpy.divide"""

    def __init__(self, operator, strength):
        self.operator = operator
        self.strength = strength

    def execute(self, a_value, b_value):
        """solve with numpy operator"""
        return self.operator(a_value, b_value)

    def __repr__(self):  # Hvorfor fungerte ikke __str__, men __repr__
        return self.operator.__name__


class Calculator:
    """Main class - the calculator"""

    def __init__(self):
        # Define the functions supported by linking them to Python
        # functions. These can be made elsewhere in the program,
        # or imported ( e . g . , from numpy)
        self.functions = {'EXP': Function(numpy.exp),
                          'LOG': Function(numpy.log),
                          'SIN': Function(numpy.sin),
                          'COS': Function(numpy.cos),
                          'SQRT': Function(numpy.sqrt)}
        # Define the operators supported.
        # Link them to Python functions (here: from numpy)
        self.operators = {'PLUSS': Operator(numpy.add, 0),
                          'GANGE': Operator(numpy.multiply, 1),
                          'DELE': Operator(numpy.divide, 1),
                          'MINUS': Operator(numpy.subtract, 0)}
        # Define the output−queue. The parse_text method fills this with RPN.
        # The evaluate_output_queue method evaluates it
        self.output_queue = Queue()

    def evaluate_rpn(self):
        """Function to solve array of reversed polish notation"""
        stack = Stack()
        for i in range(0, self.output_queue.size()):
            value = self.output_queue.pop()
            if isinstance(value, float):
                stack.push(value)
            elif isinstance(value, Function):
                temp = value.execute(stack.pop())
                stack.push(temp)
            elif isinstance(value, Operator):
                a_value = stack.pop()
                b_value = stack.pop()
                temp = value.execute(b_value, a_value)
                stack.push(temp)
        return stack.pop()

    def shunting_yard(self, input_list):
        """function for implementing shunting yard"""
        stack = Stack()
        queue = Queue()
        while len(input_list) > 0:
            elem = input_list.pop(0)
            if isinstance(elem, (float, int)):
                queue.push(elem)
            elif isinstance(elem, Function):
                stack.push(elem)
            elif elem == '(' or elem == "(":
                stack.push(elem)
            elif elem == ')' or elem == ")":
                temp = None
                while temp != '(' or temp != "(":
                    temp = stack.pop()
                    if temp != '(' or temp != "(":
                        queue.push(temp)
            elif isinstance(elem, Operator):
                temp = None
                while (isinstance(temp, Operator)
                       and temp.strength >= elem.strength) \
                        or isinstance(temp, Function) or temp is None:
                    if stack.size() > 0:
                        temp = stack.peek()
                    else:
                        temp = "None"
                    if isinstance(
                            temp, Operator) and temp.strength >= elem.strength:
                        temp = stack.pop()
                        queue.push(temp)
                stack.push(elem)
        while stack.size() > 0:
            elem = stack.pop()
            queue.push(elem)
        self.output_queue = queue

    def parser(self, text):
        """:input Math equation as text
         :returns list"""
        text = text.replace(" ", "").upper()
        output = []
        functions = "|".join(["^" + func for func in self.functions.keys()])
        operators = "|".join(
            ["^" + operations for operations in self.operators.keys()])
        floats = "^[-0123456789.]+"
        while len(text) > 0:
            if re.search(floats, text):
                check = re.search(floats, text)
                output.append(float(check.group(0)))
                text = text[check.end(0):]
            elif re.search(functions, text):
                check = re.search(functions, text)
                output.append(self.functions.get(str(check.group(0))))
                text = text[check.end(0):]
            elif re.search(operators, text):
                check = re.search(operators, text)
                output.append(self.operators.get(str(check.group(0))))
                text = text[check.end(0):]
            elif re.search(r"^\(|\)", text):
                check = re.search(r"^\(|\)", text)
                output.append(str(check.group(0)))
                text = text[check.end(0):]
        print("parser output: ", output)
        return output

    def calculate_expression(self, text):
        """Function to solve equation as string"""
        self.shunting_yard(self.parser(text))
        return self.evaluate_rpn()


class Test(unittest.TestCase):
    """Class for testing other classes"""

    def test_queue(self):
        """Function to test the queue-class"""
        queue = Queue()
        self.assertTrue(queue.is_empty())
        for i in range(0, 3):
            queue.push(i)
        self.assertEqual(queue.size(), 3)
        for i in range(0, 3):
            self.assertEqual(queue.peek(i), i)

        self.assertEqual(queue.pop(), 0)
        self.assertEqual(queue.pop(), 1)
        self.assertEqual(queue.pop(), 2)
        self.assertRaises(
            AssertionError,
            lambda: queue.peek())  # Hvorfor bruke lambda?

    def test_stack(self):
        """Function to test the stack-class"""
        stack = Stack()
        self.assertTrue(stack.is_empty())
        for i in range(0, 3):
            stack.push(i)
        self.assertEqual(stack.size(), 3)
        for i in range(0, 3):
            self.assertEqual(stack.peek(i), i)
        self.assertEqual(stack.pop(), 2)
        self.assertEqual(stack.pop(), 1)
        self.assertEqual(stack.pop(), 0)
        self.assertRaises(AssertionError, lambda: stack.pop())

    def test_function(self):
        """Function to test the function-class"""
        exponential_func = Function(numpy.exp)
        sin_func = Function(numpy.sin)
        self.assertEqual(exponential_func.execute(sin_func.execute(0)), 1)

    def test_operator(self):
        """Function to test the operator-class"""
        add_op = Operator(numpy.add, 0)
        multiply_op = Operator(numpy.multiply, 1)
        self.assertEqual(add_op.execute(1, multiply_op.execute(2, 3)), 7)

    def test_calculator(self):
        """Function to test the calculator-class"""
        calc = Calculator()
        self.assertAlmostEqual(
            calc.functions['EXP'].execute(
                calc.operators['PLUSS'].execute(
                    1,
                    calc.operators['GANGE'].execute(
                        2,
                        3))), e ** 7)

    def test_rpn(self):
        """Function to test the rpn-function i calculator"""
        calc = Calculator()
        calc.output_queue.push(1.)
        calc.output_queue.push(2.)
        calc.output_queue.push(3.)
        calc.output_queue.push(calc.operators['GANGE'])
        calc.output_queue.push(calc.operators['PLUSS'])
        calc.output_queue.push(calc.functions['EXP'])
        self.assertAlmostEqual(calc.evaluate_rpn(), e ** 7)

    def test_shunting_yard(self):
        """Test with print since comparing objects was hard and not mandatory"""
        calc = Calculator()
        print("Input: ", [Function(numpy.exp), '(', 1, Operator(
            numpy.add, 0), 2, Operator(numpy.multiply, 1), 3, ')'])
        calc.shunting_yard([Function(numpy.exp), '(', 1, Operator(
            numpy.add, 0), 2, Operator(numpy.multiply, 1), 3, ')'])
        print(calc.output_queue)
        print("Input2: ", [2, Operator(numpy.multiply, 1), 3, Operator(
            numpy.add, 0), 1])
        calc.shunting_yard([2, Operator(numpy.multiply, 1), 3, Operator(
            numpy.add, 0), 1])
        print(calc.output_queue)
        print("Input3: ", [2, Operator(numpy.divide, 1), 3, Operator(
            numpy.subtract, 0), 1])
        calc.shunting_yard([2, Operator(numpy.divide, 1), 3, Operator(
            numpy.subtract, 0), 1])
        print(calc.output_queue)

    def test_parser(self):
        """test of parser with print because of objects"""
        calc = Calculator()
        print("exp(1 pluss 2 gange 3)")
        print(calc.parser("EXP(1 pluss 2 gange 3)"))
        print("sin(1 pluss -2) dele 3.45")
        print(calc.parser("sin(1 pluss -2) dele 3.45"))

    def test_whole_calculator(self):
        """test for calculator with given examples"""
        calc = Calculator()
        self.assertAlmostEqual(
            calc.calculate_expression("exp(1 pluss 2 gange 3"), e**7)
        self.assertEqual(calc.calculate_expression(
            "((15 dele (7 minus (1 pluss 1))) gange 3) minus (2 pluss (1 pluss 1))"), 5)


def test():
    """main for tests"""
    testobj = Test()
    testobj.test_queue()
    testobj.test_stack()
    testobj.test_function()
    testobj.test_operator()
    testobj.test_calculator()
    testobj.test_rpn()
    testobj.test_shunting_yard()
    testobj.test_parser()
    testobj.test_whole_calculator()


if __name__ == '__main__':
    #test()
    calc = Calculator()
    print(calc.calculate_expression(
        "(15 gange (2 gange 3 dele 3) gange 3 minus 3 pluss -45"))
