"""A module for parsing dimension shape expressions for dltype."""

from __future__ import annotations

import enum
import math
import re
import typing

from typing_extensions import override

from dltype._lib import _log_utils

_logger: typing.Final = _log_utils.get_logger(__name__)


class _DLTypeSpecifier(enum.Enum):
    """An enum representing a way to specify a name for a dimension expression or literal."""

    EQUALS = "="

    def __repr__(self) -> str:
        return self.value


class _DLTypeGroupToken(enum.Enum):
    """An enum for grouping tokens."""

    LPAREN = "("
    RPAREN = ")"
    COMMA = ","

    def __repr__(self) -> str:
        return self.value


class _DLTypeModifier(enum.Enum):
    """An enum representing a modifier that can be applied to a dimension expression."""

    ANONYMOUS_MULTIAXIS = "..."
    NAMED_MULTIAXIS = "*"

    def __repr__(self) -> str:
        return self.value


class _DLTypeFunctionalOperator(enum.Enum):
    """A function."""

    MIN = "min"
    MAX = "max"
    MOD = "mod"

    def evaluate(self, args: tuple[int, int]) -> int:
        """Evaluate the unary operator."""
        a, b = args
        match self:
            case _DLTypeFunctionalOperator.MIN:
                return min(a, b)
            case _DLTypeFunctionalOperator.MAX:
                return max(a, b)
            case _DLTypeFunctionalOperator.MOD:
                return a % b


class _DLTypeUnaryFunctionOperator(enum.Enum):
    """A unary function."""

    ISQRT = "isqrt"

    def evaluate(self, args: tuple[int]) -> int:
        """Evaluate the unary operator."""
        match self:
            case _DLTypeUnaryFunctionOperator.ISQRT:
                return math.isqrt(args[0])


class _DLTypeInfixOperator(enum.Enum):
    """An enum representing a mathematical operator for a dimension expression."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    EXP = "^"
    DIV = "/"

    def evaluate(self, args: tuple[int, int]) -> int:
        """Evaluate the operator."""
        a, b = args
        match self:
            case _DLTypeInfixOperator.ADD:
                return a + b
            case _DLTypeInfixOperator.SUB:
                return a - b
            case _DLTypeInfixOperator.MUL:
                return a * b
            case _DLTypeInfixOperator.EXP:
                return int(a**b)
            case _DLTypeInfixOperator.DIV:
                return a // b


_DLTypeOperator = _DLTypeFunctionalOperator | _DLTypeInfixOperator | _DLTypeUnaryFunctionOperator


def op_precedence(  # noqa: PLR0911
    op: _DLTypeOperator | _DLTypeGroupToken,
) -> int:
    match op:
        case _DLTypeInfixOperator.ADD | _DLTypeInfixOperator.SUB:
            return 1
        case _DLTypeInfixOperator.MUL | _DLTypeInfixOperator.DIV:
            return 2
        case _DLTypeInfixOperator.EXP:
            return 3
        case _DLTypeFunctionalOperator.MIN | _DLTypeFunctionalOperator.MAX | _DLTypeFunctionalOperator.MOD:
            return 4
        case _DLTypeUnaryFunctionOperator.ISQRT:
            return 5
        case _DLTypeGroupToken.LPAREN:
            return 6
        case _DLTypeGroupToken.RPAREN | _DLTypeGroupToken.COMMA:
            return 0


_VALID_IDENTIFIER_RX: typing.Final = re.compile(r"^[a-zA-Z][a-zA-Z0-9\_]*$")


class DLTypeDimensionExpression:
    """A class representing a dimension that depends on other dimensions."""

    def __init__(
        self,
        identifier: str,
        postfix_expression: list[str | int | _DLTypeOperator],
        *,
        is_multiaxis_literal: bool = False,
        is_anonymous: bool = False,
        is_named_multiaxis: bool = False,
    ) -> None:
        """Create a new dimension expression."""
        self.identifier = identifier
        self.parsed_expression = postfix_expression
        # multiaxis literals cannot be evaluated until
        # the actual shape is known, so we don't consider this to be a true literal
        # for the purposes of evaluating the expression
        self.is_literal = not is_multiaxis_literal and all(
            isinstance(token, int) for token in postfix_expression
        )
        self.is_identifier = (
            is_multiaxis_literal or is_named_multiaxis or (postfix_expression == [identifier])
        )
        # this is an expression if it's not a literal value, if it's
        # an identifier that points to another dimension, or if it's an
        # identifier that doesn't just point to itself
        self.is_expression = not (self.is_identifier and self.is_literal) and (
            len(postfix_expression) > 1 or self.identifier not in postfix_expression
        )
        self.is_multiaxis_literal = is_multiaxis_literal
        self.is_anonymous = is_anonymous
        self.is_named_multiaxis = is_named_multiaxis
        _logger.debug(
            "Created new %s dimension expression %r", "multiaxis" if self.is_multiaxis_literal else "", self
        )

        # ensure we don't have any self-referential expressions
        if self.is_expression and self.identifier in self.parsed_expression:
            msg = f"Self-referential expression {self=}"
            raise SyntaxError(msg)

    @override
    def __repr__(self) -> str:
        """Get a string representation of the dimension expression."""
        if self.is_anonymous:
            return f"Anonymous<{self.identifier}>"
        if self.is_literal:
            return f"Literal<{self.identifier}={self.parsed_expression}>"
        return f"Identifier<{self.identifier}={self.parsed_expression}>"

    @classmethod
    def from_multiaxis_literal(
        cls,
        identifier: str,
        literal: int,
        *,
        is_anonymous: bool = False,
    ) -> DLTypeDimensionExpression:
        """
        Create a new dimension expression from a multi-axis literal.

        This is a special case where the expression is a single literal that is repeated across all axes.
        Anonymous axes are a special case where the actual value of the literal is irrelevant.
        """
        return cls(
            identifier,
            [literal],
            is_multiaxis_literal=True,
            is_anonymous=is_anonymous,
        )

    def evaluate(self, scope: dict[str, int]) -> int:
        """Evaluate the expression."""
        _logger.debug("Evaluating expression %s with scope %s", self, scope)
        stack: list[int] = []

        if self.is_anonymous:
            msg = "Cannot evaluate an anonymous axis"
            raise ValueError(msg)

        if self.identifier in scope:
            # if the identifier is in the scope, we return the value directly
            # however if we're an anonymous axis, we don't want to
            # return the value directly as the prior scoped value is irrelevant
            return scope[self.identifier]

        for token in self.parsed_expression:
            match token:
                case _DLTypeUnaryFunctionOperator():
                    a = stack.pop()
                    stack.append(token.evaluate((a,)))
                case _DLTypeFunctionalOperator() | _DLTypeInfixOperator():
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(token.evaluate((a, b)))
                case int():
                    stack.append(token)
                case str():
                    # intentionally allow KeyError to be raised if the identifier is not in the scope
                    stack.append(scope[token])

        if len(stack) != 1:
            msg = f"Invalid stack {stack=}"
            raise ValueError(msg)

        _logger.debug("Evaluated expression %r to %r", self, stack[0])
        return stack[0]


def _maybe_multiaxis(
    identifier: str,
    expression: list[TokenT | str | int],
) -> DLTypeDimensionExpression | None:
    # this is a modified expression, so we need to handle it differently
    if len(expression) == 1 and expression[0] == _DLTypeModifier.ANONYMOUS_MULTIAXIS.value:
        return DLTypeDimensionExpression(identifier, [], is_anonymous=True)

    if len(expression) == 2 and expression[0] == _DLTypeInfixOperator.MUL and isinstance(expression[1], str):  # noqa: PLR2004
        if not _VALID_IDENTIFIER_RX.match(expression[1]):
            msg = f"{expression[1]} is not a valid multiaxis identifier"
            raise SyntaxError(msg)
        return DLTypeDimensionExpression(expression[1], [expression[1]], is_named_multiaxis=True)

    return None


def _flush_op_by_precedence(
    stack: list[str | _DLTypeOperator],
    postfix: list[str | int | _DLTypeOperator],
    current_op: _DLTypeOperator | _DLTypeGroupToken,
) -> None:
    # Pop operators with higher or equal precedence
    while (
        stack
        and (isinstance(stack[-1], _DLTypeOperator | _DLTypeGroupToken))
        and op_precedence(stack[-1]) >= op_precedence(current_op)
    ):
        postfix.append(stack.pop())


def _postfix_from_infix(identifier: str, expression: list[TokenT | str | int]) -> DLTypeDimensionExpression:  # noqa: C901, PLR0912, PLR0915
    """
    Extract a postfix expression from an infix expression.

    Notably, this function does not handle function calls, which are handled separately.
    """
    _logger.debug("Parsing infix expression %r", expression)

    if not expression:
        msg = f"Argument list empty ({identifier})"
        raise SyntaxError(msg)

    if maybe_multiaxis := _maybe_multiaxis(identifier, expression):
        return maybe_multiaxis

    # Convert infix to postfix using shunting yard algorithm
    scope_vars: set[str] = set()
    stack: list[str | _DLTypeOperator] = []
    postfix: list[str | int | _DLTypeOperator] = []
    current_index = 0

    while current_index < len(expression):
        token = expression[current_index]
        print(f"current token {token}", postfix)
        match token:
            case _DLTypeInfixOperator():
                print(f"{token} infix")
                current_op = token
                _flush_op_by_precedence(stack, postfix, current_op)

                stack.append(current_op)
                current_index += 1
            case _DLTypeFunctionalOperator() | _DLTypeUnaryFunctionOperator() | _DLTypeGroupToken.LPAREN:
                print(f"{token} func")
                current_op = token
                _flush_op_by_precedence(stack, postfix, current_op)

                lparen, comma_indices, rparen = _get_group_indices(expression[current_index:], current_index)
                if isinstance(token, _DLTypeFunctionalOperator) and len(comma_indices) != 1:
                    msg = f"{token.value} requires two arguments, received {len(comma_indices) + 1}"
                    raise SyntaxError(msg)
                if isinstance(token, _DLTypeUnaryFunctionOperator) and len(comma_indices) != 0:
                    msg = f"{token.value} requires one argument, received {len(comma_indices) + 1}"
                    raise SyntaxError(msg)
                if token == _DLTypeGroupToken.LPAREN and len(comma_indices) != 0:
                    msg = "Group received invalid comma separator"
                    raise SyntaxError(msg)

                lhs = lparen
                for arg_idx in [*comma_indices, rparen]:
                    inner_expr = _postfix_from_infix(
                        f"{identifier}[{arg_idx}]", expression[lhs + 1 : arg_idx]
                    )
                    postfix.extend(inner_expr.parsed_expression)
                    scope_vars.update(exp for exp in inner_expr.parsed_expression if isinstance(exp, str))
                    lhs = arg_idx

                if isinstance(current_op, _DLTypeFunctionalOperator | _DLTypeUnaryFunctionOperator):
                    stack.append(current_op)
                current_index = rparen + 1

                if token == _DLTypeGroupToken.LPAREN and len(comma_indices) != 0:
                    msg = "Group received invalid comma separator"
                    raise SyntaxError(msg)

                lhs = lparen
                for arg_idx in [*comma_indices, rparen]:
                    inner_expr = _postfix_from_infix(
                        f"{identifier}[{arg_idx}]", expression[lhs + 1 : arg_idx]
                    )
                    postfix.extend(inner_expr.parsed_expression)
                    scope_vars.update(exp for exp in inner_expr.parsed_expression if isinstance(exp, str))
                    lhs = arg_idx

                if isinstance(current_op, _DLTypeUnaryFunctionOperator | _DLTypeFunctionalOperator):
                    stack.append(current_op)
                current_index = rparen + 1
            case int():
                print(f"{token} int")
                postfix.append(token)
                current_index += 1

            case str():
                print(f"{token} string")
                if _VALID_IDENTIFIER_RX.match(str(token)) is None:
                    msg = f"Invalid expression={identifier} [{token=}] pos={current_index}/{len(expression)}"
                    raise SyntaxError(msg)
                postfix.append(token)
                scope_vars.add(token)
                current_index += 1
            case _DLTypeGroupToken.RPAREN | _DLTypeGroupToken.COMMA | _DLTypeSpecifier.EQUALS:
                msg = f"Invalid expression={identifier} [{token=}] pos={current_index}/{len(expression)}"
                raise SyntaxError(msg)

        print(f"postfix after {token}", postfix, stack)
    # Pop any remaining operators
    while stack:
        postfix.append(stack.pop())

    _logger.debug("Parsed infix expression %r to postfix %r", identifier, postfix)
    return DLTypeDimensionExpression(identifier, postfix)


TokenT: typing.TypeAlias = _DLTypeSpecifier | _DLTypeGroupToken | _DLTypeOperator


def _span_to_tok(character: str) -> TokenT | None:
    return typing.cast(
        "TokenT",
        (
            _DLTypeFunctionalOperator._value2member_map_
            | _DLTypeInfixOperator._value2member_map_
            | _DLTypeUnaryFunctionOperator._value2member_map_
            | _DLTypeSpecifier._value2member_map_
            | _DLTypeGroupToken._value2member_map_
        ).get(character),
    )


def _span_to_str_or_int(span: str) -> str | int:
    if span.isnumeric():
        return int(span)
    return span


def _assert_token_list_valid(tokenized: list[str | int | TokenT]) -> None:
    if len(tokenized) == 0:
        msg = "Empty expression"
        raise SyntaxError(msg)

    if len(tokenized) == 1 and isinstance(tokenized[0], str | int):
        # special case where the only token is an identifier
        return

    if len(tokenized) == 2 and tokenized[0] == _DLTypeInfixOperator.MUL and isinstance(tokenized[1], str):  # noqa: PLR2004
        # is an anonymous named axis
        return

    # other than the special case above, fold the iterated list in to make sure the operators are balanced
    n_expected_args = 1  # expected at least one expression
    n_actual_args = 0

    for tok in reversed(tokenized):
        if isinstance(tok, _DLTypeUnaryFunctionOperator):
            n_expected_args += 1
            n_actual_args += 1
        elif isinstance(tok, _DLTypeFunctionalOperator | _DLTypeInfixOperator):
            # all operators are binary
            n_expected_args += 2
            n_actual_args += 1
        elif isinstance(tok, str | int):
            n_actual_args += 1
        elif isinstance(tok, _DLTypeGroupToken):
            continue
        else:
            raise SyntaxError(tok)  # noqa: TRY004

    if n_expected_args != n_actual_args:
        _logger.error("%s Expected %d args received %d", tokenized, n_expected_args, n_actual_args)
        raise SyntaxError("Invalid expression syntax: " + "".join(map(repr, tokenized)))


def _tokenize_string_expr(
    expression: str,
) -> list[str | int | TokenT]:
    return_list: list[str | int | TokenT] = []
    current_span = ""
    for character in expression:
        if character == " ":
            msg = "Spaces not permitted in dimension expressions"
            raise SyntaxError(msg)

        if token := _span_to_tok(character):
            if current_span:
                return_list.append(_span_to_tok(current_span) or _span_to_str_or_int(current_span))
            current_span = ""
            return_list.append(token)
        else:
            current_span += character
    if current_span:
        return_list.append(_span_to_tok(current_span) or _span_to_str_or_int(current_span))
    _assert_token_list_valid(return_list)
    return return_list


def _get_group_indices(expr: list[TokenT | int | str], offset: int) -> tuple[int, list[int], int]:
    lparen_idx: int | None = None
    comma_idx: list[int] = []
    rparen_idx: int | None = None
    nesting_depth = 0

    for idx, tok in enumerate(expr):
        if tok == _DLTypeGroupToken.LPAREN:
            nesting_depth += 1
            lparen_idx = idx + offset if nesting_depth == 1 else lparen_idx
        elif tok == _DLTypeGroupToken.COMMA and nesting_depth == 1:
            comma_idx.append(idx + offset)
        elif tok == _DLTypeGroupToken.RPAREN:
            rparen_idx = idx + offset if nesting_depth == 1 else rparen_idx
            nesting_depth -= 1

        if rparen_idx:
            break

    if lparen_idx is None:
        msg = f"Invalid function syntax {expr=}, missing ("
        raise SyntaxError(msg)
    if rparen_idx is None:
        msg = f"Invalid function syntax {expr=}, missing )"
        raise SyntaxError(msg)

    if lparen_idx > rparen_idx or any(c_idx < lparen_idx or c_idx > rparen_idx for c_idx in comma_idx):
        msg = f"Unbalanced parenthesis in expression {''.join(map(repr, expr))}"
        raise SyntaxError(msg)

    return lparen_idx, comma_idx, rparen_idx


def expression_from_string(expression: str) -> DLTypeDimensionExpression:
    """Parse a dimension expression from a string and return a parsed expression."""
    if not expression:
        msg = f"Empty expression {expression=}"
        raise SyntaxError(msg)

    # split the expression into the identifier and the expression if it has a specifier
    identifier = expression
    if _DLTypeSpecifier.EQUALS.value in expression:
        identifier, expression = expression.split(_DLTypeSpecifier.EQUALS.value, maxsplit=1)
    tokenized = _tokenize_string_expr(expression)
    return _postfix_from_infix(identifier, tokenized)
