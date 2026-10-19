"""An interpreter, NOT exec with filtered builtins.

Generated Python never becomes host bytecode. The only effect is yielding a
validated H0 command to sed.runner; responses are allowlisted visible data.
No host object, callable, exception, filesystem or network is in the language.
"""
import ast
from copy import deepcopy
import json
import re

from sed.visibility import visible_observation

MAX_SOURCE = 32768
MAX_VALUE = 131072


class Rejected(ValueError):
    pass


def bounded(value):
    if len(json.dumps(value)) > MAX_VALUE:
        raise Rejected("Tool value limit")
    return value


def validate(tool):
    if not isinstance(tool, dict) or set(tool) != {"name", "description", "parameters", "code"}:
        raise Rejected("Expected tool name, description, parameters, code")
    name, params, source = tool["name"], tool["parameters"], tool["code"]
    identifier = lambda s: type(s) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", s) and "__" not in s
    if (not identifier(name) or name in {"env", "create_mcp"} or not isinstance(params, list)
            or len(params) > 3 or any(not identifier(p) or p == "env" for p in params)
            or len(set(params)) != len(params) or type(source) is not str or len(source) > MAX_SOURCE
            or type(tool["description"]) is not str or not 0 < len(tool["description"]) <= 4096):
        raise Rejected("Invalid tool schema")
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError, ValueError):
        raise Rejected("Invalid Python") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise Rejected("One self-contained function required")
    fn = tree.body[0]
    if (fn.name != name or fn.decorator_list or fn.returns or fn.type_comment
            or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.defaults
            or fn.args.vararg or fn.args.kwarg
            or [a.arg for a in fn.args.args] != ["env", *params]
            or any(a.annotation for a in fn.args.args)):
        raise Rejected("Signature must be (env, up to three string parameters)")
    allowed = (ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.Expr, ast.Return,
        ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Name,
        ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Subscript,
        ast.Call, ast.Attribute, ast.BinOp, ast.Add, ast.Compare, ast.Eq, ast.NotEq,
        ast.In, ast.NotIn, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
        ast.JoinedStr, ast.FormattedValue)
    nodes = list(ast.walk(fn))
    if len(nodes) > 3000:
        raise Rejected("Tool syntax limit")
    for node in nodes:
        if not isinstance(node, allowed) or (isinstance(node, ast.FunctionDef) and node is not fn):
            raise Rejected("Unsupported Python syntax")
        if isinstance(node, ast.Name) and (not identifier(node.id) or
                (node.id == "env" and isinstance(node.ctx, ast.Store))):
            raise Rejected("Invalid variable")
        if isinstance(node, (ast.Assign, ast.For)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(not isinstance(t, ast.Name) or t.id == "env" for t in targets):
                raise Rejected("Only local name assignment allowed")
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or not (
                    (node.value.id == "env" and node.attr in {"step", "observe"}) or
                    (node.value.id != "env" and node.attr == "append")):
                raise Rejected("Host attributes are forbidden")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Attribute) or node.keywords):
            raise Rejected("Only env.step, env.observe and local list.append calls allowed")
        if isinstance(node, ast.Constant) and type(node.value) not in (str, int, bool, type(None)):
            raise Rejected("Only primitive constants allowed")
        if isinstance(node, ast.FormattedValue) and (node.conversion != -1 or node.format_spec):
            raise Rejected("Format conversion forbidden")
    return fn


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class Machine:
    def __init__(self, tool, arguments, visible, *, fuel=10000):
        self.fn = validate(tool)
        if (not isinstance(arguments, dict) or set(arguments) != set(tool["parameters"])
                or any(type(v) is not str or len(v) > 4096 for v in arguments.values())):
            raise Rejected("Arguments must match string parameters")
        self.locals = dict(arguments)
        self.visible = visible_observation(visible)
        self.observations = []
        self.fuel = fuel

    def tick(self):
        self.fuel -= 1
        if self.fuel < 0:
            raise Rejected("Tool instruction limit")

    def expression(self, node):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.locals:
                raise Rejected("Unknown local name")
            return deepcopy(self.locals[node.id])
        if isinstance(node, (ast.List, ast.Tuple)):
            values = []
            for item in node.elts:
                values.append((yield from self.expression(item)))
            return bounded(values)
        if isinstance(node, ast.Dict):
            values = {}
            for key, value in zip(node.keys, node.values):
                k = yield from self.expression(key)
                if type(k) is not str:
                    raise Rejected("String dictionary key required")
                values[k] = yield from self.expression(value)
            return bounded(values)
        if isinstance(node, ast.Subscript):
            value = yield from self.expression(node.value)
            key = yield from self.expression(node.slice)
            if not ((type(value) is dict and type(key) is str) or
                    (type(value) in (list, str) and type(key) is int)):
                raise Rejected("Invalid subscript")
            return deepcopy(value[key])
        if isinstance(node, ast.BinOp):
            left = yield from self.expression(node.left)
            right = yield from self.expression(node.right)
            if type(left) != type(right) or type(left) not in (str, list):
                raise Rejected("Only bounded string/list concatenation allowed")
            return bounded(left + right)
        if isinstance(node, ast.JoinedStr):
            result = ""
            for item in node.values:
                value = yield from self.expression(item.value if isinstance(item, ast.FormattedValue) else item)
                if type(value) not in (str, int, bool):
                    raise Rejected("Primitive interpolation required")
                result = bounded(result + str(value))
            return result
        if isinstance(node, ast.UnaryOp):
            return not (yield from self.expression(node.operand))
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = yield from self.expression(item)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.Compare):
            left = yield from self.expression(node.left)
            for op, item in zip(node.ops, node.comparators):
                right = yield from self.expression(item)
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.In):
                    ok = left in right
                else:
                    ok = left not in right
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            owner, method = node.func.value.id, node.func.attr
            args = []
            for arg in node.args:
                args.append((yield from self.expression(arg)))
            if owner == "env":
                if method == "observe" and not args:
                    return deepcopy(self.visible)
                if method != "step" or len(args) != 1 or type(args[0]) is not str:
                    raise Rejected("env.step requires one command")
                if args[0] not in self.visible["admissible_commands"]:
                    raise Rejected("Tool command is not currently H0 admissible")
                self.visible = visible_observation((yield args[0]))
                self.observations.append(deepcopy(self.visible))
                return deepcopy(self.visible)
            if method != "append" or type(self.locals.get(owner)) is not list or len(args) != 1:
                raise Rejected("Invalid list append")
            self.locals[owner] = bounded([*self.locals[owner], args[0]])
            bounded(self.locals)
            return None
        raise Rejected("Expression not supported")

    def block(self, nodes):
        for node in nodes:
            self.tick()
            if isinstance(node, ast.Assign):
                value = bounded((yield from self.expression(node.value)))
                for target in node.targets:
                    self.locals[target.id] = deepcopy(value)
                bounded(self.locals)
            elif isinstance(node, ast.Expr):
                yield from self.expression(node.value)
            elif isinstance(node, ast.Return):
                raise _Return((yield from self.expression(node.value)) if node.value else None)
            elif isinstance(node, ast.If):
                yield from self.block(node.body if (yield from self.expression(node.test)) else node.orelse)
            elif isinstance(node, (ast.For, ast.While)):
                items = iter((yield from self.expression(node.iter))) if isinstance(node, ast.For) else None
                while True:
                    self.tick()
                    if items is not None:
                        try:
                            self.locals[node.target.id] = next(items)
                            bounded(self.locals)
                        except StopIteration:
                            yield from self.block(node.orelse)
                            break
                    elif not (yield from self.expression(node.test)):
                        yield from self.block(node.orelse)
                        break
                    try:
                        yield from self.block(node.body)
                    except _Break:
                        break
                    except _Continue:
                        continue
            elif isinstance(node, ast.Break):
                raise _Break()
            elif isinstance(node, ast.Continue):
                raise _Continue()
            elif not isinstance(node, ast.Pass):
                raise Rejected("Statement not supported")

    def run(self):
        try:
            yield from self.block(self.fn.body)
        except _Return as result:
            value = result.value
            observations = value if type(value) is list else [value]
            if not observations or any(o not in self.observations for o in observations):
                raise Rejected("Tool must return actual action observations")
            return deepcopy(value)
        except (KeyError, IndexError, TypeError, RecursionError, _Break, _Continue):
            raise Rejected("Tool execution rejected") from None
        raise Rejected("Tool did not return observations")
