"""
Sketch (similar to Coarse-to-Fine)

- keep Python keywords as is
- strip off arguments and variable names
    - substitute tokens with types: `NUMBER`, `STRING`
- specialize `NAME` token:
    - for functions: `FUNC#<num_args>`

# Examples

x = 1 if True else 0
NAME = NUMBER if True else NUMBER

result = SomeFunc(1, 2, 'y', arg)
NAME = FUNC#4 ( NUMBER , NUMBER , STRING , NAME )

result = [x for x in DoWork(xs) if x % 2 == 0]
NAME = [ NAME for NAME in FUNC#1 ( NAME ) if NAME % NUMBER == NUMBER ]
"""

import ast
import astpretty

import token
from tokenize import tokenize, TokenInfo
import io
import builtins
from collections import defaultdict
from termcolor import colored


class ASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}  # map function name -> num args

        self.name_by_type = {
            ast.Attribute: lambda x: x.attr,
            ast.Name     : lambda x: x.id,
            ast.Subscript: lambda x: x.slice.value.id,
        }

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Call):
            self.visit_Call(node.func)
        else:
            func_name = self.name_by_type[type(node.func)](node.func)

            for arg in node.args:
                self.generic_visit(arg)

            self.functions[func_name] = len(node.args)


class SketchVocab:
    NAME_ID = "NAME"
    FUNC_ID = "FUNC"
    STR_LITERAL_ID = "STRING"
    NUM_LITERAL_ID = "NUMBER"
    # RESERVED_ID = "<reserved>"
    # ACCESSOR_ID = "<accessor>"
    # ASSIGN_ID = "<assign>"
    # ARITHMETIC_ID = "<arithmetic>"
    # OP_ID = "<op>"


class Sketch:

    def __init__(self, code_snippet: str, verbose=False):
        self.code_snippet = code_snippet

        self.names = defaultdict(lambda: [])
        self.keywords = defaultdict(lambda: [])
        self.literals = defaultdict(lambda: [])
        self.operators = defaultdict(lambda: [])

        self.ordered = []

        if verbose:
            print(colored(" * tokenizing [%s]" % code_snippet, 'yellow'))
        self.tok_list = list(tokenize(io.BytesIO(self.code_snippet.encode('utf-8')).readline))

        # AST
        self.ast_visitor = ASTVisitor()
        self.ast = None
        try:
            self.ast = self.ast_visitor.visit(ast.parse(self.code_snippet))
        except SyntaxError:
            if verbose:
                print(colored(" * skipping ast generation for [%s]" % code_snippet, 'red'))

    def refine_name(self, tok: TokenInfo):
        if self.is_reserved_keyword(tok.string):
            self.keywords[tok.string].append(tok.start[1])
            self.ordered.append(tok.string)
        else:
            self.names[tok.string].append(tok.start[1])
            if tok.string in self.ast_visitor.functions:
                self.ordered.append(SketchVocab.FUNC_ID + "#%d" % self.ast_visitor.functions[tok.string])
            else:
                self.ordered.append(SketchVocab.NAME_ID)

    def generate(self):
        """
        TODO:
        - separate support for slices?
        """

        for tok in self.tok_list:
            tok_type = token.tok_name[tok.type]

            if tok_type == 'NAME':
                self.refine_name(tok)

            elif tok_type == 'STRING':
                self.literals[tok.string].append(tok.start[1])
                self.ordered.append(SketchVocab.STR_LITERAL_ID)

            elif tok_type == 'NUMBER':
                self.literals[tok.string].append(tok.start[1])
                self.ordered.append(SketchVocab.NUM_LITERAL_ID)

            elif tok_type == 'OP':
                self.operators[tok.string].append(tok.start[1])
                self.ordered.append(tok.string)

            else:
                assert tok_type in ['ENCODING', 'NEWLINE', 'ENDMARKER', 'ERRORTOKEN'], "%s" % tok_type

        return self

    def details(self):
        return "names: %s\nkeywords: %s\nliterals: %s\noperators: %s" % (
            str(list(self.names.keys())),
            str(list(self.keywords.keys())),
            str(list(self.literals.keys())),
            str(list(self.operators.keys()))
        )

    def split(self, delim=' '):
        return str(self).split(delim)

    def __str__(self):
        return ' '.join(self.ordered)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.ordered)

    @staticmethod
    def is_reserved_keyword(name):
        RESERVED_KEYWORDS = set(dir(builtins) + [
            "and", "assert", "break", "class", "continue", "def", "del", "elif",
            "else", "except", "exec", "finally", "for", "from", "global", "if",
            "import", "in", "is", "lambda", "not", "or", "pass", "print", "raise",
            "return", "try", "while", "yield", "None", "self"
        ])  # len = 182

        return name in RESERVED_KEYWORDS


def main():
    # v = ASTVisitor()
    # t = v.visit(ast.parse('x = SomeFunc(2, 3, y, "test")'))
    # print(v.functions)
    # astpretty.pprint(tree.body[0], indent=' ' * 4)
    # exec(compile(tree, filename="<ast>", mode="exec"))

    code_snippet = "return getattr(instance, name)()"
    astpretty.pprint(ast.parse(code_snippet).body[0], indent=' ' * 4)

    sketch = Sketch(code_snippet, verbose=True).generate()

    # print(sketch.details())
    print(sketch)


if __name__ == '__main__':
    main()
