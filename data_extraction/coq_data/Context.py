from .Def_class import VernacBase, ConstrBase, Vernac2Indent
from typing import List

@VernacBase.register_subclass("Context")
class VernacContext(VernacBase):
    @classmethod
    def from_string(cls, lines: List[str], category: str) -> 'VernacContext':
        def_lines = [line.strip() for line in lines[1:-1]]
        
        try:
            name = def_lines[0].split()[1]
        except:
            name = ' '.join(def_lines[0:2]).split()[1]

        # Handle typeclass instance syntax: `{EqDec A} -> {EqDec A}
        if name.startswith('`'):
            name = name[1:]
        # Handle implicit/explicit arguments:
        if name.startswith('{') or name.startswith('('):
            name = name[1:]
        # Handle implicit/explicit arguments:
        if name.endswith('}') or name.endswith(')'):
            name = name[:-1]
        # Handle cumulative typeclass instance:
        if name.startswith('!'):
            name = name[1:]

        ident = Vernac2Indent[category]

        content = def_lines[0].split(ident, 1)[1].strip()
        if len(def_lines) > 1:
            content += " " + " ".join(def_lines[1:])
        
        return cls(category=category, name=name.strip(), content=content)

@ConstrBase.register_subclass("Context")
class ConstrContext(ConstrBase):
    pass

