from .Def_class import VernacBase, Vernac2Indent, ConstrBase
from typing import List

@VernacBase.register_subclass("Fixpoint")
class VernacFixpoint(VernacBase):
    @classmethod
    def from_string(cls, lines: List[str], category: str) -> 'VernacFixpoint':
        def_lines = [line.strip() for line in lines[1:-1]]

        if not def_lines[0].startswith("Let"):
            try:
                name = def_lines[0].split()[1]
            except:
                name = ' '.join(def_lines[0:2]).split()[1]
        else:
            try:
                name = def_lines[0].split()[2]
            except:
                name = ' '.join(def_lines[0:3]).split()[2]

        ident = Vernac2Indent[category]

        content = def_lines[0].split(ident, 1)[1].strip()
        if len(def_lines) > 1:
            content += " " + " ".join(def_lines[1:])
        
        return cls(category=category, name=name, content=content)

@ConstrBase.register_subclass("Fixpoint")
class ConstrFixpoint(ConstrBase):
    pass
