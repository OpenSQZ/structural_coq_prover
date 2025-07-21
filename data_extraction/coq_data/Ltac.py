from .Def_class import VernacBase
from typing import List

@VernacBase.register_subclass("Ltac")
class VernacLtac(VernacBase):
    @classmethod
    def from_string(cls, lines: List[str], category: str) -> 'VernacLtac':
        def_lines = [line.strip() for line in lines[1:-1]]
        name = def_lines[0].split("Name:")[1].strip()
        if def_lines[1].startswith("Body:"):
            content = ' '.join(def_lines[2:])
        else:
            raise ValueError("Ltac body Error")

        return cls(category=category, name=name, content=content)

## Ltac do not have constr
## TODO