from .Def_class import VernacBase
from typing import List

@VernacBase.register_subclass("TacticNotation")
class VernacTacticNotation(VernacBase):
    @classmethod
    def from_string(cls, lines: List[str], category: str) -> 'VernacTacticNotation':
        def_lines = [line.strip() for line in lines[1:-1]]
        name_lines = def_lines[0]
        local_name_lines = []
        content_lines = []
        for line in def_lines[1:]:
            if 'Body:' in line:
                content_lines.append(line)
                content_lines.extend(def_lines[def_lines.index(line) + 1:])
                break
            local_name_lines.append(line)
        local_name = ' '.join(local_name_lines)
        try:
            assert name_lines.startswith('Name:')
            assert local_name.startswith('LocalName:')
            assert content_lines[0].startswith('Body:')
        except:
            print(def_lines)
            raise ValueError("TacticNotation format Error")

        name_local_temp = local_name.split(' ')[1]
        name_temp = name_lines.split(' ',1)[1]

        if name_local_temp in name_temp:
            name = name_temp.split(name_local_temp)[0] + name_local_temp
        else:
            name = name_temp.split('_', 1)[0]
            print('some error may occur in TacticNotation, please check')
            print(def_lines)
        # we add name to the content as tactic notations are defined with parameters
        # so we manually add it to the content
        content = ' '.join(local_name.split()[1:]) + '\n' + ' '.join(content_lines[1:])

        return cls(category=category, name=name, content=content)

## TacticNotation do not have constr
## TODO