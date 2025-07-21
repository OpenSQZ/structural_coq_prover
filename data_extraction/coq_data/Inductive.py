from .Def_class import VernacBase, Vernac2Indent, ConstrBase, coq_constr
from typing import List, Dict, Any, Tuple

@VernacBase.register_subclass("Inductive")
class VernacInductive(VernacBase):
    @classmethod
    def from_string(cls, lines: List[str], category: str) -> Tuple['VernacInductive', List['VernacBase']]:
        def_lines = [line.strip() for line in lines[1:-1]]
        try:
            ind_name = def_lines[0].split()[1]
        except:
            ind_name = ' '.join(def_lines[0:2]).split()[1]
            
        ind_content = ' '.join(def_lines[:]).replace("@IND Constructors Begin", "").strip().replace("@IND Constructors End", "")
        ind_content = ind_content.split(Vernac2Indent[category], 1)[1].strip()


        full_text = ' '.join(def_lines)
        constructor_contents = cls.extract_constructors(full_text)
        constructor_vernacs = []

        for constr in constructor_contents:
            #  Logic.eq_refl : x = x :> A
            #  only ":" or ":>"
            if ":" in constr:
                name, content = constr.split(":", 1)
                if content.lstrip().startswith(">"):
                    name, content = constr.split(":>", 1)
            else:
                raise ValueError(f"Invalid constructor format: {constr}")
                
            constructor_vernacs.append(VernacBase(
                category="Constructor",
                name=name.strip(),
                content=content.strip()
            ))

        return cls(
            category=category,
            name=ind_name,
            content=ind_content,
        ), constructor_vernacs

    @staticmethod
    def extract_constructors(text):
        begin_marker = "@IND Constructors Begin"
        end_marker = "@IND Constructors End"   
        
        result = []
        start = 0
        
        while True:
            begin_index = text.find(begin_marker, start)
            if begin_index == -1:
                break
                
            end_index = text.find(end_marker, begin_index)
            if end_index == -1:
                break
                
            content = text[begin_index + len(begin_marker):end_index].strip()
            result.append(content)
            
            start = end_index + len(end_marker)
            
        return result

@ConstrBase.register_subclass("Inductive")
class ConstrInductive(ConstrBase):
    @classmethod
    def from_string(cls, lines: List[str], category: str) -> Tuple['ConstrInductive', List['ConstrBase']]:
        def_lines = [line.strip() for line in lines[1:-1]]
        
        for line in def_lines:
            if line.startswith("Name:"):
                name = line.split("Name:", 1)[1].strip()
                break
                
        content_lines = []
        in_type = False
        for line in def_lines:
            if line.startswith("Type:"):
                in_type = True
                continue
            elif line.startswith("@IND"):
                break
            if in_type:
                content_lines.append(line)
                
        content = coq_constr.from_string(content_lines)[1] if content_lines else None
        
        constructor_contents = cls.extract_constructors(lines)
        constructor_constrs  = []
        constructor_constr_list = []
        constructor_names = []

        for constr_lines in constructor_contents:
            constr_name = None
            constr_type_lines = []
            in_type = False
            
            for line in constr_lines:
                if line.startswith("Name:"):
                    constr_name = line.split("Name:", 1)[1].strip()
                elif line == "Type:":
                    in_type = True
                elif in_type:
                    constr_type_lines.append(line)
                    
            if constr_name and constr_type_lines:
                constr_content = coq_constr.from_string(constr_type_lines)[1]
                constructor_names.append(constr_name)
                constructor_constr_list.append(constr_content)
                constructor_constrs.append(ConstrBase(
                    category="Constructor",
                    name=constr_name,
                    content=constr_content
                ))

        return cls(
            category=category,
            name=name,
            content=coq_constr.combine_constr(content, constructor_constr_list, constructor_names),
        ), constructor_constrs

    @staticmethod
    def extract_constructors(lines: List[str]) -> List[List[str]]:
        constructors = []
        current_constructor = []
        in_constructor = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("@@IND Constructor Begin"):
                in_constructor = True
                current_constructor = []
            elif line.startswith("@@IND Constructor End"):
                in_constructor = False
                if current_constructor:
                    constructors.append(current_constructor)
            elif in_constructor:
                current_constructor.append(line)
                
        return constructors