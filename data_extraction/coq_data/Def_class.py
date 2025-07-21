from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
#########################
#######  Commom  ########
#########################

"""
_Anonymous for anonymous thm context def etc.
Body: <no body> 

In Ps: Unnamed_thm for Proof witout name

Vernac Begin/End xxx
Constr Begin/End xxx

Type Begin/End

PS xxx

"""

#########################
#### Vernac Printer  ####
#########################

### Proof
"""
string_of_theorem_kind = 
    | Theorem -> "Theorem"
    | Lemma -> "Lemma"
    | Fact -> "Fact"
    | Remark -> "Remark"
    | Property -> "Property"
    | Proposition -> "Proposition"
    | Corollary -> "Corollary"

(* let pr_thm_token k = keyword (string_of_theorem_kind k) *)
let pr_thm_token_easy k = keyword ("Proof")
"""
### Assumption
"""
type assumption_object_kind = Definitional | Logical | Conjectural | Context

(* [assumption_kind]

                |  Local      | Global
   ------------------------------------
   Definitional |  Variable   | Parameter
   Logical      |  Hypothesis | Axiom
*)

(* let pr_assumption_token many discharge kind =
match discharge, kind with
    | (NoDischarge,Decls.Logical) ->
    keyword (if many then "Axioms" else "Axiom")
    | (NoDischarge,Decls.Definitional) ->
    keyword (if many then "Parameters" else "Parameter")
    | (NoDischarge,Decls.Conjectural) -> str"Conjecture"
    | (DoDischarge,Decls.Logical) ->
    keyword (if many then "Hypotheses" else "Hypothesis")
    | (DoDischarge,Decls.Definitional) ->
    keyword (if many then "Variables" else "Variable")
    | (DoDischarge,Decls.Conjectural) ->
    anomaly (Pp.str "Don't know how to beautify a local conjecture.")
    | (_,Decls.Context) ->
    anomaly (Pp.str "Context is used only internally.") *)

let pr_assumption_token_easy many discharge kind =
keyword ("Assumption")
"""
### Definition
"""
let string_of_definition_object_kind = let open Decls in function
  | Definition -> "Definition"
  | Example -> "Example"
  | Coercion -> "Coercion"
  | SubClass -> "SubClass"
  | CanonicalStructure -> "Canonical Structure"
  | Instance -> "Instance"
  | Let -> "Let"
  | (StructureComponent|Scheme|CoFixpoint|Fixpoint|IdentityCoercion|Method) ->
    CErrors.anomaly (Pp.str "Internal definition kind.")

now 
let pr_def_token dk =
    keyword (
    if Name.is_anonymous (fst id).v
    then "Goal"
    (* else string_of_definition_object_kind dk) *)
    else "DEF" )
"""
### Inductive
"""
Inductive_kw
match k with Record -> "Record" | Structure -> "Structure"
            | Inductive_kw -> "Inductive" | CoInductive -> "CoInductive"
            | Class _ -> "Class" | Variant -> "Variant"
let kind = "IND"                 
let pr_oneind key (((coe,iddecl),indpar,s,k,lc),ntn) =
    hov 0 (
    str key ++ spc() ++
        (if coe then str"> " else str"") ++ pr_ident_decl iddecl ++
two situations should be considered
"""
### Fixpoint
"""
| VernacFixpoint (local, recs) ->
let result =
let local = match local with
    | DoDischarge -> "Let "
    | NoDischarge -> ""
in
return (
    hov 0 (str local ++ keyword "Fixpoint" ++ spc () ++
            prlist_with_sep (fun _ -> fnl () ++ keyword "with"
                ++ spc ()) pr_rec_definition recs)
the same for CoFixpoint
return (
    hov 0 (local ++ keyword "CoFixpoint" ++ spc() ++
            prlist_with_sep (fun _ -> fnl() ++ keyword "with" ++ spc ()) pr_onecorec corecs)
)
"""
### Instance Context
"""
Context at @1
Instance must at @1 or @2
"""

#########################
#### Constr Printer  ####
#########################

### Definition
"""
Names are represented as absolute paths.
Type and Body are different internal representations 
identified by "_definition" "_definition_body" respectively

Example:

Constr Begin Definition

Name: Coq.Init.Logic.not

Type: 
pure constr: (A:Prop)Prop
constr: Prod(Name(A),Sort(Prop),Sort(Prop))
processed: Variable _definition : forall ( A : Prop ) -> Prop

Body: 
pure constr: [A:Prop](#->Ind((Coq.Init.Logic.False),0,))
constr: Lambda(Name(A),Sort(Prop),Prod(Anonymous,Rel(1),MutInd(Coq.Init.Logic.False,0,)))
processed: Variable _definition_body : ( Prop => forall ( _ : A ) -> Coq.Init.Logic.False.False )

Constr End Definition
"""

### Inductive
"""
for Inductive fully qualified name
Coq.Init.Datatypes.nat.nat use the Inductive itself
Coq.Init.Datatypes.nat.S use the Constructors

in our database, Inductive do not have fully qualified name(just keep the absolute path)

example:
Constr Begin Inductive

Name: Coq.Init.Logic.True

Type: 
pure constr: Prop
constr: Sort(Prop)
processed: Variable _inductive : Prop

Constr Begin Constructors

Constructor Begin

Name: Coq.Init.Logic.True.I

Type: 
pure constr: #
constr: Rel(1)
processed: Variable _sub : Rel(1)

Constructor End

Constr End Constructors

Constr End Inductive
"""

ANONYMOUS_IDENTIFIER = "_Anonymous"
AUTO_INFERRED_TYPE = "_"
DATA_TYPES_IDENTIFIER = ["Vernac", "Constr", "Type", "ProofState"]

### NOTICE: LTAC do not have an internal context

"""
for tactic processing
when in proof, we use TACTIC Begin/End to extract vernac tactics
however, it actually works for all ltac or tactic notation
but we need fully qualified name for ltac/tactic notation (to be honest, tactic begin/end's vernac info is not enough)
so we introduce VernacLtac/VernacTacticNotation
"""

class VernacCategory(Enum):
    Ltac = "Ltac" # manually defined tactics in Coq
    Definition = "Definition" # Def without proof
    Proof = "Proof" # Def with proof | Theorem / Lemma etc.
    Assumption = "Assumption"
    Inductive = "Inductive"
    Fixpoint = "Fixpoint"
    CoFixpoint = "CoFixpoint"
    Instance = "Instance"
    Context = "Context"
    Constructor = "Constructor" # Inductive Constructors
    TacticNotation = "TacticNotation" # Ltac/Tactic notation

class ConstrCategory(Enum):
    Definition = "Definition"  # Def without proof
    Proof = "Proof" # Def with proof | Theorem / Lemma etc.
    Assumption = "Assumption"
    Inductive = "Inductive"  # Constructors nested in Inductive
    Fixpoint = "Fixpoint"
    CoFixpoint = "CoFixpoint"
    Instance = "Instance"
    Context = "Context"
    Primitive = "Primitive"
    Constructor = "Constructor" # Inductive Constructors
    Arguments = "Arguments" # without vernac

@dataclass
class ContextTokenPair:
    Origin: str
    Token_ids: Optional[str] = None # Ltac do not have constr representation so far, so no token_ids
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.Origin,
            "token_ids": self.Token_ids
        }
    
    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'ContextTokenPair':
        return cls(
            Origin=dict_data["origin"],
            Token_ids=dict_data.get("token_ids")
        )
        
@dataclass
class coq_constr:
    constr: ContextTokenPair
    pure_constr: ContextTokenPair
    processed: ContextTokenPair
    text: Optional[str] = None # Type do not have original text
    
    @classmethod
    def create_completed_goal(cls) -> 'coq_constr':
        completed_text = "goalcompleted"
        return cls(
            constr=ContextTokenPair(completed_text),
            pure_constr=ContextTokenPair(completed_text),
            processed=ContextTokenPair(completed_text),
            text=completed_text
        )

    @classmethod
    def combine_constr(cls, constr: 'coq_constr', constr_list: List['coq_constr'], constructor_names: List[str]) -> 'coq_constr':
        combined_origin_pure = constr.pure_constr.Origin
        combined_origin_constr = constr.constr.Origin
        combined_origin_processed = constr.processed.Origin
        for c, name in zip(constr_list, constructor_names):
            combined_origin_pure += f" | {name} : {c.pure_constr.Origin}"
            combined_origin_constr += f" | {name} : {c.constr.Origin}"
            combined_origin_processed += f" | {name} : {c.processed.Origin}"
        
        return cls(
            constr=ContextTokenPair(combined_origin_constr),
            pure_constr=ContextTokenPair(combined_origin_pure),
            processed=ContextTokenPair(combined_origin_processed),
            text=None
        )

    @classmethod
    def from_string(cls, lines: List[str]) -> Tuple[str, 'coq_constr']:
        origin_lines = []
        pure_constr_lines = []
        constr_lines = []
        processed_lines = []
        current_section = None
        name = None

        if not lines[-1].startswith("processed"):
            lines = lines[:-1]
            if not lines[-1].startswith("processed"):
                print(lines)
                raise ValueError("Invalid Constr block format")

        for line in lines:
            line = line.strip()

            if line.startswith("Original:"):
                current_section = "origin"
                line = line.replace("Original:", "").strip()
                if line:
                    origin_lines.append(line)
            
            elif line.startswith("pure constr:"):
                current_section = "pure_constr"
                line = line.replace("pure constr:", "").strip()
                if line:
                    pure_constr_lines.append(line)
            elif line.startswith("constr:"):
                current_section = "constr"
                line = line.replace("constr:", "").strip()
                if line:
                    constr_lines.append(line)
            elif line.startswith("processed:"):
                current_section = "processed"
                line = line.replace("processed:", "").strip()
                if line:
                    processed_parts = line.split()
                    if len(processed_parts) >= 3 and processed_parts[0] == "Variable":
                        name = processed_parts[1]
                        processed_lines.append(" ".join(processed_parts[3:]))
            elif line and current_section:
                if current_section == "origin":
                    origin_lines.append(line)
                elif current_section == "pure_constr":
                    pure_constr_lines.append(line)
                elif current_section == "constr":
                    constr_lines.append(line)
                elif current_section == "processed":
                    processed_lines.append(line)
                    
        return name, cls(
            constr=ContextTokenPair(" ".join(constr_lines)),
            pure_constr=ContextTokenPair(" ".join(pure_constr_lines)),
            processed=ContextTokenPair(" ".join(processed_lines)),
            text=" ".join(origin_lines) if origin_lines else None
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constr": self.constr.to_dict(),
            "pure_constr": self.pure_constr.to_dict(),
            "processed": self.processed.to_dict(),
            "text": self.text
        }

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'coq_constr':
        if not dict_data:
            return None
        try:
            return cls(
                constr=ContextTokenPair.from_dict(dict_data["constr"]),
                pure_constr=ContextTokenPair.from_dict(dict_data["pure_constr"]),
                processed=ContextTokenPair.from_dict(dict_data["processed"]),
                text=dict_data.get("text")
            )   
        except Exception as e:
            print(e)
            print(dict_data)


"""
Vernac Begin Proof
Proof Tactics.decide_left :
  forall (C : Prop) (decide : {C} + {~ C}),
  C ->
  forall P : {C} + {~ C} -> Prop, (forall H : C, P (left _ H)) -> P decide
Vernac End Proof
"""

# category: ["Definition", 
#            "Proof" , 
#            "Assumption", 
#            "Inductive", 
#            "Fixpoint", 
#            "CoFixpoint", 
#            "Instance", 
#            "Context"]

@dataclass
class TypeItem:
    name: str
    content: Optional[coq_constr] = None
    parent_constr: Optional[str] = None

    @classmethod
    def from_string(cls, lines: List[str], parent_constr: Optional[str] = None) -> 'TypeItem':
        if not lines[0].startswith("Type Begin"):
            raise ValueError("Invalid Type block format")

        content_lines = lines[1:-1]
        name, content = coq_constr.from_string(content_lines)
        return cls(
            name=name,
            content=content,
            parent_constr=parent_constr
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            # "name": self.name,
            "content": self.content.to_dict(),
            # "parent_constr": self.parent_constr
        }

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'TypeItem':
        return cls(
            name=dict_data.get("name"),
            content=coq_constr.from_dict(dict_data["content"]) if dict_data.get("content") else None,
            parent_constr=dict_data.get("parent_constr")
        )

"""
vernac_classes = {
    "Definition": VernacDefinition,
    "Proof": VernacProof,
    "Inductive": VernacInductive,
    "Fixpoint": VernacFixpoint,
    "CoFixpoint": VernacCoFixpoint,
    "Instance": VernacInstance,
    "Context": VernacContext,
    "Assumption": VernacAssumption,
    "Ltac": VernacLtac,
    "TacticNotation": VernacTacticNotation
}

for inductive, ind_constr added in inductive.py

"""

Vernac2Indent = {
    "Definition" : "DEF",
    "Proof" : "Proof",
    "Inductive" : "IND",
    "Assumption" : "Assumption",
    "Fixpoint" : "Fixpoint",
    "CoFixpoint" : "CoFixpoint",
    "Instance" : "Instance",
    "Context" : "Context",
    "Ltac" : "Ltac",
    "TacticNotation" : "TacticNotation"
}

class VernacBase:
    _subclasses = {}
    category: str
    name: str
    content: str
    
    def __init__(self, category: str, name: str, content: str):
        self.category = category
        self.name = name
        self.content = content

    @classmethod
    def register_subclass(cls, category: str):
        def decorator(subclass):
            cls._subclasses[category] = subclass
            return subclass
        return decorator

    @classmethod
    def create_from_lines(cls, lines: List[str]) -> 'VernacBase':
        if cls is VernacBase:
            category = cls._determine_category(lines)
            subclass = cls._subclasses.get(category)
            return subclass.from_string(lines, category)
        raise NotImplementedError

    @classmethod
    def from_string(cls, lines: List[str], category: str) -> 'VernacBase':
        def_lines = [line.strip() for line in lines[1:-1]]
        try:
            name = def_lines[0].split()[1]
        except:
            name = ' '.join(def_lines[0:2]).split()[1]

        ident = Vernac2Indent[category]
        content = def_lines[0].split(ident, 1)[1].strip()

        if len(def_lines) > 1:
            content += " " + " ".join(def_lines[1:])
        
        return cls(category=category, name=name, content=content)

    @staticmethod
    def _determine_category(lines: List[str]) -> str:
        if not lines:
            raise ValueError("Empty vernacular text")
            
        first_line = lines[0].strip()
        
        category = first_line.split()[-1]

        if category not in VernacCategory.__members__:
            raise ValueError(f"Unknown vernacular category: {category}")

        return category    
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            # "category": self.category,
            # "name": self.name,
            "content": self.content
        }
    
    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'VernacBase':
        return cls(
            category=dict_data.get("category"),
            name=dict_data.get("name"),
            content=dict_data["content"]
        )

"""
Constr_class = {
    "Definition" : ConstrDefinition,
    "Proof" : ConstrProof,
    "Assumption" : ConstrAssumption,
    "Inductive" : ConstrInductive,
    "Fixpoint" : ConstrFixpoint,
    "CoFixpoint" : ConstrCoFixpoint,
    "Instance" : ConstrInstance,
    "Context" : ConstrContext,
    "Primitive" : ConstrPrimitive,
    "Arguments" : ConstrArguments
}
"""

class ConstrBase:
    _subclasses = {}
    category: str
    name: str
    content: Optional[coq_constr] = None
    body: Optional[coq_constr] = None

    def __init__(self, category: str, name: str, content: coq_constr, body: Optional[coq_constr] = None ):
        self.category = category
        self.name = name
        self.content = content
        self.body = body

    @classmethod
    def register_subclass(cls, category: str):
        def decorator(subclass):
            cls._subclasses[category] = subclass
            return subclass
        return decorator

    @classmethod
    def create_from_lines(cls, lines: List[str]) -> 'ConstrBase':
        if cls is ConstrBase:
            category = cls._determine_category(lines)
            subclass = cls._subclasses.get(category)
            return subclass.from_string(lines, category)
        raise NotImplementedError
    
    @classmethod
    def from_string(cls, lines: List[str], category: str) -> 'ConstrBase':
        def_lines = [line.strip() for line in lines[1:-1]]
        
        for line in def_lines:
            if line.startswith("Name:"):
                name = line.split("Name:", 1)[1].strip()
                break
        
        content_lines = []
        body_lines = []
        in_content = False
        in_body = False

        for line in def_lines:
            if line == "Type:":
                in_content = True
                continue
            elif line == "Body:":
                in_body = True
                in_content = False
                continue

            if in_content:
                content_lines.append(line)
            elif in_body:
                body_lines.append(line)

        if content_lines:
            _, content = coq_constr.from_string(content_lines)
        else:
            content = None
            # raise ValueError("No type information found")

        if body_lines:
            ## Proof do not have body
            _, body = coq_constr.from_string(body_lines)
        else:
            body = None

        
        return cls(
            category=category,
            name=name,
            content=content,
            body=body
        )

    @staticmethod
    def _determine_category(lines: List[str]) -> str:
        if not lines:
            raise ValueError("Empty Constr text")
            
        first_line = lines[0].strip()
        
        category = first_line.split()[-1]

        if category not in ConstrCategory.__members__:
            raise ValueError(f"Unknown constr category: {category}")

        return category  
    
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            # "category": self.category,
            # "name": self.name,
            # content None for Ltac, only have Vernac representation
            "content": self.content.to_dict() if self.content else None,
            "body": self.body.to_dict() if self.body else None
        }
    
    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'ConstrBase':
        return cls(
            category=dict_data.get("category"),
            name=dict_data.get("name"),
            content=coq_constr.from_dict(dict_data["content"]) if dict_data.get("content") else None,
            body=coq_constr.from_dict(dict_data["body"]) if dict_data.get("body") else None
        )


@dataclass
class def_object:
    Name: str
    Kind: VernacCategory
    Type: Dict[str,TypeItem] # for local variables
    File_path: str
    Origin_context: Optional[VernacBase] = None # Primitive do not have origin context so far
    Internal_context: Optional[ConstrBase] = None # Ltac do not have constr representation
    local_vars: Optional[Dict[str, Dict[str, Union[int, str]]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        local_vars_list = []
        if self.local_vars:
            for name, data in self.local_vars.items():
                item = {
                    'name': name,
                    'token_ids': data['token_ids'],
                    'type': data.get('type', '')
                }
                local_vars_list.append(item)
        return {
            "name": self.Name,
            "kind": self.Kind,
            "type": {name: type_item.to_dict() for name, type_item in self.Type.items()},
            "file_path": self.File_path,
            "origin_context": self.Origin_context.to_dict() if self.Origin_context else None,
            "internal_context": self.Internal_context.to_dict() if self.Internal_context else None,
            "local_vars": local_vars_list  
        }
    
    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'def_object':
        local_vars = {}
        if dict_data.get("local_vars"):
            for item in dict_data["local_vars"]:
                local_vars[item['name']] = {
                    'token_ids': item['token_ids'],
                    'type': item.get('type', '')
                }
        return cls(
            Name=dict_data["name"],
            Kind=dict_data["kind"],
            Type={name: TypeItem.from_dict(type_data) 
                  for name, type_data in dict_data["type"].items()},
            File_path=dict_data["file_path"],
            Origin_context=VernacBase.from_dict(dict_data["origin_context"]) 
                if dict_data.get("origin_context") else None,
            Internal_context=ConstrBase.from_dict(dict_data["internal_context"]) 
                if dict_data.get("internal_context") else None,
            local_vars=local_vars
        )
        
@dataclass
class def_table:
    ## jsonl format
    items: Dict[str, def_object]
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {name: item.to_dict() for name, item in self.items.items()}

"""
some other tables may need
"""

@dataclass
class coq_data_base_hint:
    pass

class coq_dep:
    pass