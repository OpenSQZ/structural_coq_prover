import warnings
from .Def_class import *
from .Ps_class import *
from .Definition import *
from .Proof import *
from .Fixpoint import *
from .Inductive import *
from .Instance import *
from .Primitive import *
from .Assumption import *
from .Ltac import *
from .Context import *
from .CoFixpoint import *
from .Arguments import *
from .TacticNotation import *
from typing import List, Optional, Union, Dict, Any, Tuple
import json
from utils import read_jsonl_file, read_json_file
import os
from tqdm import tqdm

class Parser:
    def __init__(self):
        self.type_lines: Optional[List[List[str]]] = []
        self.package_mapping = read_json_file('./data/package_mapping.json')

    def get_file_prefix(self, file_path: str) -> str:
        file_path_list = file_path.split('/')

        indices = [i for i, part in enumerate(file_path_list) if part.startswith('coq-')]
        last_index = max(indices) if indices else -1

        if last_index == -1:
            raise ValueError(f"Invalid file path: {file_path}")

        for item in self.package_mapping.keys():
            if file_path_list[last_index] in item:
                prefix = self.package_mapping[item]
                break
        
        prefix_list = []
        for item in prefix:
            if item.startswith('-R') or item.startswith('-Q'):
                part = item.split(' ')[1]
                alias = item.split(' ')[2]
                if not alias:
                    raise ValueError(f"No alias found for {item}")
                if part == '.':
                    prefix_list.append((file_path_list[last_index], alias))
                else:
                    prefix_list.append((part, alias))
        
        if not prefix_list:
            raise ValueError(f"No prefix found for {file_path}")
        
        for (item, alias) in prefix_list:
            if '/' not in item:
                if item + '/' in file_path:
                    name = alias + '.' + file_path.split(item + '/',1)[1].replace('/', '.')
                    break
            else:
                if item in file_path:
                    name = alias + '.' + file_path.split(item + '/',1)[1].replace('/', '.')
                    break
        
        if name.endswith('.v'):
            name = name[:-2]
        
        return name
    
    def parse_proof(self, file: str, file_path: str, ps_name: str, max_depth: float = float('inf'), use_tqdm: bool = True, init_ps: bool = False, new_theorem_mode: bool = False) -> 'ps_item':
        """
        TODO
        ps_name now can only processed single theorem name
        Some thing like Coq.NArith.BinNat.N.recursion_succ or N.recursion_succ should be considered
        """
        if new_theorem_mode:
            actual_name = ps_name
        else:
            name = self.get_file_prefix(file_path)
            if '.' not in ps_name:
                warnings.warn(f"not complete ps_name: {ps_name}, may not be handled accurately.")
            
            temp_name = name.split('.')[-1]
            if temp_name in ps_name:
                ps_name = ps_name.split(temp_name + '.')[-1]
            
            actual_name = name + '.' + ps_name

        if isinstance(file, list):
            lines = file
        elif os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        elif isinstance(file, str):
            lines = file.split('\n')
        else:
            raise ValueError(f"Invalid file type: {type(file)}")
        
        vernac_items = []
        constr_items = []
        ps_items = []
        type_items = []

        current_block_lines = []
        in_block = False
        current_type = None # "Vernac", "Constr", "Type", or "ProofState"
        in_ps_block = False # PS block may contain other blocks

        for line in tqdm(lines, desc="Processing", total=len(lines)) if use_tqdm else lines:    
            line = line.strip()
            if not line:
                continue
            
            if any(line.startswith(prefix) for prefix in DATA_TYPES_IDENTIFIER) and ("Begin" in line or "End" in line):
                if "Begin" in line and not in_ps_block:
                    if line.startswith("ProofState"):
                        in_ps_block = True
                        current_ps_name = line.split('proof of')[1].strip()

                    current_type = line.split()[0]
                    current_block_lines = [line]
                    in_block = True
                elif "End" in line and in_block and line.startswith(current_type):
                    if current_type == "ProofState":
                        if not current_ps_name == line.split('proof of')[1].strip():
                            continue

                    current_block_lines.append(line)
                    
                    if current_type == "Vernac":
                        try:
                            vernac_item, constructor_vernacs = VernacBase.create_from_lines(current_block_lines)
                            # add constructor first, then Ind itself
                            # for tokenizer, Ind will contain constructor 
                            vernac_items.extend(constructor_vernacs)
                            vernac_items.append(vernac_item)
                        except:
                            vernac_item = VernacBase.create_from_lines(current_block_lines)
                            vernac_items.append(vernac_item)
                    elif current_type == "Constr":
                        try:
                            constr_item, constructor_constrs = ConstrBase.create_from_lines(current_block_lines)
                            constr_items.extend(constructor_constrs)
                            constr_items.append(constr_item)
                        except:
                            constr_item = ConstrBase.create_from_lines(current_block_lines)
                            constr_items.append(constr_item)
                        if self.type_lines:
                            for type_block in self.type_lines:
                                item = TypeItem.from_string(type_block, constr_item.name)
                                type_items.append(item)
                            self.type_lines = []
                    elif current_type == "Type":
                        self.type_lines.append(current_block_lines)
                    elif current_type == "ProofState":
                        item = PSItem.from_string(current_block_lines, max_depth)
                        ps_items.append(item)
                        in_ps_block = False 
                        current_ps_name = None
                    in_block = False
                    current_block_lines = []
                    current_type = None
            
            elif in_block:
                current_block_lines.append(line)
                
        current_item = None

        if in_ps_block and current_block_lines:
            current_item = PSItem.from_string(current_block_lines, max_depth)
        
        def get_type_item(type_items, actual_name):
            type_dict = {}
            for item in reversed(type_items):
                if item.parent_constr.split('.')[-1] == actual_name.split('.')[-1]:
                    type_dict[item.name] = item
                else:
                    break
            if len(type_dict) > 0:
                return type_dict
            else:
                return {}
            
        if init_ps:
            def_table_obj = self.make_def_object(file_path, (vernac_items, constr_items, type_items))
            assert len(ps_items) == 0
            return def_table_obj, current_item, actual_name, get_type_item(type_items, actual_name)

        if current_item:
            return current_item, actual_name, False, get_type_item(type_items, actual_name)
        elif ps_items[-1].Name.split('.')[-1] == actual_name.split('.')[-1]:
            return ps_items[-1], actual_name, True, get_type_item(type_items, actual_name)
        else:
            raise ValueError("No PSItem found")


    def parse(self, file: str, file_path: str, max_depth: float = float('inf'), use_tqdm: bool = True) -> Tuple[def_table, ps_table]:
        """
        TODO
        Now, types in proofstate are not considered, maybe add as local vars?
        """
        if isinstance(file, list):
            lines = file
        elif os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            raise ValueError(f"Invalid file type: {type(file)}")
        
        vernac_items = []
        constr_items = []
        ps_items = []
        type_items = []

        current_block_lines = []
        in_block = False
        current_type = None  # "Vernac", "Constr", "Type", or "ProofState"
        in_ps_block = False # PS block may contain other blocks

        for line in tqdm(lines, desc="Processing", total=len(lines)) if use_tqdm else lines:    
            line = line.strip()
            if not line:
                continue
            
            if any(line.startswith(prefix) for prefix in DATA_TYPES_IDENTIFIER) and ("Begin" in line or "End" in line):
                if "Begin" in line and not in_ps_block:
                    if line.startswith("ProofState"):
                        in_ps_block = True
                        current_ps_name = line.split('proof of')[1].strip()

                    current_type = line.split()[0]
                    current_block_lines = [line]
                    in_block = True
                elif "End" in line and in_block and line.startswith(current_type):
                    if current_type == "ProofState":
                        if not current_ps_name == line.split('proof of')[1].strip():
                            continue

                    current_block_lines.append(line)
                    
                    if current_type == "Vernac":
                        try:
                            vernac_item, constructor_vernacs = VernacBase.create_from_lines(current_block_lines)
                            # add constructor first, then Ind itself
                            # for tokenizer, Ind will contain constructor 
                            vernac_items.extend(constructor_vernacs)
                            vernac_items.append(vernac_item)
                        except:
                            vernac_item = VernacBase.create_from_lines(current_block_lines)
                            vernac_items.append(vernac_item)
                    elif current_type == "Constr":
                        try:
                            constr_item, constructor_constrs = ConstrBase.create_from_lines(current_block_lines)
                            constr_items.extend(constructor_constrs)
                            constr_items.append(constr_item)
                        except:
                            constr_item = ConstrBase.create_from_lines(current_block_lines)
                            constr_items.append(constr_item)
                        if self.type_lines:
                            for type_block in self.type_lines:
                                item = TypeItem.from_string(type_block, constr_item.name)
                                type_items.append(item)
                            self.type_lines = []
                    elif current_type == "Type":
                        self.type_lines.append(current_block_lines)
                    elif current_type == "ProofState":
                        item = PSItem.from_string(current_block_lines, max_depth)
                        ps_items.append(item)
                        in_ps_block = False 
                        current_ps_name = None

                    in_block = False
                    current_block_lines = []
                    current_type = None
            
            elif in_block:
                current_block_lines.append(line)

        def_table_obj = self.make_def_object(file_path, (vernac_items, constr_items, type_items))
        ps_table_obj = self.make_ps_item(file_path, (vernac_items, constr_items, ps_items, type_items))
        return (def_table_obj,ps_table_obj)
    
    def make_def_object(self, file_path: str, items: Tuple[List[VernacBase], List[ConstrBase], List[TypeItem]]) -> List[def_object]:
        vernac_items, constr_items, type_items = items
        def_objects = []

        vernac_map = {item.name: item for item in vernac_items}
        constr_map = {item.name: item for item in constr_items}

        constr_to_types = {}
        # Ltac do not have constr representation
        for type_item in type_items:
            if type_item.parent_constr:
                if type_item.parent_constr not in constr_to_types:
                    constr_to_types[type_item.parent_constr] = {}
                constr_to_types[type_item.parent_constr][type_item.name] = type_item

        for name, constr_item in constr_map.items():
            Origin_context = vernac_map.get(name)

            # we do not need primitive so far
            if constr_item.category == "Primitive":
                continue

            # for constructor, the type is the type of its inductive type
            if constr_item.category == "Constructor":
                type_dict = constr_to_types.get(name.rsplit(".", 1)[0], {})
            else:
                type_dict = constr_to_types.get(name, {})

            def_objects.append(def_object(
                Name=name,
                Kind=constr_item.category,
                Type=type_dict,
                File_path=file_path,
                Origin_context=Origin_context,
                Internal_context=constr_item
            ))

        
        """
        may not need? 
        TODO
        Primitive may not need to be added to def_objects
        however arguments is necessary, while the vernac representation will cause error

        last version, use vernac representation to match constr. so some bug would occur.
        now use constr directly, only ltac should be considered as it do not have constr representation
        1.20 2025, add TacticNotation to def_objects do not have constr representation as well (a special case of ltac, add it to def_objects as Ltac is okay)
        """

        # for name, vernac_item in vernac_map.items():
        # use vernac_item directly, tactic notation is not fully qualified name
        for vernac_item in vernac_items:
            if vernac_item.category == 'Ltac' or vernac_item.category == 'TacticNotation':
                def_objects.append(def_object(
                    Name=vernac_item.name.split('.')[-1],
                    # Kind=vernac_item.category,
                    Kind='Ltac',
                    Type={},
                    File_path=file_path,
                    Origin_context=vernac_item,
                    Internal_context=None
                ))

        return def_objects


    def make_ps_item(self, file_path: str, items: Tuple[List[VernacBase], List[ConstrBase], List[PSItem], List[TypeItem]]) -> List[ps_object]:
        vernac_items, constr_items, ps_items, type_items = items
        # ps_objects = {item.Name: item for item in ps_items}
        return ps_items