import os
import asyncio
import re
from utils import read_normal_file, read_jsonl_file
from data_extraction.coq_data.Parser import Parser
from coqc import Coqc
from typing import Union, List, Tuple, Dict
from data_extraction.coq_tokenize.tokenizer import Tokenizer
from coq_prover.coq_context.proof_data_class import *

'''
the functions "char_to_line", "get_prf_range" and "get_base_file_content" are based on code by @wangxiaodie
see kuibu/interactive-coqtop/-/blob/master/src/process_prfstate_info.py
'''

def char_to_line(coq_file_path: str, char_range: str):
    # 将输入的范围字符串转换为两个部分
    start_char_str, end_char_str = char_range.split('-')
    start_char = int(start_char_str)
    # 如果范围的结束是 'UNKNOWN'，将其设为文件总长度（字节数）
    with open(coq_file_path, 'rb') as file:  # 以二进制模式打开以获取精确字节数
        file.seek(0, 2)  # 移动到文件末尾
        file_length = file.tell()  # 获取文件总字节数

    if end_char_str == 'UNKNOWN':
        end_char = file_length
    else:
        end_char = int(end_char_str)

    # 检查范围是否合法
    if start_char < 0 or end_char > file_length:
        raise ValueError(
            f"coq_file: {coq_file_path} \n字符范围超出了文件内容的长度: start_char-{start_char}, "
            f"end_char-{end_char}, file_length-{file_length}"
        )
    
    # 逐行读取文件并计算字节偏移
    current_byte_offset = 0
    start_line = None
    end_line = None
    
    with open(coq_file_path, 'rb') as file:
        for line_number, line in enumerate(file, start=1):
            line_byte_length = len(line)  # 当前行的字节长度

            # 检查起始字符范围是否落在当前行
            if start_line is None and current_byte_offset + line_byte_length > start_char:
                start_line = line_number
            
            # 检查结束字符范围是否落在当前行
            if current_byte_offset + line_byte_length >= end_char:
                end_line = line_number
                break

            # 累计字节偏移
            current_byte_offset += line_byte_length

    # 如果 end_line 未找到，设置为最后一行
    if end_line is None:
        end_line = line_number

    return start_line, end_line

def get_prf_range(glob_file_path, prf_name):
    with open(glob_file_path, 'r', errors='ignore') as f:
        lines = f.readlines()
        total_lines = len(lines)
        i = 0
        proof_key_words = ['prf', 'var', 'def']
        while i < total_lines:
            line = lines[i].strip()
            if not any(line.startswith(keyword) for keyword in proof_key_words):
                i += 1
            else:
                # 判断是否匹配 prf 121:123格式
                match = re.search(r'(?:prf|var|def)\s+(\d+:\d+)', line)
                if match:
                    # 取出 prf_name，空格分隔的最后一个
                    cur_prf_name = line.split(" ")[-1]
                    
                    if prf_name != cur_prf_name:
                        i += 1
                        continue

                    # 取第一行，默认一定会取
                    start_number = int(match.group(1).split(':')[0])

                    end_number = None
                    j = i + 1
                    # 确定后续prf的范围
                    # 范围0，如果是prf 开头最好
                    # 范围1， 当前到下一个 非 binder 开头 & 非 大写 的元素
                    while j < total_lines:
                        next_line = lines[j].strip()
                        if next_line.startswith('prf') or next_line.startswith('var') or next_line.startswith('def'):
                            if end_number is None:
                                next_match = re.search(r'(?:prf|var|def)\s+(\d+:\d+)', next_line)
                                if next_match:
                                    end_number = int(next_match.group(1).split(':')[0])
                            i = j
                            break
                        elif next_line[0].islower() and not next_line.startswith('binder'):
                            lower_match = re.search(r'\s+(\d+:\d+)', next_line)
                            if lower_match:
                                end_number = int(lower_match.group(1).split(':')[0])
                                i = j
                                break  # 找到第一个符合条件的行就停止搜索
                        j += 1
                    if end_number is not None:
                        prf_range =  f"{start_number}-{end_number}"
                    else:
                        # 没找到end_number时，存最后一行
                        prf_range = f"{start_number}-UNKNOWN"
                    
                    print(f"prf_range: {prf_range}")
                    return prf_range
    
    print("error: prf_name not found!")
    return ""

'''
current_file: path to the current file, example: "./data/coq/coq_train/coq-tactician-stdlib.8.11.dev/theories/NArith/BinNat.v"
current_name: name of theorem proving, example: "Coq.NArith.BinNat.N.recursion_succ"
'''
def get_base_file_content(current_file: str, current_name: str):
    glob_path = current_file.replace(".v", ".glob")
    if not os.path.exists(glob_path):
        raise ValueError(f"glob file not existing: {glob_path}")
    
    prf_range = get_prf_range(glob_path, current_name.split(".")[-1])
    print(f"prf_range: {prf_range}")
    
    if prf_range == "":
        raise ValueError("error: prf_range not found!")
    
    with open(current_file, 'r', errors='ignore') as f:
        origin_content = f.read()
        lines = origin_content.splitlines()
        start_line, end_line = char_to_line(current_file, prf_range)
        if start_line:
            if end_line == len(lines):
                cur_proof_contents = lines[int(start_line) - 1: int(end_line)]
            else:
                cur_proof_contents = lines[int(start_line) - 1: int(end_line)-1]
            if start_line != end_line:
                cur_proof_contents_str = " ".join(cur_proof_contents)
                cur_proof_start = cur_proof_contents_str[: cur_proof_contents_str.index(". ") + 2] + "Proof."
            else:
                current_line = lines[int(start_line)]
                cur_proof_start = current_line.split("Proof.")[0]
            return "\n".join(lines[:int(start_line) - 1]) + "\n" + cur_proof_start + "\n" + "Unset Linear Execution.\n"
            
    raise ValueError("error: start_line in None!")

class TacticRunner:
    def __init__(self, coqc_share, config_path, tokenizer, new_theorem_mode=False, def_table=None, current_file=None, current_name=None, coqtop_mode=False, log_prefix: str = None, base_file_content: str = None):
        self.coqc = coqc_share
        if coqtop_mode:
            self.coqtop = Coqc(config_path=config_path, mode="coqtop")
        self.new_theorem_mode = new_theorem_mode
        self.parser = Parser()
        self.tokenizer = tokenizer

        self.error_debug_file = log_prefix + '/error_debug.txt'
        self.debug_success_file = log_prefix + '/debug_success.txt'

        self.base_file_content = base_file_content

        self.package_paths = []

        if def_table:
            if isinstance(def_table, dict):
                ## here def_table is a dict key should be name, so need to convert
                self.def_table = def_table
            elif isinstance(def_table, str):
                def_table = read_jsonl_file(def_table)
                self.def_table = {item['name']: item for item in def_table}
            else:
                raise ValueError("def_table must be a path to a jsonl file or a dictionary")
        if current_file and current_name:
            self.init_base_file_content(current_file, current_name)

    def init_base_file_content(self, current_file, current_name):
        self.current_file = current_file
        self.current_name = current_name
        if not self.new_theorem_mode:
            self.base_file_content = get_base_file_content(current_file, current_name)

    '''
    description:
        run tactic with coqc based on current state
    input:
        tactics: the tactic / list of tactics to run
        previous_result: result from previous step, (ps_item: PSItem, [tactic])
    output:
        success_results: [(ps_item: PSItem, [tactic])]
        failed_results: [ps_item: PSItem, [tactic], error]
    '''

    async def run(self, tactics: Union[List[Dict[str, str]], Dict[str, str], List[str], str], previous_result: PreviousResult = None, tactic_trace: List[TacticTrace_Single] = [], refresh_mode: bool = False, if_gathered: bool = True):
        helpful_error = 'Attempt to save an incomplete proof'
        if not isinstance(tactics, list):
            tactics = [tactics]
        
        if not tactics:
            return TacticResultGroup(status=TacticStatus.FAIL, tactic_results=[])

        results = []
        if previous_result:
            current_tactic = previous_result.tactic_sequence
            ps_steps = previous_result.ps_item.get_proof_steps()
            tactic_str = ''
            for tactic in current_tactic:
                tactic_str += tactic + ' ' if tactic.endswith('.') else tactic + '. '
        else:
            current_tactic = None
            tactic_str = ''
            ps_steps = 0

        if refresh_mode:
            return await self.run_refresh(tactic_str + '\n' + 'idtac.')

        ## TODO: actually here is an rough implementation, to prove Nat.succ P.succ may be useful
        ## however, apply P.succ (here current_name is succ) is not allowed
        ## maybe check the fully qualified name? 
        
        tactic_list = []
        reason_list = []
        for tactic in tactics:
            if isinstance(tactic, dict):
                if 'tactic' in tactic:
                    tactic_cur = tactic['tactic'].strip()
                    if tactic_cur.endswith(';'):
                        tactic_cur = tactic_cur[:-1]+'.'
                    tactic_list.append(tactic_cur)
                    reason_list.append(tactic['reason'])
                elif 'refined_tactic' in tactic:
                    tactic_cur = tactic['refined_tactic'].strip()
                    if tactic_cur.endswith(';'):
                        tactic_cur = tactic_cur[:-1]+'.'
                    tactic_list.append(tactic_cur)
                    reason_list.append(tactic['reason'])
            elif isinstance(tactic, str):
                tactic_list.append(tactic)
                reason_list.append('')
            else:
                raise ValueError(f"tactic must be a string or a dictionary with 'tactic' or 'refined_tactic' key, but got {type(tactic)}")

        assert len(tactic_list) == len(reason_list)

        tactics_ = [tactic for tactic in tactic_list if self.current_name not in tactic and "repeat" not in tactic]
        
        async def process_tactic(tactic, reason):
            origin_tactic = tactic
            tactic_module = None
            if any('.' in token for token in tactic.split()):
                tactic_copy = tactic
                
                tactic_part = []
                for token in tactic.split():
                    token = token.strip()
                    if '.' in token:
                        if token.endswith('.'):
                            if '.' in token[:-1]:
                                if token.startswith('('):
                                    tactic_part.append('(' + token[:-1].split('.')[-1] + '.')
                                elif token.startswith('['):
                                    tactic_part.append('[' + token[:-1].split('.')[-1] + '.')
                                else:
                                    tactic_part.append(token[:-1].split('.')[-1] + '.')
                            else:
                                tactic_part.append(token)
                        else:
                            if token.startswith('('):
                                tactic_part.append('(' + token.split('.')[-1])
                            elif token.startswith('['):
                                tactic_part.append('[' + token.split('.')[-1])
                            else:
                                tactic_part.append(token.split('.')[-1])
                    else:
                        tactic_part.append(token)
                tactic = ' '.join(tactic_part)

                tactic_module_part = []
                for token in tactic_copy.split():
                    token = token.strip()
                    if '.' in token:
                        if token.endswith('.'):
                            if '.' in token[:-1]:
                                if token.startswith('('):
                                    tactic_module_part.append('(' + '.'.join(token[:-1].split('.')[-2:]) + '.')
                                elif token.startswith('['):
                                    tactic_module_part.append('[' + '.'.join(token[:-1].split('.')[-2:]) + '.')
                                else:
                                    tactic_module_part.append('.'.join(token[:-1].split('.')[-2:]) + '.')
                            else:
                                tactic_module_part.append(token)
                        else:
                            if token.startswith('('):
                                tactic_module_part.append('(' + '.'.join(token.split('.')[-2:]))
                            elif token.startswith('['):
                                tactic_module_part.append('[' + '.'.join(token.split('.')[-2:]))
                            else:
                                tactic_module_part.append('.'.join(token.split('.')[-2:]))
                    else:
                        tactic_module_part.append(token)
                tactic_module = ' '.join(tactic_module_part)

            com_tactic = tactic_str + ' ' + tactic if tactic.endswith('.') else tactic_str + ' ' + tactic + '.'
            actual_tactic = self.base_file_content + '\n' + com_tactic + '\nQed.'
            if tactic_module:
                com_tactic_module = tactic_str + ' ' + tactic_module if tactic_module.endswith('.') else tactic_str + ' ' + tactic_module + '.'
                actual_tactic_mudule = self.base_file_content + '\n' + com_tactic_module + '\nQed.'
            (output, error, temp_file) = await self.coqc.run(self.current_file, actual_tactic, package_paths=self.package_paths)
            # print(output)
            # print('=======================')
            if not helpful_error in error and tactic_module:
                (output_mod, error_mod, temp_file_mod) = await self.coqc.run(self.current_file, actual_tactic_mudule, package_paths=self.package_paths)
                if helpful_error in error_mod:
                    tactic = tactic_module
                    actual_tactic = actual_tactic_mudule
                    output = output_mod
                    error = error_mod
                    temp_file = temp_file_mod

            if ('not found in the current environment' in error or 'Nametab.GlobalizationError' in error):
                token_list = self.tokenizer.encode(origin_tactic, return_global=True)
                formal_names = self.tokenizer.decode(token_list).split()

                require_packages = []
                import_prefix = 'Require Import'
                for formal_name in formal_names:
                    def_ = self.def_table[formal_name]
                    file_path = def_['file_path']
                    if 'hott' not in file_path and 'stdlib' not in file_path and 'tactician' not in file_path:
                        self.package_paths.append(file_path)
                    require_packages.append(import_prefix + ' ' + file_path.split('/')[-1].split('.')[0] + '.')
                require_packages = list(set(require_packages))
                package_str = '\n'.join(require_packages)
                tactic_package = package_str + '\n' + tactic if tactic.endswith('.') else package_str + '\n' + tactic + '.'
                com_tactic = tactic_str + '\n' + tactic_package
                actual_tactic = self.base_file_content + '\n' + com_tactic + '\nQed.'
                (output, error, temp_file) = await self.coqc.run(self.current_file, actual_tactic, package_paths=self.package_paths)
                if 'Cannot find a physical path bound to logical path' in error:
                    with open(self.error_debug_file, 'a') as f:
                        f.write('==================\n')
                        f.write(f"current_theorem: {self.current_name}\n")
                        f.write(f"actual_tactic: {tactic}\n")
                        f.write(f"tactic_package: {tactic_package}\n")
                        f.write(f"origin_tactic: {origin_tactic}\n")
                        f.write(f"com_tactic: {com_tactic}\n\n")
                        f.write(f"package_paths: {self.package_paths}\n")
                        f.write(f"error: {error}\n")
                if helpful_error in error:
                    tactic = tactic_package
            
            ## some error should be ignored
            if not output:
                return None
            
            if "Error" in error:
                # if 'Syntax error' in error:
                #     print('==================')
                #     print('current_theorem',self.current_name)
                #     print('response tactics',tactics)
                #     print('processed_tactic',tactic)
                #     print('origin_tactic',origin_tactic)
                #     print(f"Syntax error: {error}")
                #     print('com_tactic',com_tactic)
                error = error.split("Error:",1)[1].strip()
            elif 'Timeout error' in error:
                with open(self.error_debug_file, 'a') as f:
                    f.write(f"error: {error}\n")
                    f.write(f"current_theorem: {self.current_name}\n")
                    f.write(f"actual_tactic: {tactic}\n")
                    f.write(f"origin_tactic: {origin_tactic}\n")
                    f.write(f"com_tactic: {com_tactic}\n\n")
                return None
            # else:
            #     with open(self.error_debug_file, 'a') as f:
            #         f.write(f"Invalid error: {error}\n")
            #         f.write(f"error: {error}\n")
            #         f.write(f"current_theorem: {self.current_name}\n")
            #         f.write(f"actual_tactic: {tactic}\n")
            #         f.write(f"origin_tactic: {origin_tactic}\n")
            #         f.write(f"com_tactic: {com_tactic}\n\n")
            #     return None
                # raise ValueError(f"Invalid error: {error}")

            ps, actual_name, if_end, type_dict = self.parser.parse_proof(output, self.current_file, self.current_name, use_tqdm=False, new_theorem_mode=self.new_theorem_mode)
            
            if not ps.Content:
                return TacticResult(status=TacticStatus.FAIL, ps_item=ps, tactic_sequence=current_tactic + [tactic] if current_tactic else [tactic], error_message=error, tactic_trace=tactic_trace + [TacticTrace_Single(tactic=tactic, error_message=error, reason=reason)])
            
            # success but ps not changed when init, skip
            # maybe failed when init as well
            if not ps.Content.ProofStates:
                if helpful_error in error:
                    error = 'Error: tactic did not change proof state'
                    return TacticResult(status=TacticStatus.FAIL, ps_item=ps, tactic_sequence=current_tactic + [tactic] if current_tactic else [tactic], error_message=error, tactic_trace=tactic_trace + [TacticTrace_Single(tactic=tactic, error_message=error, reason=reason)])
                return TacticResult(status=TacticStatus.FAIL, ps_item=ps, tactic_sequence=current_tactic + [tactic] if current_tactic else [tactic], error_message=error, tactic_trace=tactic_trace + [TacticTrace_Single(tactic=tactic, error_message=error, reason=reason)])
        
            ps_item = self.tokenizer.process_ps_proof(ps, def_table=None, type_dict=type_dict, actual_name=actual_name, txt_file_path=temp_file, if_refined_ps=False)
            
            # when exactly create new proofstate, it is a success
            # for ltac expansion, it may create more than one proofstate
            if ps_item.get_proof_steps():
                if ps_item.get_proof_steps() > ps_steps and ps_item.get_proof_state().Tactic.Name != 'idtac':
                    if if_end:
                        with open(self.debug_success_file, 'a') as f:
                            curr_tac = current_tactic + [tactic] if current_tactic else [tactic]
                            f.write(f"SUCCEEDED: {self.current_name} was proved successfully\n")
                            f.write(f"tactic: {'. '.join(curr_tac)}\n\n")
                        return TacticResult(status=TacticStatus.COMPLETED, ps_item=ps_item, tactic_sequence=current_tactic + [tactic] if current_tactic else [tactic], tactic_trace=tactic_trace + [TacticTrace_Single(tactic=tactic, reason=reason)])
                    else:
                        if ps_item.get_proof_state().After_state[0].goal.processed.Origin == 'goalcompleted':
                            return TacticResult(status=TacticStatus.SUBCOM, ps_item=ps_item, tactic_sequence=current_tactic + [tactic] if current_tactic else [tactic], tactic_trace=tactic_trace + [TacticTrace_Single(tactic=tactic, reason=reason)])
                        else:
                            return TacticResult(status=TacticStatus.SUCCESS, ps_item=ps_item, tactic_sequence=current_tactic + [tactic] if current_tactic else [tactic], tactic_trace=tactic_trace + [TacticTrace_Single(tactic=tactic, reason=reason)])
                else:
                    if ps_item.get_proof_steps() == ps_steps and helpful_error in error:
                        error = 'Error: tactic did not change proof state'
                        return TacticResult(status=TacticStatus.FAIL, ps_item=ps_item, tactic_sequence=current_tactic + [tactic] if current_tactic else [tactic], error_message=error, tactic_trace=tactic_trace + [TacticTrace_Single(tactic=tactic, error_message=error, reason=reason)])
                    else:
                        return TacticResult(status=TacticStatus.FAIL, ps_item=ps_item, tactic_sequence=current_tactic + [tactic] if current_tactic else [tactic], error_message=error, tactic_trace=tactic_trace + [TacticTrace_Single(tactic=tactic, error_message=error, reason=reason)])
            
            with open(self.error_debug_file, 'a') as f:
                f.write(f"do not match any pattern in tactic runner\n")
                f.write(f"error: {error}\n")
                f.write(f"current_theorem: {self.current_name}\n")
                f.write(f"actual_tactic: {tactic}\n")
                f.write(f"origin_tactic: {origin_tactic}\n")
                f.write(f"com_tactic: {com_tactic}\n\n")
            return None
    
        tasks = [process_tactic(tactic, reason) for tactic, reason in zip(tactics_, reason_list)]
        results = await asyncio.gather(*tasks)

        if not if_gathered:
            return [result for result in results if result is not None]
        tactic_result_group = self.handle_tactic_runner_result(results)
        return tactic_result_group
    
    def handle_tactic_runner_result(self, results: List[TacticResult]):
        all_success = True
        for idx, result in enumerate(results):
            if result is None:
                continue
            ## completed and subcompleted will not appear at the same time
            ## and for the same group, only one of them will be used
            if result.status == TacticStatus.COMPLETED:
                return TacticResultGroup(status=TacticStatus.COMPLETED, completed_results=result, tactic_results=[result for i, result in enumerate(results) if (i != idx and result is not None)])
            elif result.status == TacticStatus.SUBCOM:
                return TacticResultGroup(status=TacticStatus.SUBCOM, subcompleted_results=result, tactic_results=[result for i, result in enumerate(results) if (i != idx and result is not None)])
            elif result.status == TacticStatus.FAIL:
                all_success = False
        return TacticResultGroup(tactic_results=[result for result in results if result is not None], all_success=all_success)

    async def run_refresh(self, tactics: str):
        pending_error = 'There are pending proofs'
        content = self.base_file_content + '\n' + tactics
        output, error, temp_file = await self.coqc.run(self.current_file, content, package_paths=self.package_paths)
        if "Timeout error" in error:
            output, error, temp_file = await self.coqc.run(self.current_file, content, package_paths=self.package_paths, timeout=3000)
        if not pending_error in error:
            # raise ValueError(f"Invalid error: {error}")
            with open(self.error_debug_file, 'a') as f:
                f.write('refresh error\n')
                f.write(f"error: {error.strip()}\n")
                f.write(f"current_theorem: {self.current_name}\n")
                f.write(f"tactics: {tactics}\n\n")
            return None
        ps, actual_name, _, type_dict = self.parser.parse_proof(output, self.current_file, self.current_name, use_tqdm=False, new_theorem_mode=self.new_theorem_mode)
        ps_item = self.tokenizer.process_ps_proof(ps, def_table=None, type_dict=type_dict, actual_name=actual_name, txt_file_path=temp_file, if_refined_ps=False)
        assert ps_item.Content.ProofStates[-1].Tactic.Name == 'idtac'
        assert len(ps_item.Content.ProofStates[-1].After_state) == 1
        return ps_item.Content.ProofStates[-1].After_state

    async def run_tactic_coqtop(self, tactics: List[str]):
        tactics_str = ' '
        for tactic in tactics:
            tactics_str += tactic + ' ' if tactic.endswith('.') else tactic + '. '
    
        # need idtac to fresh the print logic
        content = self.base_file_content + '\n' + tactics_str + '\n' + 'idtac.' + '\n' + 'Show.'
        output, error = await self.coqtop.run(self.current_file, content)
        # print(output)
        # print(error)
        
        if 'Error' in error:
            error = error.split("Error:",1)[1].strip().split('\n')[0]
        else:
            error = None

        output_lines = output.split('\n')
        ps_index = None
        for i in range(len(output_lines) - 1, -1, -1):
            if output_lines[i].startswith("PS"):
                ps_index = i
                break
        if ps_index is not None:
            result_lines = output_lines[ps_index+1:]
            result = '\n'.join(result_lines)
            if error is not None:
                return None, error
            else:
                return result, None
        else:
            raise ValueError("error: ps not found")
        

        