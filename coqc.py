import warnings
from utils import read_json_file
import subprocess
import asyncio
import time
import os
from utils import get_config
from datetime import datetime
import uuid
from typing import Optional, Dict, Tuple, List

UnsetLinear = 'Unset Linear Execution.'
SetLinear = 'Set Linear Execution.'

UnsetTacStr = 'Unset PrintTacticStr.'
SetTacStr = 'Set PrintTacticStr.'

def checkErrorUseful(error):
    ignore_patterns = [
        "Compiled library",
        "Cannot find a physical path bound to logical path matching suffix",
        "makes inconsistent assumptions",
        "error loading shared library:",
        "File not found",
        "Unable to locate library",
        "Error: Native compiler exited",
        "Error: Unknown interpretation for notation",
        "Error: This command does not support this attribute",
        "cmxs",
        "Uncaught exception Not_found"
    ]
    for ignore_pattern in ignore_patterns:
        if ignore_pattern in error:
            return False
    return True

def_start_keywords = [
    "Axiom",
    "Definition",
    "Fixpoint",
    "Lemma",
    "Theorem",
    "Instance",
    "Corollary",
    "Inductive",
    "Variable",
    "Parameter",
    "Notation",
    "Ltac"
]

def getInfoFromError(file_,error):
    info_part = error[:error.find("Error:")]
    file_path = info_part[:info_part.find(":\n")]
    line_part_start_index = info_part.rfind("line ")
    line_part_length = info_part[line_part_start_index:].find(", ")
    line_num = int(info_part[line_part_start_index+5:line_part_start_index + line_part_length])
    theorem_name = ""
    theorem_line_num = -1
    next_def_line_num = -1
    with open(file_, "r") as file:
        lines = file.readlines()
        index = line_num - 1
        while index >= 0:
            line = lines[index]
            for def_start_keyword in def_start_keywords:
                if (def_start_keyword + " ") in line:
                    theorem_name_start_index = line.find(def_start_keyword + " ") + len(def_start_keyword) + 1
                    theorem_name = line[theorem_name_start_index:].split(" ")[0]
                    theorem_line_num = index + 1
                    break
            if theorem_name != "":
                break
            index -= 1
        index = line_num - 1
        while index < len(lines):
            line = lines[index]
            for def_start_keyword in def_start_keywords:
                if (def_start_keyword + " ") in line:
                    next_def_line_num = index + 1
                    break
            if next_def_line_num != -1:
                break
            index += 1
    return {
        "path": file_path,
        "theorem_name": theorem_name,
        "theorem_line_num": theorem_line_num,
        "next_def_line_num": next_def_line_num
    }

def write_patch_file(file,error,patch_info,data_dir,patch_prefix):
    with open(file, "r") as f:
        file_content = f.read()
        lines = file_content.split("\n")
    lines_before = lines[:patch_info["theorem_line_num"] - 1]

    UnsetCommand = UnsetTacStr if 'LetIn must declare at least one binding' in error else UnsetLinear
    SetCommand = SetTacStr if 'LetIn must declare at least one binding' in error else SetLinear

    new_file_content = "\n".join(lines_before) + "\n" + UnsetCommand + "\n"
    if patch_info["next_def_line_num"] == -1:
        new_file_content += "\n".join(lines[patch_info["theorem_line_num"] - 1:])
    else:
        new_file_content += "\n".join(lines[patch_info["theorem_line_num"] - 1: patch_info["next_def_line_num"] - 1]) + "\n" + SetCommand + "\n" + "\n".join(lines[patch_info["next_def_line_num"] - 1:])
    patch_file = file.replace(data_dir, patch_prefix)
    patch_dir = os.path.dirname(patch_file)
    os.makedirs(patch_dir, exist_ok=True)
    with open(patch_file, "w") as f:
        f.write(new_file_content)
    with open(file, "w") as f:
        f.write(new_file_content)
        
def handle_error_patch(file,error,data_dir,patch_prefix):
    if checkErrorUseful(error):
        patch_info = getInfoFromError(file,error)
        write_patch_file(file,error,patch_info,data_dir,patch_prefix)
        return None
    else:
        return error


class Coqc:
    def __init__(self, config_path: str = None, mode: str = "data", new_theorem_mode: bool = False, max_coqc_workers: int = 200):
        self.max_coqc_workers = max_coqc_workers
        self.semaphore = asyncio.Semaphore(self.max_coqc_workers)
        if not (mode == "data" or mode == "proof" or mode == "coqtop"):
            raise ValueError("mode must be either 'data' or 'proof' or 'coqtop'")
        self.new_theorem_mode = new_theorem_mode
        self.mode = mode
        if config_path is None or not os.path.exists(config_path):
            raise ValueError("config_path must be a valid path to a config.json file")
        self.config(config_path)
        
        if self.mode == "data":
            self.succ = 0
            time = datetime.now().strftime("%Y-%m-%d-%H")
            self.current_log_file = os.path.join(self.log_file, f"coqc_error_log_{time}.txt")
        elif self.mode == "proof":
            if os.path.exists(self.temp_dir):
                self.temp_dir = f"{self.temp_dir}_{uuid.uuid4().hex}"
            os.makedirs(self.temp_dir, exist_ok=True)

    async def run(self, file_path: str, content: str = None, init_ps: bool = False, package_paths: List[str] = [], patch_mode: bool = False, timeout: int = 900):
        if self.mode == "data":
            if content is not None:
                warnings.warn("content is not None, but it is not used in data mode, so content will be ignored")
            return await self._run_coqc(file_path, patch_mode)
        elif self.mode == "proof":
            if content is None:
                raise ValueError("content is required")
            return await self._run_coqc_temp(content, file_path, init_ps, package_paths=package_paths, timeout=timeout)
        elif self.mode == "coqtop":
            if content is None:
                raise ValueError("content is required")
            return await self._run_coqtop(content, file_path)

    def _find_package_needed(self, stderr: str, file_path: str, mappings_only: bool = False):
        if 'Cannot find a physical path bound to logical path matching suffix' in stderr:
            package_needed = stderr.strip().split('\n')[-1].split('.')[0].strip()
            package_mapping = []
            for key,values in self.package_mapping.items():
                for value in values:
                    if package_needed in value:
                        package_mapping.append(key)
                        break
            
            if not package_mapping:
                raise ValueError(f"No package mapping found for {package_needed}")
            
            filtered_flags = []
            for path in package_mapping:
                if 'tactician' in path:
                    continue
                operator_list = self.package_mapping[path]
                for operator in operator_list:
                    if operator.startswith("-Q") or operator.startswith("-R"):
                        parts = operator.split()
                        path_part = path if parts[1] == '.' else f"{path}/{parts[1]}"
                        filtered_flags.extend([parts[0], path_part, parts[2]])
            if mappings_only:
                return filtered_flags
            else:
                return [self.coqc_path, *filtered_flags, file_path]
        else:
            return None

    async def _run_coqc(self, file_path: str, patch_mode: bool = False):
        ## TODO: now stable version of coqc, so all the cmxs files should be init
        ## need a full version of coqc and now the whole pipeline can be done
        if patch_mode:
            patch_file = file_path.replace(self.data_dir, self.patch_prefix)
            if os.path.exists(patch_file):
                with open(patch_file, "r") as pf, open(file_path, "w") as f:
                    f.write(pf.read())
            with open(file_path, "r") as f:
                original_content = f.read()

        output_file = file_path.replace(".v", ".txt")
        args = self._get_args(file_path)

        for _ in range(10):
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
                
            if process.returncode == 0:
                output_lines = stdout.decode().strip().split('\n')
                self.succ += 1
                print(self.succ)
                with open(output_file, "w") as output_file:
                    for line in output_lines:
                        output_file.write(line + "\n")
                return output_lines
            else:
                if 'Native compiler exited with status 2' in stderr.decode():
                    print('Native compiler error,start eval $(opam env)')
                    cmd = f"eval $(opam env) && {' '.join(args)}"
                    print(cmd)
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    if process.returncode == 0:
                        output_lines = stdout.decode().strip().split('\n')
                        self.succ += 1
                        print(self.succ)
                        with open(output_file, "w") as output_file:
                            for line in output_lines:
                                output_file.write(line + "\n")
                        return output_lines

                if not patch_mode:
                    with open(self.current_log_file, "a") as log_file:
                        error_msg = f"Error in file {file_path}:\nCommand: {' '.join(args)}\n{stderr.decode()}\n"
                        log_file.write(error_msg)
                    return []
                else:
                    patch_result = handle_error_patch(file_path, stderr.decode(), self.data_dir, self.patch_prefix)
                    if patch_result is not None:
                        with open(self.current_log_file, "a") as log_file:
                            error_msg = f"Error in file {file_path}:\nCommand: {' '.join(args)}\n{stderr.decode()}\n"
                            log_file.write(error_msg)
                        return []
                    print(f"patch {file_path} success")

        with open(self.current_log_file, "a") as log_file:
            with open(file_path, "w") as f:
                f.write(original_content)
            error_msg = f"Error in file {file_path}:\nCommand: {' '.join(args)}\n{stderr.decode()}\n"
            log_file.write(f"Patch mode failed after 10 attempts for {file_path}\n{error_msg}")

        return []
    
    async def _run_coqc_temp(self, content: str, theorem_path: str, init_ps: bool = False, package_paths: List[str] = [], timeout: int = 900) -> Tuple[int, str, str]:
        mappings = self._get_args(theorem_path, mappings_only=True, package_paths=package_paths)
        temp_ident, temp_file = self._get_temp_filename(theorem_path)

        origin_filename = os.path.basename(theorem_path).rsplit('.',1)[0]

        if self.new_theorem_mode:
            if init_ps:
                final_content = content.replace('Admitted.', 'idtac.')
            else:
                final_content = content
        else:
            final_content = content + "\nidtac." if init_ps else content
        
        if not 'coq-hott.8.11' in theorem_path:
            args = [self.coqc_path, *mappings, temp_file]
        else:
            args = [self.hoqc_path, *mappings, temp_file]

        actual_name = self._get_actual_name(theorem_path)
        
        try:
            with open(temp_file, 'w') as f:
                f.write(final_content)
            
            # try:
            #     process = await asyncio.create_subprocess_exec(
            #         *args,
            #         stdout=asyncio.subprocess.PIPE,
            #         stderr=asyncio.subprocess.PIPE
            #     )
            #     stdout, stderr = await process.communicate()
            #     return stdout.decode().replace(temp_ident, actual_name), stderr.decode(), temp_file
            # except Exception as e:
            #     print(final_content)
            #     raise e
            
            # async def run_process():
            #     async with self.semaphore:
            #         process = await asyncio.create_subprocess_exec(
            #             *args,
            #             stdout=asyncio.subprocess.PIPE,
            #             stderr=asyncio.subprocess.PIPE
            #         )
            #     return await process.communicate()
        
            try:
                time_start = time.time()
                ## here is a fallback, now our coqc have recursive bug, so we need to wait for 300 seconds
                # stdout, stderr = await asyncio.wait_for(run_process(), timeout=timeout)
                async with self.semaphore:
                    process = await asyncio.create_subprocess_exec(
                        *args,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                time_end = time.time()
                # print(f"Time taken: {time_end - time_start} seconds")
                if self.new_theorem_mode:
                    mapping = self._find_package_needed(stderr.decode(), theorem_path, mappings_only=True)
                    if mapping:
                        args = [self.coqc_path, *mapping, temp_file]
                        
                        process = await asyncio.wait_for(asyncio.create_subprocess_exec(
                            *args,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        ), timeout=timeout)
                        stdout, stderr = await process.communicate()
                if self.new_theorem_mode and init_ps:
                    return stdout.decode().replace(temp_ident, actual_name).replace(' ' + origin_filename + '.', ' ' + actual_name + '.').split('\n'), stderr.decode(), temp_file, final_content.replace('idtac.', '')
                
                return stdout.decode().replace(temp_ident, actual_name).replace(' ' + origin_filename + '.', ' ' + actual_name + '.'), stderr.decode(), temp_file
            
            except asyncio.TimeoutError:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        if process.returncode is None:
                            process.kill()
                
                if process.returncode is None:
                    await process.wait()

                print(f"timed out after {timeout} seconds")
                print('command: ', mappings)
                return "Timeout error", f"Timeout error", temp_file
            except Exception as e:
                with open(self.current_log_file, "a") as log_file:
                    log_file.write(f"Error in file {theorem_path}:\nCommand: {' '.join(args)}\n{e}\n")
                return "Timeout error unexpected", f"Timeout error unexpected", temp_file
                
        finally:
            self._cleanup(temp_file)

    async def _run_coqtop(self, content: str, theorem_path: str):
        mappings = self._get_args(theorem_path, mappings_only=True)
        args = [self.coqtop_path, *mappings]
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        if not content.endswith('\n'):
            content += '\n'
            
        stdout, stderr = await process.communicate(input=content.encode())
        return stdout.decode(), stderr.decode()
    
    def config(self, config_path: str):
        config = get_config(config_path)
        self.temp_dir = config.paths.temp_dir
        self.dep_map = read_json_file(config.paths.dep_file)
        self.coqc_path = config.paths.coqc_path
        self.hoqc_path = config.paths.hoqc_path
        self.coqtop_path = config.paths.coqtop_path
        self.data_dir = config.paths.data_dir
        self.patch_prefix = config.paths.patch_prefix
        self.package_mapping = read_json_file(config.paths.package_mapping)
        self.log_file = config.paths.coqc_error_log

    def _get_all_dependencies(self, file_path):
        dependence_set = set()
        processed_paths = set()

        def add_dependencies(file_path):
            if file_path in processed_paths:
                return
            processed_paths.add(file_path)
            dependencies = self.dep_map.get(file_path, [])
            if dependencies:
                dependence_set.add(dependencies[0].split('/')[0])

                for path in dependencies[1:]:
                    dep_key = path.split('/')[0]
                    dependence_set.add(dep_key)
                    full_path = path
                    add_dependencies(full_path)
        
        for file in self.dep_map:
            if file_path in file or file in file_path:
                file_path = file
                break
        
        add_dependencies(file_path)
        return dependence_set
    
    def _get_actual_name(self, file_path: str):
        file_path_list = file_path.split('/')
        indices = [i for i, part in enumerate(file_path_list) if part.startswith('coq-')]
        last_index = max(indices) if indices else -1

        if self.new_theorem_mode:
            file_name = file_path_list[-1]
            if file_name.endswith('.v'):
                file_name = file_name[:-2]
            return file_name

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

    def _get_args(self, file_path: str, mappings_only: bool = False, package_paths: List[str] = []):
        dependence_set = self._get_all_dependencies(file_path)
        if package_paths:
            for path in package_paths:
                dependence_set.update(self._get_all_dependencies(path))
        
        filtered_flags = []
        for path in dependence_set:
            if 'tactician' in path:
                continue
            operator_list = self.package_mapping[path]
            for operator in operator_list:
                if operator.startswith("-Q") or operator.startswith("-R"):
                    parts = operator.split()
                    path_part = path if parts[1] == '.' else f"{path}/{parts[1]}"
                    filtered_flags.extend([parts[0], os.path.join(self.data_dir, path_part), parts[2]])

        if mappings_only:
            return filtered_flags
        if "coq-hott.8.11" in file_path:
            if "coq-hott.8.11/coq/theories/Init/Nat" in file_path:
                args = [self.coqc_path ,file_path]
            elif 'coq-hott.8.11/coq/theories/Init/Peano' in file_path or 'coq-hott.8.11/coq/theories/Init/Prelude.v' in file_path:
                args = [self.hoqc_path, file_path]
            else:
                # args = ["/coq-hott.8.11/hoqc", *filtered_flags, file_name] ### hott need extra processing
                    args = [self.hoqc_path, *filtered_flags, file_path] ### hott need extra processing
        elif "tactician" in file_path: 
            return [self.coqc_path, file_path]
        #     print(file_path)
            ## Some bug can not be fixed so far
            # Cannot load a library with the same name as the current one.
            # some .cmxs bugs
            # # args = [self.coqc_path, "-noinit", *filtered_flags, file_path]
            # args = [self.coqc_path, *filtered_flags, file_path]

        else:
            args = [self.coqc_path, *filtered_flags, file_path]
        return args
    
    def _get_temp_filename(self, theorem_path: str) -> str:
        # some file need the original name
        if self.mode == "proof":
            temp = f"temp{uuid.uuid4().hex}"
            uuid_dir = os.path.join(self.temp_dir, temp)
            os.makedirs(uuid_dir, exist_ok=True)
            original_filename = os.path.basename(theorem_path)
            return temp, os.path.join(uuid_dir, original_filename)
        else:
            raise ValueError("_get_temp_filename should only be called in temporary mode")
    
    def _cleanup(self, temp_file: str) -> None:
        if self.mode == "data":
            return
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        base = os.path.splitext(temp_file)[0]
        for ext in ['.vo', '.glob', '.vok', '.vos', '.aux']:
            aux_file = base + ext
            if os.path.exists(aux_file):
                os.remove(aux_file)

        temp_dir = os.path.dirname(temp_file)
        if os.path.exists(temp_dir) and os.path.basename(temp_dir).startswith('temp'):
            try:
                os.rmdir(temp_dir)
            except OSError:
                import shutil
                shutil.rmtree(temp_dir)

    def __del__(self):
        if self.mode == "proof" and hasattr(self, 'temp_dir'):
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
                os.rmdir(self.temp_dir)