import os
from coq_parser import customized
from coqc import Coqc
from utils import get_config, read_json_file
from coq_tokenization import run_tokenizer
from data_arg_infer import data_argumentation_async
from embedding_infer import process_whole_file
import asyncio
from coq_parser import Parser
from coq_tokenization import Tokenizer
from datetime import datetime
import json
from config_helper import ConfigHelper

async def main(args):
    config_path = "./config.json"
    config = get_config(config_path)

    config_helper = ConfigHelper(config_path)
    config_helper.validate4data_generation()
    
    if args.mode == 'data_generation':
        # for the first data_generation, patch_mode is recommended
        # patch mode will automatically fix the coqc error and generate a patch file
        # run ./data/patch.sh to apply the patch
        # patch_mode = True
        # current_indent = customized(patch_mode)
        # subprocess.run(["./data_extraction/patch.sh"])
        
        current_indent = await customized()
        def_output_path, ps_output_path, tokenizer_path = run_tokenizer(current_indent)
        arged_def_output_path = await data_argumentation_async(def_output_path)
        emb_def_output_path = process_whole_file(arged_def_output_path, config.paths.emb_model_path)
        print(f"Data generation completed. Output files:")
        print(f"  - {def_output_path}")
        print(f"  - {ps_output_path}")
        print(f"  - {arged_def_output_path}")
        print(f"  - {emb_def_output_path}")

        with open('./config.json', 'r') as f:
            config = json.load(f)
        config['paths']['def_table_path'] = def_output_path
        config['paths']['ps_table_path'] = ps_output_path
        config['paths']['emb_data_path'] = emb_def_output_path
        config['paths']['tokenizer_path'] = tokenizer_path
        with open('./config.json', 'w') as f:
            json.dump(config, f)
        print(f"Config file updated with new paths.")

    elif args.mode == 'new_theorem':
        ## for new premises only, if new theorem is only a proof, it will not be added to the dataset
        
        coqc = Coqc(config_path = config_path)
        parser = Parser()
        if not (os.path.exists(config.paths.def_table_path) and os.path.exists(config.paths.ps_table_path)):
            raise FileNotFoundError("Def table or PS table not found. Please run data_generation first.")

        with open(config.paths.theorem2proof_file, 'r') as f:
            theorem_list = f.readlines()

        ## the dependency of the theorem should be added manually to all_deps.json
        ## TODO: coqdep can be used to get the dependency of the theorem

        def_table = []
        tokenizer = Tokenizer(config.paths.tokenizer_path)

        for theorem in theorem_list:
            result = await coqc.run(theorem, patch_mode=False)
            if result is not None:
                def_table_obj, _ = parser.parse(file = result, file_path = theorem, max_depth = 1, use_tqdm = False)
                def_table.extend(def_table_obj)
                tokenizer.add_global_tokens(def_table_obj)

        current_time = datetime.now().strftime("%Y-%m-%d-%H")
        tokenizer.save(tokenizer_path.rsplit('_',1)[0] + f'_{current_time}.json')

        def_table_arged = await data_argumentation_async(def_table)
        def_table_embed = process_whole_file(def_table_arged)

        pre_def_table = read_json_file(config.paths.def_table_path)
        predef_arged_table = read_json_file(config.paths.def_table_path.replace('.jsonl', 'arged.jsonl'))
        predef_emb_table = read_json_file(config.paths.emb_data_path)

        new_def_table = pre_def_table + def_table
        new_def_arged_table = predef_arged_table + def_table_arged
        new_def_emb_table = predef_emb_table + def_table_embed

        new_def_table_path = f"{config.paths.output_data}_Def_{current_time}.jsonl"
        new_def_arged_table_path = f"{config.paths.output_data}_Def_arged_{current_time}.jsonl"
        new_def_emb_table_path = f"{config.paths.output_data}_Def_emb_{current_time}.jsonl"
        
        for table, path in zip([new_def_table, new_def_arged_table, new_def_emb_table], [new_def_table_path, new_def_arged_table_path, new_def_emb_table_path]):
            with open(path, 'w', encoding='utf-8') as f:
                for item in table:
                    f.write(json.dumps(item.to_dict()) + '\n')
        new_tokenizer_path = tokenizer_path.rsplit('_',1)[0] + f'_{current_time}.json'
        tokenizer.save(new_tokenizer_path)

        print(f"New theorem proof generation completed. Output files:")
        print(f"  - {new_def_table_path}")
        print(f"  - {new_def_arged_table_path}")
        print(f"  - {new_def_emb_table_path}")
        print(f"  - {config.paths.tokenizer_path.rsplit('_',1)[0] + f'_{current_time}.json'}")
        print(f"  - proof file did not change, so no output file for proof")

        with open('./config.json', 'r') as f:
            config = json.load(f)
        config['paths']['def_table_path'] = new_def_table_path
        config['paths']['ps_table_path'] = new_def_arged_table_path
        config['paths']['emb_data_path'] = new_def_emb_table_path
        config['paths']['tokenizer_path'] = new_tokenizer_path
        with open('./config.json', 'w') as f:
            json.dump(config, f)
        print(f"Config file updated with new paths.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='data_generation', help='data_generation or new_theorem')
    args = parser.parse_args()
    asyncio.run(main(args))
    asyncio.run(main(args))