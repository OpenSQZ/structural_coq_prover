import time
import asyncio
from coq_prover.coq_context.proof_generator import ProofGenerator
import sys
import time

class TeeOutput:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.file = open(file_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()

timestr = time.strftime("%Y%m%d_%H")
sys.stdout = TeeOutput('log_'+timestr+'.txt')
sys.stderr = TeeOutput('log_'+timestr+'.txt')

async def generate_mode(config_path, file_path, theorem_name, tactic_list, proof_traces=None, proof_summary=None, public_notes=None, max_retries=None):
    start_time = time.time()
    proof_generator = ProofGenerator(config_path=config_path, generate_mode=True)
    result = await proof_generator.generate_step(
        theorem_name=theorem_name,
        theorem_file_path=file_path,
        tactic_sequence=tactic_list,
        proof_traces=proof_traces,
        proof_summary=proof_summary,
        public_notes=public_notes,
        max_retries=max_retries
    )
    print(f"finish in time {time.time() - start_time}")
    print(result)

async def package_proof_generate(config_path, package_name):
    start_time = time.time()
    proof_generator = ProofGenerator(config_path=config_path)
    result = await proof_generator.run_proof_generation(package_name=package_name)
    print(f"finish in time {time.time() - start_time}")
    print(result)

async def theorem_proof_generate(config_path, file_path, theorem_name):
    start_time = time.time()
    proof_generator = ProofGenerator(config_path=config_path)
    result = await proof_generator.run_proof_generation(theorem_file_path=file_path, theorem_name=theorem_name)
    print(f"finish in time {time.time() - start_time}")
    print(result)

async def theorem_proof_generate_new_theorem_mode(config_path):
    start_time = time.time()
    proof_generator = ProofGenerator(config_path=config_path)
    result = await proof_generator.run_proof_generation()
    print(f"finish in time {time.time() - start_time}")
    print(result)

async def coq_all_proof_generate(config_path, total_shards, shard):
    start_time = time.time()
    proof_generator = ProofGenerator(config_path=config_path)
    await proof_generator.run_proof_generation(package_name="all", ratio=0.1, total_shards=total_shards, shard=shard)
    print(f"finish in time {time.time() - start_time}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Coq Proof Generator - Generate proofs for Coq theorems using ML models')
    parser.add_argument('--mode', type=str, choices=['generate', 'package', 'theorem', 'new_theorem', 'all'], 
                        default='all', help='Proof generation mode')
    parser.add_argument('--shard', type=int, help='Shard number for distributed proof generation')
    parser.add_argument('--config', type=str, default='./config.json', help='Path to configuration file')
    parser.add_argument('--package', type=str, help='Package name for package mode')
    parser.add_argument('--file', type=str, help='File path for theorem mode')
    parser.add_argument('--theorem', type=str, help='Theorem name for theorem mode')
    parser.add_argument('--tactics', nargs='+', help='List of tactics for generate mode')
    args = parser.parse_args()

    config_path = args.config
    from utils import get_config
    config = get_config(config_path)
    print(f'Starting proof generation in {args.mode} mode')
    
    if args.mode == 'generate':
        if not args.file or not args.theorem or not args.tactics:
            raise ValueError('generate mode requires --file, --theorem, and --tactics')
        asyncio.run(generate_mode(config_path, args.file, args.theorem, args.tactics))
        
    elif args.mode == 'package':
        if not args.package:
            raise ValueError('package mode requires --package')
        asyncio.run(package_proof_generate(config_path, args.package))
        
    elif args.mode == 'theorem':
        if not args.file or not args.theorem:
            raise ValueError('theorem mode requires --file and --theorem')
        asyncio.run(theorem_proof_generate(config_path, args.file, args.theorem))
        
    elif args.mode == 'new_theorem':
        asyncio.run(theorem_proof_generate_new_theorem_mode(config_path))
        
    elif args.mode == 'all':
        if args.shard is None:
            raise ValueError('all mode requires --shard for distributed generation')
        
        total_shards = config.params.total_shards
        if config.flags.resume_mode:
            print('Resuming from previous state')
        if config.flags.ablation_mode:
            print('Running in ablation mode')
            
        asyncio.run(coq_all_proof_generate(config_path, total_shards, args.shard))
