from coq_prover.coq_finetune.ft_data_generator import FTDataGenerator
import asyncio

if __name__ == '__main__':
    config_path = './config.json'
    ft_data_generator = FTDataGenerator(config_path=config_path)
    asyncio.run(ft_data_generator.generate_data())