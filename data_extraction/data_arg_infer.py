from coq_prover.coq_context.data_arg_api import data_argumentation_async
import argparse
import asyncio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run data argumentation with configurable batch size')
    parser.add_argument('--input_file', type=str, default="/path/to/def_table_train.jsonl",
                        help='Input JSONL file path (default: /path/to/def_table_train.jsonl)')
    parser.add_argument('--batch_size', '--bs', type=int, default=150,
                        help='Batch size for processing (default: 150)')

    args = parser.parse_args()

    print(f"Processing file: {args.input_file}")
    print(f"Batch size: {args.batch_size}")

    asyncio.run(data_argumentation_async(args.input_file, args.batch_size))