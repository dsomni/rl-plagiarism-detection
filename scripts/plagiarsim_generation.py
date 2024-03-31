import argparse
import os
import warnings

import pandas as pd
from g4f.client import Client
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from tqdm import tqdm

### Constants
MAX_SYMBOLS = 200

AVAILABLE_MODELS = [
    # "gpt-3.5-turbo",
    "pi",
    # "llama2-70b",
    # "mixtral-8x7b",
    # #"airoboros-70b",
]

### Class for Logger instance


class Logger:
    """Manage log messages"""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def log(self, message: str):
        """Log message to console

        Args:
            message (str): message to log
        """
        if self.verbose:
            print(message)


LOGGER = Logger()


### Prompts
def prompt_1(text: str, history: list) -> str:
    prefix = "One more time but in different way " if len(history) != 0 else ""
    return f"{prefix}Paraphrase and reformulate the following text as much as you can \
        keeping the initial idea. Use synonyms. Return it as one text line. \
        Do not make it longer than initial text. \
        Write the answer directly, WITHOUT any annotations. \
        Let it it be ONLY the paraphrased version of text in your answer:\n {text}"


def prompt_21(text: str) -> str:
    return f"Write directly in several words the main topic \
        of the following text, without details: \n {text}"


def prompt_22(text: str) -> str:
    return f"Write a short sentence (no more than {MAX_SYMBOLS} symbols) \
        on the following topic: \n {text}"


### Generations


def generate_save_type1(dataset: list[str], save_path: str, num: int, desc: str):
    client = Client()

    data = []
    columns = ["initial"]
    for model in AVAILABLE_MODELS:
        for i in range(num):
            columns.append(f"{model}_{i+1}")

    loop = tqdm(
        desc=desc,
        total=len(dataset) * len(AVAILABLE_MODELS) * num,
        disable=not LOGGER.verbose,
        leave=False,
    )
    for text in dataset:
        data_row = [text]
        for model in AVAILABLE_MODELS:
            history = []
            for _ in range(num):
                answer, history = access_llm(
                    client,
                    model,
                    prompt_1(text, history),
                    history,
                )
                data_row.append(answer)
                loop.update(1)
        data.append(data_row)

    pd.DataFrame(data, columns=columns).to_csv(save_path)


### Utils
def access_llm(client, model: str, content: str, history: list) -> tuple[str, list]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[*history, {"role": "user", "content": content}],
        )

        return response.choices[0].message.content, [
            *history,
            {"role": "user", "content": content},
            {"role": "assistant", "content": response.choices[0].message.content},
        ]
    except:
        pass

    return "", history


def construct_absolute_path(*relative_path: str) -> str:
    """Turn relative file path to absolute

    Raises:
        FileNotFoundError

    Returns:
        str: absolute path
    """
    return os.path.abspath(os.path.join(*relative_path))


def generate_plagiarism():
    """Generate plagiarism"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Plagiarism generation options")
    parser.add_argument(
        "-p",
        "--plagiarism",
        type=int,
        dest="plagiarism",
        choices=[0, 1, 2],
        default="0",
        help="plagiarism types to generate (default: 0 = all)",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        dest="save_path",
        default="./generated",
        help="relative path to save generated results (default: ./generated)",
    )
    parser.add_argument(
        "-d",
        "--data-load-path",
        type=str,
        dest="data_load_path",
        default="./data",
        help="relative path to AG News data (default: ./data)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        dest="num",
        default=2,
        action=argparse.BooleanOptionalAction,
        help="number of plagiarized versions for 1 data item (default: 2)",
    )

    parser.add_argument(
        "-w",
        "--ignore-warnings",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="ignore warnings (default: True)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )

    namespace = parser.parse_args()
    (
        plagiarism,
        save_path,
        data_load_path,
        num,
        ignore_warnings,
        verbose,
    ) = (
        namespace.plagiarism,
        namespace.save_path,
        namespace.data_load_path,
        namespace.num,
        namespace.ignore_warnings,
        namespace.verbose,
    )
    verbose: bool = bool(verbose)
    ignore_warnings: bool = bool(ignore_warnings)

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    # Set up logger
    LOGGER.verbose = verbose

    # Load datasets
    load_path = construct_absolute_path(data_load_path)

    train_dataset = to_map_style_dataset(AG_NEWS(root=load_path, split="train"))
    test_dataset = to_map_style_dataset(AG_NEWS(root=load_path, split="test"))

    train_sentences = [x[1] for x in train_dataset]
    test_sentences = [x[1] for x in test_dataset]

    LOGGER.log(f"{len(train_sentences)=}")
    LOGGER.log(f"{len(test_sentences)=}")

    # Generate Type 1 Plagiarism
    if plagiarism <= 1:
        generate_save_type1(
            # train_dataset[:2],
            ["I love chocolate", "Cat eat the mouse"],
            construct_absolute_path(save_path, "train1.csv"),
            num,
            "Generating Type 1 for train data",
        )
        LOGGER.log("Finish generation Type 1 for train data!")

        # LOGGER.log("Generating Type 1 for test data...")
        # generate_save_type1(
        #     test_dataset, construct_absolute_path(save_path, "test1.csv"), num
        # )
        # LOGGER.log("Finish generation Type 1 for test data!")

    LOGGER.log("Done!")


if __name__ == "__main__":
    generate_plagiarism()
