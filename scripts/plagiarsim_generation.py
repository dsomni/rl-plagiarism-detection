import argparse
import asyncio
import os
import warnings

import g4f
import nest_asyncio
import numpy as np
import pandas as pd

# from g4f.client import Client
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from tqdm import tqdm

nest_asyncio.apply()

### Constants
API_ATTEMPTS = 3

DEFAULT_NUM = 2
RANDOM_STATE = 42
INITIAL_PERCENT = 0.001

REQUESTS_DELAY = 11

MAX_SYMBOLS = 200

AVAILABLE_MODELS = [
    "openchat_3.5",
    "pi",
    "mixtral-8x7b",
    # "llama2-70b",
    # "gpt-3.5-turbo",
    ## "gpt-3.5-turbo",  # requires google engine
    ## "mixtral-8x7b",
    # # "pi", #requires google engine
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
        Write the answer directly, WITHOUT any annotations, brackets and comments. \
        Let it it be ONLY the paraphrased text in your answer:\n {text}"


def prompt_21(text: str) -> str:
    return f"Write directly in several words the main topic \
        of the following text, without details: \n {text}"


def prompt_22(text: str) -> str:
    return f"Write a short sentence (no more than {MAX_SYMBOLS} symbols) \
        on the following topic: \n {text}"


### Generations


async def generate_type1(text: str, model, num: int, loop: tqdm):
    data_row = {}

    history = []
    for i in range(num):
        answer, history = await access_llm(
            model,
            prompt_1(text, history),
            history,
        )
        data_row[f"{model}_{i+1}"] = answer
        await asyncio.sleep(REQUESTS_DELAY)
        loop.update(1)

    return data_row


async def generate_save_type1(dataset: list[str], save_path: str, num: int, desc: str):
    data = {}

    data["initial"] = []
    for model in AVAILABLE_MODELS:
        for i in range(num):
            data[f"{model}_{i+1}"] = []

    loop = tqdm(
        desc=desc,
        total=len(dataset) * len(AVAILABLE_MODELS) * num,
        disable=not LOGGER.verbose,
        leave=False,
    )

    for i, text in enumerate(dataset):
        data["initial"].append(text)
        raw_data = await asyncio.gather(
            *[generate_type1(text, model, num, loop) for model in AVAILABLE_MODELS]
        )

        for d in raw_data:
            for k, v in d.items():
                data[k].append(v)

        pd.DataFrame(data).to_csv(construct_absolute_path(save_path, f"{i}.csv"))


### Utils
async def access_llm(model: str, content: str, history: list) -> tuple[str, list]:
    for i in range(API_ATTEMPTS):
        try:
            response = await g4f.ChatCompletion.create_async(
                model=model,
                messages=[*history, {"role": "user", "content": content}],
            )

            return response, [
                *history,
                {"role": "user", "content": content},
                {"role": "assistant", "content": response},
            ]
        except Exception:
            print(model, i)
            await asyncio.sleep(REQUESTS_DELAY)
            continue

    return "", history


def soft_make_dir(path: str):
    try:
        os.mkdir(path)
    except:
        pass


def construct_absolute_path(*relative_path: str) -> str:
    """Turn relative file path to absolute

    Raises:
        FileNotFoundError

    Returns:
        str: absolute path
    """
    return os.path.abspath(os.path.join(*relative_path))


async def generate_plagiarism():
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
        default=DEFAULT_NUM,
        action=argparse.BooleanOptionalAction,
        help=f"number of plagiarized versions for 1 data item (default: {DEFAULT_NUM})",
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

    np.random.seed(RANDOM_STATE)
    np.random.shuffle(train_sentences)
    np.random.shuffle(test_sentences)

    train_sentences = train_sentences[: int(len(train_sentences) * INITIAL_PERCENT)]
    test_sentences = test_sentences[: int(len(test_sentences) * INITIAL_PERCENT)]

    LOGGER.log(f"{len(train_sentences)=}")
    LOGGER.log(f"{len(test_sentences)=}")

    # Generate Type 1 Plagiarism
    if plagiarism <= 1:
        train1_path = construct_absolute_path(save_path, "train1")
        soft_make_dir(train1_path)
        await generate_save_type1(
            train_sentences,
            # ["I love chocolate", "Cat eat the mouse", "Elephant is so big in the zoo"],
            train1_path,
            num,
            "Generating Type 1 for train data",
        )
        LOGGER.log("Finish generation Type 1 for train data!")

        test1_path = construct_absolute_path(save_path, "test1")
        soft_make_dir(test1_path)
        await generate_save_type1(
            test_sentences,
            # ["I love chocolate", "Cat eat the mouse", "Elephant is so big in the zoo"],
            test1_path,
            num,
            "Generating Type 1 for test data",
        )

        # LOGGER.log("Finish generation Type 1 for test data!")

    LOGGER.log("Done!")


if __name__ == "__main__":
    asyncio.run(generate_plagiarism())
