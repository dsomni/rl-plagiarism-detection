import argparse
import asyncio
import contextlib
import os
import sys
import warnings
from typing import Optional

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
API_ATTEMPTS = 2

DEFAULT_NUM = 2
RANDOM_STATE = 42
INITIAL_PERCENT = 0.001

REQUESTS_DELAY = 11

MAX_SYMBOLS = 150

AVAILABLE_MODELS = [
    "openchat_3.5",
    "pi",  # requires google engine
    "mixtral-8x7b",
    # "llama2-70b",
    # "gpt-3.5-turbo",
    ## "gpt-3.5-turbo",  # requires google engine
    ## "mixtral-8x7b",
    # # "pi", #requires google engine
    # #"airoboros-70b",
]

SKIPPED_MODELS = set()

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
def prompt_1(text: str, history: list = []) -> str:
    prefix = "One more time but in different way " if len(history) != 0 else ""
    return f"{prefix}Paraphrase and reformulate the following text as much as you can \
        keeping the initial idea. Use synonyms. Return it as one text line. \
        Do not make it longer than initial text. \
        Write the answer directly, WITHOUT any annotations, brackets and comments. \
        Let it it be ONLY the paraphrased text in your answer:\n {text}"


def prompt_21(text: str, history: list = []) -> str:
    prefix = "One more time but in different way " if len(history) != 0 else ""
    return f"{prefix}Write directly in several words the main topic \
        of the following text, without details: \n {text}"


def prompt_22(text: str, history: list = []) -> str:
    prefix = "One more time but in different way " if len(history) != 0 else ""
    return f"{prefix}Write a short sentence (no more than {MAX_SYMBOLS} symbols) \
        on the following topic: \n {text}"


### Generations


async def generate_type1(text: str, model, num: int, loop: Optional[tqdm] = None):
    data_row = {}

    history = []
    for i in range(num):
        answer, history = await access_llm(
            model,
            prompt_1(text, history),
            history,
        )
        data_row[f"{model}_{i+1}"] = answer
        if loop is not None:
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

        save_df(pd.DataFrame(data), construct_absolute_path(save_path, f"{i}.csv"))


async def generate_type2(idea: str, model, num: int, loop: Optional[tqdm] = None):
    data_row = {}

    history = []
    for i in range(num):
        answer, history = await access_llm(
            model,
            prompt_22(idea, history),
            history,
        )
        data_row[f"{model}_{i+1}"] = answer
        if loop is not None:
            loop.update(1)

    return data_row


async def generate_save_type2(dataset: list[str], save_path: str, num: int, desc: str):
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

    ideas_data = []
    ideas_columns = ["initial", "idea", "model"]

    model_idx = 0

    for i, text in enumerate(dataset):
        really_available_models = list(
            filter(lambda m: m not in SKIPPED_MODELS, AVAILABLE_MODELS)
        )

        data["initial"].append(text)

        idea, _ = await access_llm(
            really_available_models[model_idx],
            prompt_21(text),
        )
        ideas_data.append((text, idea, really_available_models[model_idx]))

        raw_data = await asyncio.gather(
            *[generate_type2(idea, model, num, loop) for model in AVAILABLE_MODELS]
        )

        for d in raw_data:
            for k, v in d.items():
                data[k].append(v)

        save_df(pd.DataFrame(data), construct_absolute_path(save_path, f"{i}.csv"))
        save_df(
            pd.DataFrame(ideas_data, columns=ideas_columns),
            construct_absolute_path(save_path, f"ideas_{i}.csv"),
        )

        model_idx = (model_idx + 1) % len(really_available_models)


### Utils
async def access_llm(model: str, content: str, history: list = []) -> tuple[str, list]:
    if model in SKIPPED_MODELS:
        return "", history

    for i in range(API_ATTEMPTS):
        try:
            response = await g4f.ChatCompletion.create_async(
                model=model,
                messages=[*history, {"role": "user", "content": content}],
            )

            await asyncio.sleep(REQUESTS_DELAY)

            return response, [
                *history,
                {"role": "user", "content": content},
                {"role": "assistant", "content": response},
            ]
        except Exception:
            print(model, i)
            await asyncio.sleep(REQUESTS_DELAY)
            continue

    SKIPPED_MODELS.add(model)
    return "", history


async def fill_gaps_1(file_path: str, num: int, exit_on_fail: bool = True):
    def get_model(column: str) -> str:
        return "_".join(list(column.split("_"))[:-1])

    async def inner_loop(i_curr: int, num: int) -> tuple[bool, int, dict]:
        for i in range(i_curr, len(df)):
            for j in range(1, len(df.columns)):
                if df.isnull().iloc[i, j]:
                    data_row = await generate_type1(
                        str(df.iloc[i, 0]), get_model(df.columns[j]), num
                    )
                    return False, i, data_row
        return True, i, {}

    LOGGER.log(f"Fixing '{file_path}'...")

    df = pd.read_csv(file_path)

    i_curr = 0
    no_gap = False
    while not no_gap:
        no_gap, i, data_row = await inner_loop(i_curr, num)
        if not no_gap:
            lengths = 1
            for k, v in data_row.items():
                df.loc[i, k] = v
                lengths *= len(v)
            if lengths > 0:
                i_curr = i + 1
                LOGGER.log(f"Done {i}!")
                save_df(df, file_path)
            elif exit_on_fail:
                sys.exit(0)
            else:
                await asyncio.sleep(REQUESTS_DELAY * 2)

    LOGGER.log(f"Done with '{file_path}'\n")


async def fill_gaps_2(
    file_path: str, ideas_path: str, num: int, exit_on_fail: bool = True
):
    def get_model(column: str) -> str:
        return "_".join(list(column.split("_"))[:-1])

    async def inner_loop(
        df: pd.DataFrame, ideas_df: pd.DataFrame, i_curr: int, num: int
    ) -> tuple[bool, int, dict]:
        for i in range(i_curr, len(df)):
            for j in range(1, len(df.columns)):
                if df.isnull().iloc[i, j]:
                    idea = str(ideas_df.iloc[i, 1])

                    data_row = await generate_type2(idea, get_model(df.columns[j]), num)
                    return False, i, data_row
        return True, i, {}

    LOGGER.log(f"Fixing '{file_path}'...")

    df = pd.read_csv(file_path)
    ideas_df = pd.read_csv(ideas_path)

    i_curr = 0
    no_gap = False
    while not no_gap:
        no_gap, i, data_row = await inner_loop(df, ideas_df, i_curr, num)
        if not no_gap:
            lengths = 1
            for k, v in data_row.items():
                df.loc[i, k] = v
                lengths *= len(v)
            if lengths > 0:
                i_curr = i + 1
                LOGGER.log(f"Done {i}!")
                save_df(df, file_path)
            elif exit_on_fail:
                sys.exit(0)
            else:
                await asyncio.sleep(REQUESTS_DELAY * 2)

    LOGGER.log(f"Done with '{file_path}'\n")


def get_max_index(path: str) -> int:
    max_index = -1
    for file in os.listdir(path):
        with contextlib.suppress(ValueError):
            max_index = max(max_index, int(os.path.splitext(file)[0]))
    return max_index


def soft_make_dir(path: str):
    with contextlib.suppress(Exception):
        os.mkdir(path)


def construct_absolute_path(*relative_path: str) -> str:
    """Turn relative file path to absolute

    Raises:
        FileNotFoundError

    Returns:
        str: absolute path
    """
    return os.path.abspath(os.path.join(*relative_path))


def save_df(df: pd.DataFrame, path: str):
    df.loc[:, ~df.columns.str.contains("^Unnamed")].to_csv(path, index=False)


def prepare_sentences(sentences: list[str]) -> list[str]:
    permuted = list(np.random.permutation(sentences))
    return permuted[: int(len(permuted) * INITIAL_PERCENT)]


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
        "-f",
        "--fixing",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="fix the existing files (default: 0 = all)",
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
        fixing,
        save_path,
        data_load_path,
        num,
        ignore_warnings,
        verbose,
    ) = (
        namespace.plagiarism,
        namespace.fixing,
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

    LOGGER.log(f"Train size = {int(len(train_sentences) * INITIAL_PERCENT)}")
    LOGGER.log(f"Test size = {int(len(test_sentences) * INITIAL_PERCENT)}\n")

    # Fixing
    if fixing:
        train1_index = get_max_index(construct_absolute_path(save_path, "train1"))
        test1_index = get_max_index(construct_absolute_path(save_path, "test1"))

        await fill_gaps_1(
            construct_absolute_path(save_path, "train1", f"{train1_index}.csv"),
            num,
        )
        await fill_gaps_1(
            construct_absolute_path(save_path, "test1", f"{test1_index}.csv"),
            num,
        )

        train2_index = get_max_index(construct_absolute_path(save_path, "train2"))
        test2_index = get_max_index(construct_absolute_path(save_path, "test2"))
        await fill_gaps_2(
            construct_absolute_path(save_path, "train2", f"{train2_index}.csv"),
            construct_absolute_path(save_path, "train2", f"ideas_{train2_index}.csv"),
            num,
        )

        await fill_gaps_2(
            construct_absolute_path(save_path, "test2", f"{test2_index}.csv"),
            construct_absolute_path(save_path, "test2", f"ideas_{test2_index}.csv"),
            num,
        )

        return

    # Generate Type 1 Plagiarism
    if plagiarism <= 1:
        np.random.seed(RANDOM_STATE)
        train_sentences1 = prepare_sentences(train_sentences)
        test_sentences1 = prepare_sentences(test_sentences)

        train1_path = construct_absolute_path(save_path, "train1")
        soft_make_dir(train1_path)
        await generate_save_type1(
            train_sentences1,
            train1_path,
            num,
            "Generating Type 1 for train data",
        )
        LOGGER.log("Finish generation Type 1 for train data!")

        test1_path = construct_absolute_path(save_path, "test1")
        soft_make_dir(test1_path)
        await generate_save_type1(
            test_sentences1,
            test1_path,
            num,
            "Generating Type 1 for test data",
        )
        LOGGER.log("Finish generation Type 1 for test data!")

    # Generate Type 2 Plagiarism
    if plagiarism == 0 or plagiarism == 2:
        np.random.seed(RANDOM_STATE * 2)
        train_sentences2 = prepare_sentences(train_sentences)
        test_sentences2 = prepare_sentences(test_sentences)

        train2_path = construct_absolute_path(save_path, "train2")
        soft_make_dir(train2_path)
        await generate_save_type2(
            train_sentences2,
            train2_path,
            num,
            "Generating Type 2 for train data",
        )
        LOGGER.log("Finish generation Type 2 for train data!")

        test2_path = construct_absolute_path(save_path, "test2")
        soft_make_dir(test2_path)
        await generate_save_type2(
            test_sentences2,
            test2_path,
            num,
            "Generating Type 2 for test data",
        )
        LOGGER.log("Finish generation Type 2 for test data!")

    LOGGER.log("Done!")


if __name__ == "__main__":
    asyncio.run(generate_plagiarism())
