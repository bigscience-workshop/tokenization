import json
from argparse import ArgumentParser
import itertools


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--tokenizer-json-path", type=str, help="Path to `tokenizer.json` file")
    return parser.parse_args()

def _remove_replace(data):
    normalizer = data["normalizer"]
    if normalizer["type"] == "Sequence":
        normalizers = normalizer["normalizers"]
        assert len(normalizers) == 2

        updated_normalizers = [elt for elt in normalizers if elt["type"] != "Replace"]

        assert len(updated_normalizers) == 1

        data["normalizer"] = updated_normalizers[0]
        normalizer = data["normalizer"]

    assert normalizer["type"] == "Precompiled"
    return data

def _add_empty_strings(data):
    # Adding spaces to vocabulary
    num_max_spaces = 20
    space_char = "▁"

    if space_char * 2 not in data["model"]["vocab"]:
        offset_idx = len(data["model"]["vocab"]) - 2
        for idx in range(num_max_spaces, 1, -1):
            print(idx + offset_idx, " : ", space_char * idx, " : ", len(space_char * idx))
            data["model"]["vocab"][space_char * idx] = idx + offset_idx

        lines_to_append = []
        for tup in itertools.product([space_char * idx for idx in range(1, num_max_spaces - 1)], repeat=2):
            merge_rule = " ".join(tup)
            if len(merge_rule) < num_max_spaces + 1:
                lines_to_append.append(merge_rule)
        lines_to_append = sorted(lines_to_append, key=lambda x: len(x))

        data["model"]["merges"].extend(lines_to_append)

        # Fixing the whole tokenizer.
        data["normalizer"] = {
            "type": "Sequence",
            "normalizers": [
                data["normalizer"],
                {"type": "Replace", "pattern": {"Regex": "\n"}, "content": "\n "},
                # ^ matches beginning of string as well as beginning of lines in multiline mode.
                {"type": "Replace", "pattern": {"Regex": "^ "}, "content": ""},  # add_prefix_space
                {"type": "Replace", "pattern": {"Regex": "^"}, "content": " "},
                {"type": "Replace", "pattern": {"Regex": "\n "}, "content": "\n"},
                # ^ matches beginning of string as well as beginning of lines in multiline mode.
                {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
            ]}

        data["pre_tokenizer"] = None
        data["decoder"] = {
            "type": "Metaspace",
            "replacement": "▁",
            "add_prefix_space": True
        }
    return data


def main():
    args = get_args()

    with open(args.tokenizer_json_path, "r") as fi:
        data = json.load(fi)

    data = _remove_replace(data)
    data = _add_empty_strings(data)

    with open(args.tokenizer_json_path, "w") as fo:
        json.dump(data, fo, indent=2)


if __name__ == "__main__":
    main()